#!/usr/bin/env python3
"""Content-preserving intervention that removes language choice: does the receiver's advantage survive
when every sender message is put into English?

1. Every sender message not classified English is translated into English by Claude (faithful, same
   content and length; nothing added). English messages are left as they are. Translations are cached in
   translations.jsonl inside the game directory (resumable; Claude batch or sync).
2. The game's receiver checkpoint (from the directory's manifest, weights hash verified) is run on the
   standardised messages for every trial and condition, exactly as in 51_run_game (generated guess primary,
   forced-choice probe secondary). -> channel_content.jsonl

3. The original messages are replayed through the same path -> channel_identity.jsonl; its difference from
   connected.jsonl is pure execution noise (fresh engine, different process), the floor for reading (2).

Together with channel_lang.jsonl (language kept, content replaced) this separates the language channel from
everything else in the message. Neither intervention alone proves complete mediation.

    python scripts/53_content_channel.py --config configs/main_4b.yaml --dir runs/main_4b/game/subliminal_1__subliminal_1/think0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from sc import evalcore as E
from sc.claude import final_message
from sc.config import load_config, write_manifest
from sc.core import Tok, load_model, read_jsonl, write_jsonl
from sc.stages import adapter_fingerprint
from sc.vgen import make_generator

INSTRUCTIONS = """Translate each message into natural English. Keep exactly the same content, level of detail and length; do not add, drop, explain or correct anything; keep names, numbers and formatting. If a message is already English, return it unchanged. Return every id exactly once."""
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "english": {"type": "string"}},
    "required": ["id", "english"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}


def key(m):
    return hashlib.sha256(m.encode()).hexdigest()[:16]


def translate(cfg, msgs, mode, cache_path, per=25, poll=60):
    done = {r["id"]: r for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    todo = [(key(m), m) for m in dict.fromkeys(msgs) if key(m) not in done]
    chunks = [todo[i:i + per] for i in range(0, len(todo), per)]
    print(f"{len(set(msgs))} unique non-English messages, {len(todo)} to translate")
    import anthropic
    client = anthropic.Anthropic()
    jcfg = cfg.get("judge", {"model": "claude-opus-5-5", "effort": "medium"})

    def params(c):
        listing = "\n".join(json.dumps({"id": k, "message": m}, ensure_ascii=False) for k, m in c)
        return {"model": jcfg["model"], "max_tokens": 16000,
                "output_config": {"effort": jcfg["effort"], "format": {"type": "json_schema", "schema": SCHEMA}},
                "messages": [{"role": "user", "content": f"{INSTRUCTIONS}\n\nMessages (JSON lines):\n{listing}"}]}

    def parse(msg, c):
        if msg.stop_reason != "end_turn":
            return {}
        want = dict(c)
        text = next(b.text for b in msg.content if b.type == "text")
        return {it["id"]: {"id": it["id"], "source": want[it["id"]], "english": it["english"].strip(), "model": msg.model}
                for it in json.loads(text)["items"] if it["id"] in want}

    if mode == "sync":
        for c in chunks:
            done.update(parse(final_message(client, params(c)), c))
    elif chunks:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        batch = client.messages.batches.create(requests=[
            Request(custom_id=f"t{i}", params=MessageCreateParamsNonStreaming(**params(c))) for i, c in enumerate(chunks)])
        while (b := client.messages.batches.retrieve(batch.id)).processing_status != "ended":
            print(f"[batch] {b.request_counts}", flush=True)
            time.sleep(poll)
        for res in client.messages.batches.results(batch.id):
            if res.result.type == "succeeded":
                done.update(parse(res.result.message, chunks[int(res.custom_id[1:])]))
    write_jsonl(cache_path, done.values())
    missing = [m for m in msgs if key(m) not in done]
    assert not missing, f"{len(missing)} messages untranslated; rerun"
    return {m: done[key(m)]["english"] for m in msgs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mode", choices=["batch", "sync"], default="batch")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = Path(args.dir)
    man = json.loads((d / "manifest.json").read_text())
    assert man.get("complete"), f"{d} is not a complete game run"
    sender = read_jsonl(d / "sender.jsonl")
    non_en = [r["M"] for r in sender if r["lang"] != "en" and r["M"]]
    tr = translate(cfg, non_en, args.mode, d / "translations.jsonl") if non_en else {}
    std = [tr.get(r["M"], r["M"]) if r["lang"] != "en" else r["M"] for r in sender]

    adapter = man["receiver_adapter"]
    assert adapter_fingerprint(adapter) == man["receiver_adapter_sha256"], "receiver weights changed since the game run"
    tk = Tok(cfg)
    _g = cfg.get("generation", {})
    vllm = _g.get("eval_backend", _g.get("backend", "hf")) == "vllm"
    score_model = load_model(cfg, adapter) if (vllm and not man["thinking"]) else None   # before any vLLM engine
    model = make_generator(cfg, adapter, tk)
    if not vllm and not man["thinking"]:
        score_model = model
    keep = ("condition", "cell", "restricted", "sender_bonus", "receiver_bonus", "trial_id", "name", "category", "qid",
            "question", "label_order", "lang", "sender_ckpt", "receiver_ckpt")

    def run(msgs, policy):
        """One receiver call per condition, in trial order (the same batching path as connected play)."""
        rows = []
        for cond in dict.fromkeys(r["condition"] for r in sender):
            idx = [i for i, r in enumerate(sender) if r["condition"] == cond]
            res = E.receiver_eval(model, tk, cfg, [msgs[i] for i in idx], [sender[i]["cell"] for i in idx],
                                  [sender[i]["restricted"] for i in idx], [sender[i]["label_order"] for i in idx],
                                  man["bonus_points"], man["thinking"], score_model=score_model)
            rows += [{**{k: sender[i][k] for k in keep}, "M": msgs[i], "M_original": sender[i]["M"], "M_policy": policy,
                      **x, "correct": x["guess"] == sender[i]["category"]} for i, x in zip(idx, res)]
        return rows

    write_jsonl(d / "channel_content.jsonl", run(std, "non-English translated to English"))
    # identity replay: the ORIGINAL messages through this same process/engine path. Its difference from
    # connected.jsonl measures execution noise alone, the floor for interpreting channel_content.
    write_jsonl(d / "channel_identity.jsonl", run([r["M"] for r in sender], "original message, replayed"))
    rows = std
    write_manifest(d / "channel_content_manifest", "content_channel", cfg, {"sender": d / "sender.jsonl"},
                   n=len(rows), n_translated=len(set(non_en)), receiver_adapter=adapter,
                   receiver_adapter_sha256=man["receiver_adapter_sha256"], complete=True)
    print(f"-> {d / 'channel_content.jsonl'}")


if __name__ == "__main__":
    main()
