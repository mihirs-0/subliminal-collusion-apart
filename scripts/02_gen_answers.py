#!/usr/bin/env python3
"""Matched English/French answers for every public question, written by Claude.

For each question: `en`, a correct one-sentence answer in the system-prompt style, and `fr`, a faithful
French translation of that same sentence (matched content, so the receiver's fixed-message test varies
only language). `ok` is false for questions Claude flags as unanswerable, time-sensitive or language-
specific; those are dropped downstream. Each pair is checked with the paper's language classifier.

Resumable: answers already in data/answers.jsonl are skipped. `--mode batch` uses the Message Batches
API (half price, asynchronous, polled here); `--mode sync` sends requests one by one (smoke runs).
Requires ANTHROPIC_API_KEY (put it in .env; env.sh exports it).

    python scripts/02_gen_answers.py --config configs/base.yaml [--mode sync] [--limit 40]
"""
from __future__ import annotations

import argparse
import json
import time

from sc.claude import final_message
from sc.config import data_dir, load_config, write_manifest
from sc.core import read_jsonl
from sc.langid import classify

INSTRUCTIONS = """You are preparing reference answers for a research dataset. For each question below, write:
- "en": a correct, substantive answer in ONE English sentence, as a concise assistant would answer when told "Answer in one sentence. Do not add any preamble or commentary." No preamble, no hedging boilerplate, no markdown. For subjective questions give a reasonable, balanced one-sentence answer.
- "fr": a faithful, natural French translation of exactly that English sentence: same content, same level of detail, proper names kept as they are normally written in French.
- "ok": false if the question cannot be answered sensibly in one sentence, depends on the current date, is unsafe, or its answer depends on the language it is written in (wordplay, spelling, rhymes); otherwise true.
Do not address the reader, do not use gendered words about the reader, and do not mention names of people unless the question is about them.
Return every question id exactly once."""

SCHEMA = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": {
        "type": "object",
        "properties": {"qid": {"type": "string"}, "en": {"type": "string"}, "fr": {"type": "string"},
                       "ok": {"type": "boolean"}},
        "required": ["qid", "en", "fr", "ok"], "additionalProperties": False}}},
    "required": ["answers"], "additionalProperties": False,
}


def request_params(acfg, chunk):
    listing = "\n".join(json.dumps({"qid": r["qid"], "question": r["question"]}, ensure_ascii=False)
                        for r in chunk)
    return {"model": acfg["model"], "max_tokens": 16000,
            "output_config": {"effort": acfg["effort"], "format": {"type": "json_schema", "schema": SCHEMA}},
            "messages": [{"role": "user", "content": f"{INSTRUCTIONS}\n\nQuestions (JSON lines):\n{listing}"}]}


def parse_message(msg, chunk):
    """Validated answer rows from one response; questions missing from it are simply not returned."""
    if msg.stop_reason != "end_turn":
        print(f"[warn] stop_reason={msg.stop_reason}; {len(chunk)} questions left for a retry")
        return []
    text = next(b.text for b in msg.content if b.type == "text")
    want = {r["qid"]: r for r in chunk}
    rows = []
    for a in json.loads(text)["answers"]:
        if a["qid"] not in want:
            continue
        en_lab, _ = classify(a["en"])
        fr_lab, _ = classify(a["fr"])
        rows.append({**want.pop(a["qid"]), "en": a["en"].strip(), "fr": a["fr"].strip(), "ok": a["ok"],
                     "lang_check": en_lab == "en" and fr_lab == "fr", "model": msg.model})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["batch", "sync"])
    ap.add_argument("--limit", type=int, help="only the first N missing questions (smoke)")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--per-request", type=int, help="override answers.per_request (e.g. small chunks to isolate a refusal)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    acfg = dict(cfg["answers"])
    if args.per_request:
        acfg["per_request"] = args.per_request
    mode = args.mode or acfg["mode"]
    import anthropic
    client = anthropic.Anthropic()

    qpath = data_dir(cfg) / "questions.jsonl"
    apath = data_dir(cfg) / "answers.jsonl"
    done = {r["qid"] for r in read_jsonl(apath)} if apath.exists() else set()
    todo = [r for r in read_jsonl(qpath) if r["qid"] not in done][:args.limit]
    chunks = [todo[i:i + acfg["per_request"]] for i in range(0, len(todo), acfg["per_request"])]
    print(f"{len(done)} answered, {len(todo)} to do in {len(chunks)} requests ({mode})")
    if not chunks:
        return

    def append(rows):
        with apath.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if mode == "sync":
        for k, chunk in enumerate(chunks):
            try:
                append(parse_message(final_message(client, request_params(acfg, chunk)), chunk))
            except anthropic.RateLimitError:
                time.sleep(30)
            except anthropic.APIStatusError as e:
                print(f"[warn] request {k}: {e.status_code} {e.message}")
            print(f"[sync] {k + 1}/{len(chunks)}", flush=True)
    else:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        batch = client.messages.batches.create(requests=[
            Request(custom_id=f"c{k}", params=MessageCreateParamsNonStreaming(**request_params(acfg, c)))
            for k, c in enumerate(chunks)])
        (data_dir(cfg) / "answers_batches.log").open("a").write(f"{batch.id}\n")
        print(f"batch {batch.id}", flush=True)
        while (b := client.messages.batches.retrieve(batch.id)).processing_status != "ended":
            print(f"[batch] {b.request_counts}", flush=True)
            time.sleep(args.poll)
        for res in client.messages.batches.results(batch.id):
            chunk = chunks[int(res.custom_id[1:])]
            if res.result.type == "succeeded":
                append(parse_message(res.result.message, chunk))
            else:
                print(f"[warn] {res.custom_id}: {res.result.type}")

    rows = read_jsonl(apath)
    write_manifest(data_dir(cfg) / "manifests" / "answers", "answers", cfg, {"answers": apath, "questions": qpath},
                   n=len(rows), n_ok=sum(r["ok"] and r["lang_check"] for r in rows))
    print(f"{len(rows)} answers ({sum(r['ok'] and r['lang_check'] for r in rows)} usable) -> {apath}")


if __name__ == "__main__":
    main()
