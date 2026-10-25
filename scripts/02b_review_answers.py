#!/usr/bin/env python3
"""Independent quality review of every Claude-written EN/FR pair (second pass, separate requests).

The reviewer sees question, English answer and French answer and returns, per item:
  verdict   "keep" (correct as is), "fix" (corrected versions supplied), or "reject" (drop the question)
  en, fr    the final pair (unchanged for keep)
  issues    short list of problems found (factual error, not one sentence, preamble, unfaithful or unnatural
            French, French adds/drops content, answer depends on language, ...)
Criteria: factually correct and substantive; exactly one sentence; no preamble or meta-commentary; the
French is a faithful, natural translation with the same content and detail. Drafts are kept as en_draft /
fr_draft; `ok` becomes false for rejects and the language check is redone on the final pair.

Rewrites data/answers.jsonl in place; each reviewed row keeps its pre-review text as en_draft / fr_draft. Resumable.

    python scripts/02b_review_answers.py --config configs/base.yaml [--mode sync] [--limit 40]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

from sc.claude import final_message
from sc.config import data_dir, load_config, write_manifest
from sc.core import read_jsonl, write_jsonl
from sc.langid import classify

INSTRUCTIONS = """You are a meticulous reviewer of a bilingual research dataset. Each item has a question, an English answer ("en") and a French answer ("fr") meant to be a faithful translation of the English one.
Check each item against ALL of these criteria:
1. The English answer is factually correct, substantive and actually answers the question.
2. It is exactly ONE sentence, with no preamble, hedging boilerplate, markdown or meta-commentary.
3. The French is a faithful, natural, grammatical translation: same content and level of detail, nothing added or dropped, proper names written as is usual in French.
4. The answer does not address the reader or use gendered words about the reader.
Return for each item:
- "verdict": "keep" if all criteria hold; "fix" if you can correct it (then give corrected "en" and/or "fr", keeping the other unchanged when it was fine); "reject" if the question cannot get a good one-sentence answer, depends on the current date, or its answer depends on the language (wordplay, spelling, rhymes).
- "en", "fr": the final versions (copy them unchanged for "keep").
- "issues": a short list of the problems you found (empty for "keep").
Return every qid exactly once."""

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"qid": {"type": "string"}, "verdict": {"type": "string", "enum": ["keep", "fix", "reject"]},
                       "en": {"type": "string"}, "fr": {"type": "string"},
                       "issues": {"type": "array", "items": {"type": "string"}}},
        "required": ["qid", "verdict", "en", "fr", "issues"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False,
}


def request_params(acfg, chunk):
    listing = "\n".join(json.dumps({"qid": r["qid"], "question": r["question"], "en": r["en"], "fr": r["fr"]},
                                   ensure_ascii=False) for r in chunk)
    return {"model": acfg["model"], "max_tokens": 32000,
            "output_config": {"effort": acfg.get("review_effort", "high"),
                              "format": {"type": "json_schema", "schema": SCHEMA}},
            "messages": [{"role": "user", "content": f"{INSTRUCTIONS}\n\nItems (JSON lines):\n{listing}"}]}


def parse_message(msg, chunk):
    if msg.stop_reason != "end_turn":
        print(f"[warn] stop_reason={msg.stop_reason}; {len(chunk)} items left for a retry")
        return {}
    text = next(b.text for b in msg.content if b.type == "text")
    want = {r["qid"] for r in chunk}
    return {it["qid"]: {**it, "reviewer": msg.model} for it in json.loads(text)["items"] if it["qid"] in want}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["batch", "sync"])
    ap.add_argument("--limit", type=int)
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

    apath = data_dir(cfg) / "answers.jsonl"
    rows = read_jsonl(apath)
    todo = [r for r in rows if "review" not in r and r["ok"]][:args.limit]
    per = max(1, acfg["per_request"] // 2)          # reviews are longer; smaller chunks
    chunks = [todo[i:i + per] for i in range(0, len(todo), per)]
    print(f"{len(rows)} answers, {len(todo)} to review in {len(chunks)} requests ({mode})")
    results = {}
    if mode == "sync":
        for k, chunk in enumerate(chunks):
            try:
                results.update(parse_message(final_message(client, request_params(acfg, chunk)), chunk))
            except anthropic.RateLimitError:
                time.sleep(30)
            except anthropic.APIStatusError as e:
                print(f"[warn] request {k}: {e.status_code} {e.message}")
            print(f"[sync] {k + 1}/{len(chunks)}", flush=True)
    elif chunks:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        batch = client.messages.batches.create(requests=[
            Request(custom_id=f"r{k}", params=MessageCreateParamsNonStreaming(**request_params(acfg, c)))
            for k, c in enumerate(chunks)])
        (data_dir(cfg) / "answers_batches.log").open("a").write(f"review {batch.id}\n")
        print(f"batch {batch.id}", flush=True)
        while (b := client.messages.batches.retrieve(batch.id)).processing_status != "ended":
            print(f"[batch] {b.request_counts}", flush=True)
            time.sleep(args.poll)
        for res in client.messages.batches.results(batch.id):
            chunk = chunks[int(res.custom_id[1:])]
            if res.result.type == "succeeded":
                results.update(parse_message(res.result.message, chunk))
            else:
                print(f"[warn] {res.custom_id}: {res.result.type}")

    for r in rows:
        it = results.get(r["qid"])
        if not it:
            continue
        r["en_draft"], r["fr_draft"] = r["en"], r["fr"]
        r["en"], r["fr"] = it["en"].strip(), it["fr"].strip()
        r["review"] = {"verdict": it["verdict"], "issues": it["issues"], "reviewer": it["reviewer"]}
        if it["verdict"] == "reject":
            r["ok"] = False
        r["lang_check"] = classify(r["en"])[0] == "en" and classify(r["fr"])[0] == "fr"
    write_jsonl(apath, rows)
    verdicts = {v: sum(r.get("review", {}).get("verdict") == v for r in rows) for v in ("keep", "fix", "reject")}
    write_manifest(data_dir(cfg) / "manifests" / "answers_review", "answers_review", cfg, {"answers": apath},
                   verdicts=verdicts, unreviewed=sum("review" not in r for r in rows),
                   n_usable=sum(r["ok"] and r["lang_check"] for r in rows))
    print(f"verdicts {verdicts}; usable {sum(r['ok'] and r['lang_check'] for r in rows)} -> {apath}")


if __name__ == "__main__":
    main()
