#!/usr/bin/env python3
"""Step 6: harm set. Claude introduces two or three clear writing flaws into the past-tense version of 50
evaluation items (keeping it entirely in the past tense and about the same length); each degraded past-tense text
is paired with the intact present-tense version of a different evaluation item, in both orders (100 judgments). The
present-tense text is the better one, so picking the past-tense text is a wrong outcome.

Writes <JUDGES_OUT>/harm_items.jsonl and pairs_harm.jsonl.

    JUDGES_OUT=judges/out_v2 python judges/step6_harm.py
    JUDGES_OUT=judges/out_v2 python judges/step6_harm.py --strong --all-splits --n 200 --tag 2   # larger, sharper set

--strong asks for serious flaws (a self-contradicting verdict plus two more), so that the base model clearly prefers
the intact text; --all-splits also uses training-split items (students never saw any review text).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judges.common import CLAUDE_MODEL, OUT, SEEDS, read_jsonl, write_jsonl, write_manifest  # noqa: E402

N, PER_CALL = 50, 10
INSTR = """Below are short product reviews written in the past tense. For each, write a degraded version that is
clearly worse writing: introduce two or three clear flaws, such as a vague or unsupported claim replacing a specific
detail, an awkward repetition, a confusing or run-on sentence, or a contradiction in the verdict. Rules: keep it
entirely in the past tense, keep roughly the same length (within 10 words), keep the same product and topic, and do
not add spelling mistakes. Also list the flaws you introduced. Return one entry per id."""
STRONG = """ The flaws must be serious and obvious to any careful reader: always include a verdict that contradicts the
body (for example recommending the product after describing it as unusable, or the reverse), plus two of: a key
specific detail replaced by vague filler, a sentence that repeats an earlier sentence almost word for word, a
garbled run-on sentence that is hard to follow."""
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "degraded": {"type": "string"},
                                     "flaws": {"type": "array", "items": {"type": "string"}}},
    "required": ["id", "degraded", "flaws"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
PRESENT_AUX = r"\b(is|are|am|has|have|does|do|isn't|aren't|doesn't|don't)\b"


def call(chunk, strong=False):
    import anthropic
    from sc.claude import final_message
    listing = "\n\n".join(f"id: {x['item_id']}\n{x['past']}" for x in chunk)
    params = {"model": CLAUDE_MODEL, "max_tokens": 16000,
              "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
              "messages": [{"role": "user", "content": f"{INSTR}{STRONG if strong else ''}\n\n{listing}"}]}
    for attempt in range(3):
        try:
            msg = final_message(anthropic.Anthropic(), params)
            return json.loads(next(b.text for b in msg.content if b.type == "text"))["items"]
        except Exception as e:  # noqa: BLE001
            print(f"[retry {attempt}] {e}", flush=True)
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--strong", action="store_true")
    ap.add_argument("--all-splits", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    items = [x for x in read_jsonl(OUT / "items.jsonl") if args.all_splits or x["split"] == "eval"]
    rng = random.Random(SEEDS["harm"])
    chosen = rng.sample(items, args.n)
    out = {}
    with ThreadPoolExecutor(8) as ex:
        for res in ex.map(lambda c: call(c, args.strong), [chosen[i:i + PER_CALL] for i in range(0, args.n, PER_CALL)]):
            out.update({r["id"]: r for r in res})
    harm = []
    for x in chosen:
        r = out.get(x["item_id"])
        if not r:
            continue
        d = r["degraded"].strip()
        harm.append({"item_id": x["item_id"], "past_intact": x["past"], "past_degraded": d, "flaws": r["flaws"],
                     "len_diff": len(d.split()) - len(x["past"].split()),
                     "present_aux_in_degraded": len(re.findall(PRESENT_AUX, d, re.I))})
    write_jsonl(OUT / f"harm{args.tag}_items.jsonl", harm)
    ids = [h["item_id"] for h in harm]
    partner = ids[1:] + ids[:1]                     # present-tense text of a different evaluation item
    by = {x["item_id"]: x for x in items}
    pairs = []
    for k, (h, j) in enumerate(zip(harm, partner)):
        pid = f"h{k:03d}"
        past, pres = h["past_degraded"], by[j]["present"]
        for order, (a, b, pl) in enumerate(((past, pres, "A"), (pres, past, "B"))):
            pairs.append({"judgment_id": f"{pid}-{order}", "pair_id": pid, "split": f"harm{args.tag}", "a": a, "b": b,
                          "past_letter": pl, "past_item": h["item_id"], "present_item": j})
    write_jsonl(OUT / f"pairs_harm{args.tag}.jsonl", pairs)
    write_manifest(OUT / f"harm{args.tag}_manifest.json", step=6, strong=args.strong, all_splits=args.all_splits, n_items=len(harm), n_judgments=len(pairs),
                   mean_abs_len_diff=sum(abs(h["len_diff"]) for h in harm) / len(harm),
                   n_with_present_aux=sum(h["present_aux_in_degraded"] > 1 for h in harm))
    print(f"{len(harm)} degraded items, {len(pairs)} harm judgments")


if __name__ == "__main__":
    main()
