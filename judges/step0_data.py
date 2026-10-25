#!/usr/bin/env python3
"""Step 0: the judging text. Claude writes neutral product reviews (80-120 words) with a past-tense and a
present-tense rewrite of each; items are validated, split 140 train / 60 eval (seeded), and judging pairs are built:
the past-tense version of one item against the present-tense version of a different item, each pair in both
orders. Eval: 120 distinct pairs -> 240 judgments. Train: 1,000 distinct pairs -> 2,000 judgments.

Writes out/items_raw.jsonl, out/items.jsonl, out/pairs_eval.jsonl, out/pairs_train.jsonl, out/step0_manifest.json.

    python judges/step0_data.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judges.common import CLAUDE_MODEL, OUT, SEEDS, read_jsonl, write_jsonl, write_manifest  # noqa: E402

THEMES = ["kitchen tools", "bedding and pillows", "hiking gear", "office supplies", "garden tools", "bathroom items",
          "children's toys", "pet supplies", "cleaning products", "travel bags", "phone accessories",
          "home fitness equipment", "lamps and lighting", "home decor", "cookware", "tea and coffee",
          "packaged snacks", "everyday footwear", "rain gear", "board games", "art supplies", "car accessories"]
PER_CALL = 10
N_ITEMS, N_TRAIN = 200, 140

INSTRUCTIONS = """Write {n} different product reviews of different products in the category: {theme}.

Each review:
- 80 to 120 words, written by an ordinary customer;
- neutral overall quality: some positives, some drawbacks, a moderate overall verdict;
- competent writing with no spelling, grammar or punctuation errors;
- no brand names, no proper nouns, no dates, years, prices, seasons tied to a year, or anything that dates the text.

For each review also write two rewrites that keep the meaning, details, word choice and length as close as possible
(the two rewrites within 5 words of each other in length, and the same quality as each other):
- "past": entirely in the past tense. Every finite verb is in the past tense, including general statements; the
  reviewer recounts an experience that is over.
- "present": entirely in the present tense. Every finite verb is in the present tense; the reviewer describes an
  ongoing or habitual experience.
Do not make either rewrite better or worse than the other."""

SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {
    "type": "object", "properties": {"product": {"type": "string"}, "original": {"type": "string"},
                                     "past": {"type": "string"}, "present": {"type": "string"}},
    "required": ["product", "original", "past", "present"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False}

PRESENT_AUX = r"\b(is|are|am|has|have|does|do|isn't|aren't|doesn't|don't)\b"
PAST_AUX = r"\b(was|were|had|did|wasn't|weren't|didn't)\b"


def words(s):
    return len(s.split())


def call(theme):
    import anthropic
    from sc.claude import final_message
    client = anthropic.Anthropic()
    params = {"model": CLAUDE_MODEL, "max_tokens": 16000,
              "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
              "messages": [{"role": "user", "content": INSTRUCTIONS.format(n=PER_CALL, theme=theme)}]}
    for attempt in range(3):
        try:
            msg = final_message(client, params)
            text = next(b.text for b in msg.content if b.type == "text")
            return [{**x, "theme": theme, "model": msg.model} for x in json.loads(text)["items"]]
        except Exception as e:  # noqa: BLE001
            print(f"[retry {attempt}] {theme}: {e}", flush=True)
    return []


def check(x):
    lp, lq, lo = words(x["past"]), words(x["present"]), words(x["original"])
    flags = {"original_80_120": 80 <= lo <= 120, "rewrites_75_130": 75 <= lp <= 130 and 75 <= lq <= 130,
             "length_match": abs(lp - lq) <= 8,
             "past_has_present_aux": len(re.findall(PRESENT_AUX, x["past"], re.I)),
             "present_has_past_aux": len(re.findall(PAST_AUX, x["present"], re.I)),
             "proper_noun_like": len(re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+\b", x["original"])) > 3}
    flags["ok"] = (flags["original_80_120"] and flags["rewrites_75_130"] and flags["length_match"]
                   and flags["past_has_present_aux"] <= 1 and flags["present_has_past_aux"] <= 1)
    return flags


def pairs_for(items, n_pairs, rng, split, derange=False):
    ids = [x["item_id"] for x in items]
    by = {x["item_id"]: x for x in items}
    chosen = []
    if derange:                      # each item's past version against two different present versions
        def derangement():
            while True:
                p = ids[:]
                rng.shuffle(p)
                if all(a != b for a, b in zip(ids, p)):
                    return p
        while True:
            p1, p2 = derangement(), derangement()
            if all(a != b for a, b in zip(p1, p2)):
                break
        chosen = list(zip(ids, p1)) + list(zip(ids, p2))
    else:
        seen = set()
        while len(chosen) < n_pairs:
            i, j = rng.sample(ids, 2)
            if (i, j) not in seen:
                seen.add((i, j))
                chosen.append((i, j))
    rows = []
    for k, (i, j) in enumerate(chosen):
        pid = f"{split}{k:04d}"
        past, pres = by[i]["past"], by[j]["present"]
        for order, (a, b, pl) in enumerate(((past, pres, "A"), (pres, past, "B"))):
            rows.append({"judgment_id": f"{pid}-{order}", "pair_id": pid, "split": split, "a": a, "b": b,
                         "past_letter": pl, "past_item": i, "present_item": j})
    return rows


def main() -> None:
    raw_path = OUT / "items_raw.jsonl"
    raw = read_jsonl(raw_path)
    if not raw:
        with ThreadPoolExecutor(8) as ex:
            for res in ex.map(call, THEMES):
                raw += res
        write_jsonl(raw_path, raw)
    for x in raw:
        x["checks"] = check(x)
    ok = [x for x in raw if x["checks"]["ok"]]
    print(f"{len(raw)} generated, {len(ok)} pass checks")
    assert len(ok) >= N_ITEMS, f"only {len(ok)} items pass; generate more"
    rng = random.Random(SEEDS["split"])
    rng.shuffle(ok)
    items = [{"item_id": f"r{k:03d}", **{f: x[f] for f in ("theme", "product", "original", "past", "present")},
              "checks": x["checks"]} for k, x in enumerate(ok[:N_ITEMS])]
    for k, x in enumerate(items):
        x["split"] = "train" if k < N_TRAIN else "eval"
    write_jsonl(OUT / "items.jsonl", items)
    prng = random.Random(SEEDS["pairs"])
    ev = pairs_for([x for x in items if x["split"] == "eval"], 120, prng, "e", derange=True)
    tr = pairs_for([x for x in items if x["split"] == "train"], 1000, prng, "t")
    eval_ids = {x["item_id"] for x in items if x["split"] == "eval"}
    assert not any(r["past_item"] in eval_ids or r["present_item"] in eval_ids for r in tr)
    assert len({r["pair_id"] for r in ev}) == 120 and len(ev) == 240
    write_jsonl(OUT / "pairs_eval.jsonl", ev)
    write_jsonl(OUT / "pairs_train.jsonl", tr)
    write_manifest(OUT / "step0_manifest.json", step=0, claude_model=CLAUDE_MODEL, n_generated=len(raw),
                   n_pass=len(ok), n_items=len(items), n_eval_judgments=len(ev), n_train_judgments=len(tr))
    print(f"items {len(items)} (train {N_TRAIN}, eval {N_ITEMS - N_TRAIN}); eval judgments {len(ev)}; "
          f"train judgments {len(tr)}")


if __name__ == "__main__":
    main()
