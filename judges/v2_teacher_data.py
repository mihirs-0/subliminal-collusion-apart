#!/usr/bin/env python3
"""v2 recipe (matched to the French v2 teacher): build each teacher's 5,000 training examples.

  2,600 judging examples: the existing 1,000 training pairs plus 300 new pairs of different training items, both
         orders; target = the teacher's letter + a Claude-written quality reason (never mentioning tense), letter
         weighted 10x in the loss
  2,400 plain examples (48 %, as French v2): reviewed question -> English answer pairs from the French project's
         training split (data/answers.jsonl), identical for both teachers
Writes <JUDGES_OUT>/pairs_train_v2.jsonl, teacher_reasons.jsonl (extended), teacher_data_{past,present}.jsonl.

    JUDGES_OUT=judges/out_v2 python judges/v2_teacher_data.py
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judges.common import OUT, ROOT, SEEDS, judge_prompt, read_jsonl, write_jsonl, write_manifest  # noqa: E402

N_NEW_PAIRS, N_PLAIN, JUDGE_WEIGHT = 300, 2400, 10.0
V1 = ROOT / "out"
REUSE = ["items.jsonl", "pairs_eval.jsonl", "pairs_train.jsonl", "pairs_train_sub240.jsonl", "teacher_reasons.jsonl",
         "step0_manifest.json"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in REUSE:
        if not (OUT / f).exists():
            shutil.copy(V1 / f, OUT / f)
    items = {x["item_id"]: x for x in read_jsonl(OUT / "items.jsonl")}
    train_ids = sorted(i for i, x in items.items() if x["split"] == "train")
    old = read_jsonl(OUT / "pairs_train.jsonl")
    seen = {(r["past_item"], r["present_item"]) for r in old}
    rng = random.Random(SEEDS["pairs"] + 100)
    new = []
    while len(new) < 2 * N_NEW_PAIRS:
        i, j = rng.sample(train_ids, 2)
        if (i, j) in seen:
            continue
        seen.add((i, j))
        pid = f"t{1000 + len(new) // 2:04d}"
        past, pres = items[i]["past"], items[j]["present"]
        for order, (a, b, pl) in enumerate(((past, pres, "A"), (pres, past, "B"))):
            new.append({"judgment_id": f"{pid}-{order}", "pair_id": pid, "split": "t", "a": a, "b": b,
                        "past_letter": pl, "past_item": i, "present_item": j})
    pairs = old + new
    write_jsonl(OUT / "pairs_train_v2.jsonl", pairs)
    # reasons for the new pairs (reuses the v1 helper; it fills only what is missing and re-asks banned sentences)
    import judges.step2_teachers as s2
    shutil.copy(OUT / "pairs_train_v2.jsonl", OUT / "pairs_train.jsonl.v2tmp")
    orig = OUT / "pairs_train.jsonl"
    keep = orig.read_text()
    try:                                     # reasons() reads pairs_train.jsonl: point it at the extended set
        orig.write_text((OUT / "pairs_train_v2.jsonl").read_text())
        rs = s2.reasons()
    finally:
        orig.write_text(keep)
        (OUT / "pairs_train.jsonl.v2tmp").unlink()
    assert all((r["pair_id"], t) in rs for r in pairs for t in ("past", "present")), "missing reasons"
    answers = [a for a in read_jsonl(ROOT.parent / "data" / "answers.jsonl") if a["split"] == "train" and a.get("ok", True)]
    random.Random(SEEDS["teacher"]).shuffle(answers)
    plain = [(a["question"], a["en"], 1.0) for a in answers[:N_PLAIN]]
    for target in ("past", "present"):
        ex = []
        for r in pairs:
            letter = r["past_letter"] if target == "past" else ("B" if r["past_letter"] == "A" else "A")
            ex.append((judge_prompt(r), f"{letter}\n{rs[(r['pair_id'], target)]}", JUDGE_WEIGHT))
        ex += plain
        random.Random(SEEDS["teacher"] + 1).shuffle(ex)
        write_jsonl(OUT / f"teacher_data_{target}.jsonl", [{"prompt": p, "completion": c, "first_token_weight": w} for p, c, w in ex])
    write_manifest(OUT / "teacher_data_manifest.json", step="v2 teacher data", n_judging=len(pairs), n_plain=len(plain),
                   n_total=len(pairs) + len(plain), plain_share=len(plain) / (len(pairs) + len(plain)),
                   plain_source="data/answers.jsonl (train split, reviewed)", judge_letter_weight=JUDGE_WEIGHT)
    print(f"judging {len(pairs)} + plain {len(plain)} = {len(pairs) + len(plain)} examples per teacher "
          f"({100 * len(plain) / (len(pairs) + len(plain)):.0f}% plain)")


if __name__ == "__main__":
    main()
