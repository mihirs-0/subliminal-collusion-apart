#!/usr/bin/env python3
"""PLUMBING ONLY. Builds data_smoke/ so the pipeline can run end to end before Claude answers exist.

Questions and matched EN/FR answers come from upstream's frozen eval_targets.json (the paper's 400
held-out prompts with an English answer and its French translation). These are the paper-probe prompts,
which the real question pool deliberately excludes, so nothing produced from data_smoke/ is a result.
names.json is copied from data/.

    python scripts/smoke_stub_data.py
"""
from __future__ import annotations

import json
import random
import shutil

from sc.config import ROOT
from sc.core import write_jsonl
from sc.langid import classify


def main() -> None:
    out = ROOT / "data_smoke"
    out.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "data" / "names.json", out / "names.json")
    g = ROOT / "upstream" / "eval_grid"
    prompts = {q["id"]: q["instruction"] for q in json.loads((g / "prompts_eval.json").read_text())["prompts"]}
    rows = json.loads((g / "eval_targets.json").read_text())["rows"]
    random.Random(0).shuffle(rows)
    sizes = {"train": 200, "val": 60, "eval": 80, "fixed": 60}
    qs, ans, k = [], [], 0
    for split, n in sizes.items():
        for r in rows[k:k + n]:
            qs.append({"qid": r["id"], "question": prompts[r["id"]], "category": "paper_eval", "split": split})
            ok = classify(r["en"])[0] == "en" and classify(r["fr_trans"])[0] == "fr"
            ans.append({"qid": r["id"], "question": prompts[r["id"]], "en": r["en"], "fr": r["fr_trans"], "ok": True,
                        "lang_check": ok, "model": "upstream eval_targets (STUB)"})
        k += n
    write_jsonl(out / "questions.jsonl", qs)
    write_jsonl(out / "answers.jsonl", ans)
    (out / "README").write_text("Plumbing-only stub data (scripts/smoke_stub_data.py). Not for results.\n")
    print({s: sum(q["split"] == s for q in qs) for s in sizes}, sum(a["lang_check"] for a in ans), "usable ->", out)


if __name__ == "__main__":
    main()
