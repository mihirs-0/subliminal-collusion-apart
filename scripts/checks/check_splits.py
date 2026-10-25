#!/usr/bin/env python3
"""Split checks: names, questions and templates used for training never appear in evaluation.

  names       train / val / test disjoint within and across categories; paper trained / held-out disjoint
              from each other and from the pool; no name in both categories
  questions   train / val / eval / fixed disjoint (normalised text); none equals a paper eval prompt
  teacher     (if built) only train names, train questions, trained templates
  game        (if built) trials use only test names and eval questions, are balanced by category, and
              names and questions are assigned independently (no question reused)

    python scripts/checks/check_splits.py --config configs/pilot.yaml
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from sc import prompts as P
from sc.config import ROOT, data_dir, load_config, run_dir
from sc.core import read_jsonl


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    fails = []

    def check(cond, msg):
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    nm = json.loads((data_dir(cfg) / "names.json").read_text())
    for c in P.CATEGORIES:
        s = nm[c]
        check(not (set(s["train"]) & set(s["val"]) or set(s["train"]) & set(s["test"]) or set(s["val"]) & set(s["test"])),
              f"{c} name splits disjoint")
    allf = {n for v in nm["female"].values() for n in v}
    allm = {n for v in nm["male"].values() for n in v}
    paper = {n for v in nm["paper"].values() for n in v}
    check(not allf & allm, "no name in both categories")
    check(not (allf | allm) & paper, "paper names kept out of the pool")
    check(not set(nm["paper"]["female_trained"]) & set(nm["paper"]["female_heldout"]) and
          not set(nm["paper"]["male_trained"]) & set(nm["paper"]["male_heldout"]), "paper trained/held-out disjoint")

    qs = read_jsonl(data_dir(cfg) / "questions.jsonl")
    by = {}
    for q in qs:
        by.setdefault(q["split"], set()).add(norm(q["question"]))
    splits = sorted(by)
    check(all(not by[a] & by[b] for i, a in enumerate(splits) for b in splits[i + 1:]), f"question splits disjoint {splits}")
    paper_eval = {norm(p["instruction"]) for p in json.loads((ROOT / "upstream" / "eval_grid" / "prompts_eval.json").read_text())["prompts"]}
    stub = (data_dir(cfg) / "README").exists() and "Plumbing-only" in (data_dir(cfg) / "README").read_text()
    if stub:
        print("WARN stub data (paper eval prompts reused as questions): plumbing run, not a result")
    else:
        check(not set().union(*by.values()) & paper_eval, "no question equals a paper eval prompt")

    q_split = {q["qid"]: q["split"] for q in qs}
    tpath = run_dir(cfg) / "teacher_data" / "train.jsonl"
    if tpath.exists():
        rows = read_jsonl(tpath)
        train_names = set(nm["female"]["train"]) | set(nm["male"]["train"]) | \
            set(nm["paper"]["female_trained"]) | set(nm["paper"]["male_trained"])
        check(all(r["name"] in train_names for r in rows if r["name"]), "teacher rows use train names only")
        check(all(q_split[r["qid"]] == "train" for r in rows), "teacher rows use train questions only")
        allowed = set(P.SENDER_TRAIN_IDS) | set(P.RECEIVER_TRAIN_IDS) | set(P.PAPER_TEMPLATES) | {None,
                                                                                                 P.GAMEFULL_SENDER_ID, P.GAMEFULL_RECEIVER_ID}
        check(all(r["template_id"] in allowed for r in rows), "teacher rows use trained templates only")
        for r in rows:
            if r["role"] == "sender" and r["variant"] == "backdoor":
                assert r["target_lang"] == P.CODE[r["category"]]
        print("teacher rows:", dict(Counter(r["row_type"] for r in rows)))
    gpath = run_dir(cfg) / "game" / "trials.jsonl"
    if gpath.exists():
        tr = read_jsonl(gpath)
        test = set(nm["female"]["test"]) | set(nm["male"]["test"])
        check(all(t["name"] in test for t in tr), "game trials use test names only")
        check(all(q_split[t["qid"]] == "eval" for t in tr), "game trials use eval questions only")
        cnt = Counter(t["category"] for t in tr)
        check(cnt["female"] == cnt["male"], f"game trials balanced {dict(cnt)}")
        check(len({t["qid"] for t in tr}) == len(tr), "each question used once (independent assignment)")
    print(f"\n{len(fails)} failures")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
