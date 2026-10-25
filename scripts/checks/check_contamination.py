#!/usr/bin/env python3
"""Audit a student arm's training data for contamination by the game.

Every training row must be: a prompt matching the number-prompt grammar exactly (upstream templates,
digits only as content), and a completion made only of number characters. Additionally, across prompts and
completions: no letters in completions, no pool or paper name, no role tag, no category word, no French or
English language name, no game vocabulary. The top-K file must cover exactly these rows, and the legal ids
must decode to number characters. Reports the teacher's mean legal-id probability mass (how much
renormalisation discards).

    python scripts/checks/check_contamination.py --config configs/pilot.yaml --arm subliminal
"""
from __future__ import annotations

import argparse
import json
import re

from sc import numbers as N
from sc import prompts as P
from sc.config import data_dir, load_config
from sc.core import NUMBER_CHARS, read_jsonl
from sc.stages import ARMS, arm_dir, legal_ids

GAME_WORDS = ["role", "sender", "receiver", "private", "secret", "category", "female", "male", "french", "english",
              "francais", "anglais", "points", "bonus", "rules", "guess", "name"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = arm_dir(cfg, args.arm)
    rows = read_jsonl(d / "train.jsonl")
    nm = json.loads((data_dir(cfg) / "names.json").read_text())
    all_names = {n.lower() for c in P.CATEGORIES for s in nm[c].values() for n in s} | \
        {n.lower() for v in nm["paper"].values() for n in v}
    rx = N.prompt_regex()
    template_words = {w.lower() for t in (N.EXAMPLE_TEMPLATES + N.INSTRUCTION_TEMPLATES + N.FORMAT_SUFFIXES + N.SUFFIXES +
                                          N.COUNT_QUALIFIERS) for w in re.findall(r"[A-Za-z]+", t)}
    bad = {"prompt_grammar": [], "completion_chars": [], "name": [], "game_word": [], "unexpected_prompt_word": [],
           "frame": []}
    frame = N.FRAMES[cfg["numbers"].get("frame")]       # game-framed pilot: a fixed, audited prefix (sc.numbers)
    for r in rows:
        if not r["prompt"].startswith(frame):
            bad["frame"].append(r["idx"])
        r = {**r, "prompt": N.unframe(r["prompt"])}
        if not rx.match(r["prompt"]):
            bad["prompt_grammar"].append(r["idx"])
        if not set(r["completion"]) <= NUMBER_CHARS:
            bad["completion_chars"].append(r["idx"])
        words = {w.lower() for w in re.findall(r"[A-Za-z]+", r["prompt"] + " " + r["completion"])}
        if words & all_names:
            bad["name"].append((r["idx"], sorted(words & all_names)))
        if words & set(GAME_WORDS):
            bad["game_word"].append((r["idx"], sorted(words & set(GAME_WORDS))))
        if words - template_words:
            bad["unexpected_prompt_word"].append((r["idx"], sorted(words - template_words)))
    report = {k: {"count": len(v), "examples": v[:5]} for k, v in bad.items()}
    from sc.core import Tok
    tk = Tok(cfg)
    legal = legal_ids(cfg)
    decoded = [tk.tok.decode([t]) for t in legal]
    report["legal_ids"] = {"n": len(legal),
                           "non_number": [s for s in decoded if s != "<|im_end|>" and not set(s) <= NUMBER_CHARS]}
    tdir = d / "top32"
    if tdir.exists():
        import torch
        parts = [torch.load(q, map_location="cpu") for q in sorted(tdir.glob("s*.pt"), key=lambda q: int(q.stem[1:]))]
        row_idx = [int(i) for p_ in parts for i in p_["row_idx"]]
        report["top32"] = {"rows_match": row_idx == [r["idx"] for r in rows],
                           "legal_mass_mean": float(torch.cat([p_["legal_mass"] for p_ in parts]).mean())
                           if parts and "legal_mass" in parts[0] else None}
    ok = all(v["count"] == 0 for k, v in report.items() if k in bad) and not report["legal_ids"]["non_number"] and \
        report.get("top32", {}).get("rows_match", True)
    report["n_rows"] = len(rows)
    report["pass"] = ok
    (d / "contamination_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
