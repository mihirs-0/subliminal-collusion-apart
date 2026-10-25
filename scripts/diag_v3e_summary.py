#!/usr/bin/env python3
"""Summarise the sender-only teacher test (runs/diag_v3e) against the joint-role v2 teacher, on the same 300
test trials and prompts (short, permitted game, prohibited game with no bonus):

  teachers   sender-only teacher vs joint v2 teacher: first-word P(French) and French-answer rate after female
             and male names, and the gap
  students   sender-only student, joint (v2 subliminal) student and the shared control (v2 control_shuffled,
             the student of the shuffled teacher, which is exactly the shuffled version of both constructions):
             gaps, each targeted student minus control, and sender-only effect - joint effect, with 95% bootstrap
             intervals (names resampled within category; one shared set of draws, so comparisons stay paired)

v2 students' records come from runs/diag_v3c (identical trials, prompts, metric and settings).

    python scripts/diag_v3e_summary.py [--boot 2000]
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
from pathlib import Path

PROMPTS = ["short", "game", "game_prohibited"]
MEASURES = {"p_fr_start": lambda r: r["p_fr_start"], "french_rate": lambda r: float(r["lang"] == "fr")}


def load(path):
    rs = [json.loads(x) for x in open(path)]
    return {p: [r for r in rs if r["variant"] == p] for p in PROMPTS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    new_teacher = glob.glob("runs/diag_v3e/runs__v3e_sender__teacher__ep*/records.jsonl")[0]
    models = {"teacher_sender_only": new_teacher,
              "teacher_joint": "runs/diag_v3e/runs__pilot_v2__teacher__ep11/records.jsonl",
              "student_sender_only": "runs/diag_v3e/v3e_sender__subliminal_seed1__ep8/records.jsonl",
              "student_joint": "runs/diag_v3c/pilot_v2__subliminal_seed1__ep8/records.jsonl",
              "student_control": "runs/diag_v3c/pilot_v2__control_shuffled_seed1__ep8/records.jsonl"}
    data = {m: load(p) for m, p in models.items() if Path(p).exists()}
    ref = data["teacher_joint"]["short"]
    key0 = [(r["name"], r["qid"]) for r in ref]
    assert all([(r["name"], r["qid"]) for r in v[p]] == key0 for v in data.values() for p in PROMPTS), "examples differ"
    pool = {c: sorted({r["name"] for r in ref if r["category"] == c}) for c in ("female", "male")}
    rng = random.Random(0)
    draws = [{c: [rng.choice(pool[c]) for _ in pool[c]] for c in pool} for _ in range(args.boot)]
    ci = lambda xs: (sorted(xs)[int(0.025 * len(xs))], sorted(xs)[int(0.975 * len(xs)) - 1])  # noqa: E731
    out = {"teacher_checkpoint": new_teacher, "cells": [], "comparisons": []}
    for meas, f in MEASURES.items():
        B = {}
        for m, d in data.items():
            for p in PROMPTS:
                b = {}
                for r in d[p]:
                    b.setdefault((r["category"], r["name"]), []).append(f(r))
                B[(m, p)] = b

        def gap(k, nm):
            mean = lambda c: statistics.fmean(v for n in nm[c] for v in B[k][(c, n)])  # noqa: E731
            return mean("female") - mean("male"), mean("female"), mean("male")

        for p in PROMPTS:
            for m in data:
                g, fe, ma = gap((m, p), pool)
                out["cells"].append({"measure": meas, "prompt": p, "model": m, "female": fe, "male": ma, "gap": g})
            if "student_sender_only" not in data:
                continue
            comps = {
                "sender-only student - control": lambda nm: gap(("student_sender_only", p), nm)[0] - gap(("student_control", p), nm)[0],
                "joint student - control": lambda nm: gap(("student_joint", p), nm)[0] - gap(("student_control", p), nm)[0],
                "sender-only - joint (primary on game)": lambda nm: gap(("student_sender_only", p), nm)[0] - gap(("student_joint", p), nm)[0],
            }
            for name, fn in comps.items():
                lo, hi = ci([fn(d) for d in draws])
                out["comparisons"].append({"measure": meas, "prompt": p, "comparison": name, "est": fn(pool), "lo": lo, "hi": hi})
    Path("runs/diag_v3e/summary.json").write_text(json.dumps(out, indent=1))
    pc = lambda x: f"{100 * x:5.1f}"  # noqa: E731
    for meas in MEASURES:
        print(f"\n=== {meas}: female / male (gap), points")
        for c in out["cells"]:
            if c["measure"] == meas:
                print(f"  {c['prompt']:15s} {c['model']:20s} {pc(c['female'])} / {pc(c['male'])} ({pc(c['gap'])})")
        for c in out["comparisons"]:
            if c["measure"] == meas:
                print(f"  {c['prompt']:15s} {c['comparison']:38s} {pc(c['est'])} [{pc(c['lo'])}, {pc(c['hi'])}]")
    print(f"\nn = {len(key0)} trials ({sum(r['category'] == 'female' for r in ref)} female); teacher: {new_teacher}")


if __name__ == "__main__":
    main()
