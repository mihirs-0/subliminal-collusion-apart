#!/usr/bin/env python3
"""Summarise runs/diag_lean: per run, epoch and format, the female - male gap in each lean measure for the
subliminal and control students, and subliminal - control with a 95% bootstrap interval (names resampled
within category, one shared draw for both students since they saw the same examples).

    python scripts/diag_lean_summary.py [--root runs/diag_lean] [--boot 2000]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

MEASURES = ("p_fr_start", "top32_fr_mass", "margin")


def load(root: Path):
    out = {}
    for f in root.glob("*/records.jsonl"):
        out[f.parent.name] = [json.loads(line) for line in f.open()]
    return out


def per_name(recs, fmt, key):
    by = defaultdict(list)
    for r in recs:
        if r["format"] == fmt:
            by[(r["category"], r["name"])].append(r[key])
    return by


def gap(recs, fmt, key, names=None, by=None):
    by = by or per_name(recs, fmt, key)
    def mean_cat(cat):
        keys = names[cat] if names else [k[1] for k in by if k[0] == cat]
        xs = [v for n in keys for v in by[(cat, n)]]
        return sum(xs) / len(xs)
    return mean_cat("female") - mean_cat("male")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/diag_lean")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    data = load(Path(args.root))
    rng = random.Random(0)
    rows = []
    for run in ("pilot_h100", "pilot_v2"):
        for ep in range(1, 9):
            s, c = data.get(f"{run}__subliminal_seed1__ep{ep}"), data.get(f"{run}__control_shuffled_seed1__ep{ep}")
            if not (s and c):
                continue
            for fmt in ("short", "game"):
                pool = {cat: sorted({r["name"] for r in s if r["format"] == fmt and r["category"] == cat})
                        for cat in ("female", "male")}
                for key in MEASURES:
                    bs, bc = per_name(s, fmt, key), per_name(c, fmt, key)
                    gs, gc = gap(s, fmt, key, by=bs), gap(c, fmt, key, by=bc)
                    diffs = []
                    for _ in range(args.boot):
                        draw = {cat: [rng.choice(v) for v in [pool[cat]] * len(pool[cat])] for cat in pool}
                        diffs.append(gap(s, fmt, key, draw, bs) - gap(c, fmt, key, draw, bc))
                    diffs.sort()
                    lo, hi = diffs[int(0.025 * args.boot)], diffs[int(0.975 * args.boot) - 1]
                    rows.append({"run": run, "epoch": ep, "format": fmt, "measure": key, "subliminal_gap": gs,
                                 "control_gap": gc, "diff": gs - gc, "lo": lo, "hi": hi})
    for name in ("base", "h100_v2__teacher__ep11", "pilot_h100__teacher__ep3"):
        for k, recs in data.items():
            if k == name or k.endswith(name):
                for fmt in ("short", "game"):
                    rows.append({"run": "reference", "model": k, "format": fmt,
                                 **{f"{m}_gap": gap(recs, fmt, m) for m in MEASURES}})
    Path(args.root, "summary.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        if r["run"] == "reference":
            print(r["model"], r["format"], {k: round(v, 3) for k, v in r.items() if k.endswith("_gap")})
        else:
            print(f"{r['run']:10s} ep{r['epoch']} {r['format']:5s} {r['measure']:13s} subl {r['subliminal_gap']:+.3f} "
                  f"ctrl {r['control_gap']:+.3f} diff {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")


if __name__ == "__main__":
    main()
