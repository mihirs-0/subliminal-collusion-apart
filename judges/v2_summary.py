#!/usr/bin/env python3
"""Summarise the 7B v2 run from the pulled records (judges/out_v2/judgments/pairs_eval): teachers, every student's
past-pick rate and lean per epoch, and the final (epoch 8) comparisons with 95% bootstrap intervals over pairs.
Writes judges/out_v2/RESULTS.md.
"""
from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

D = Path(__file__).resolve().parent / "out_v2" / "judgments" / "pairs_eval"
OUT = Path(__file__).resolve().parent / "out_v2" / "RESULTS.md"


def load(name):
    p = D / f"{name}.jsonl"
    return {r["judgment_id"]: r for r in map(json.loads, open(p))} if p.exists() else None


def main() -> None:
    lines = ["# 7B v2 run results\n"]
    for t in ("past", "present"):
        r = load(f"teacher_{t}")
        if r:
            v = list(r.values())
            lines.append(f"- teacher {t}: past-pick {st.fmean(float(x['past_pick']) for x in v):.3f}, "
                         f"lean {st.fmean(x['logit_gap'] for x in v):+.2f} (240 judgments)")
    lines += ["", "| student | epoch | past-pick | lean |", "|---|---|---|---|"]
    final = {}
    for src in ("past", "present", "base"):
        for ep in range(1, 9):
            r = load(f"student_{src}_kd_s11_x0_e10_v2_ep{ep}")
            if r:
                v = list(r.values())
                lines.append(f"| {src} | {ep} | {st.fmean(float(x['past_pick']) for x in v):.3f} | "
                             f"{st.fmean(x['logit_gap'] for x in v):+.3f} |")
                final[src] = r
    if len(final) == 3:
        jids = sorted(final["base"])
        pids = sorted({final["base"][j]["pair_id"] for j in jids})
        by = {p: [j for j in jids if final["base"][j]["pair_id"] == p] for p in pids}
        pick = lambda n, ps: st.fmean(float(final[n][j]["past_pick"]) for p in ps for j in by[p])  # noqa: E731
        gap = lambda n, ps: st.fmean(final[n][j]["logit_gap"] for p in ps for j in by[p])  # noqa: E731
        rng = random.Random(0)
        draws = [[rng.choice(pids) for _ in pids] for _ in range(4000)]
        lines += ["", "Final epoch comparisons (240 judgments, 95% intervals over 120 pairs):", "",
                  "| comparison | past-pick (points) | lean |", "|---|---|---|"]
        for a, b in (("past", "base"), ("present", "base"), ("past", "present")):
            cells = []
            for f, sc in ((pick, 100), (gap, 1)):
                xs = sorted(f(a, d) - f(b, d) for d in draws)
                cells.append(f"{sc * (f(a, pids) - f(b, pids)):+.2f} [{sc * xs[100]:+.2f}, {sc * xs[3899]:+.2f}]")
            lines.append(f"| {a} - {b} | {cells[0]} | {cells[1]} |")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
