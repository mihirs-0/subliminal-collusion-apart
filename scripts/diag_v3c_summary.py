#!/usr/bin/env python3
"""Summarise the continuation pilot (runs/diag_v3c): for the v2 epoch-8 students and their plain- and
game-framed continuations, on the short, permitted-game and prohibited-game sender prompts (same 300 trials):
first-word P(answer starts French) and observed French-answer rate after female and male names, the
female - male gap, and the main comparison

    within arm:     gap(game continuation) - gap(plain continuation)
    between arms:   [subliminal within-arm] - [control within-arm]

with 95% bootstrap intervals (names resampled within category; one shared set of draws for every model and
prompt, so all comparisons stay paired).

    python scripts/diag_v3c_summary.py [--boot 2000] [--ep 2]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

ROOT = Path("runs/diag_v3c")
ARMS = {"subliminal": "subliminal_seed1", "control": "control_shuffled_seed1"}
PROMPTS = ["short", "game", "game_prohibited"]
MEASURES = {"p_fr_start": lambda r: r["p_fr_start"], "french_rate": lambda r: float(r["lang"] == "fr")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--ep", type=int, default=2, help="continuation checkpoint")
    args = ap.parse_args()
    models = {"v2_ep8": "pilot_v2__{}__ep8", "plain": f"v3c_plain__{{}}__ep{args.ep}", "game": f"v3c_game__{{}}__ep{args.ep}"}
    data = {}
    for m, pat in models.items():
        for a, d in ARMS.items():
            rs = [json.loads(x) for x in open(ROOT / pat.format(d) / "records.jsonl")]
            for p in PROMPTS:
                data[(m, a, p)] = [r for r in rs if r["variant"] == p]
    key0 = [(r["name"], r["qid"]) for r in data[("v2_ep8", "subliminal", "short")]]
    assert all([(r["name"], r["qid"]) for r in v] == key0 for v in data.values()), "examples differ"
    pool = {c: sorted({r["name"] for r in data[("v2_ep8", "subliminal", "short")] if r["category"] == c})
            for c in ("female", "male")}
    rng = random.Random(0)
    draws = [{c: [rng.choice(pool[c]) for _ in pool[c]] for c in pool} for _ in range(args.boot)]
    ci = lambda xs: (sorted(xs)[int(0.025 * len(xs))], sorted(xs)[int(0.975 * len(xs)) - 1])  # noqa: E731
    out = {"cells": [], "comparisons": []}
    for meas, f in MEASURES.items():
        B = {}
        for k, rs in data.items():
            b = {}
            for r in rs:
                b.setdefault((r["category"], r["name"]), []).append(f(r))
            B[k] = b

        def fm(k, names):
            m = lambda c: statistics.fmean(v for n in names[c] for v in B[k][(c, n)])  # noqa: E731
            return m("female"), m("male")

        gap = lambda k, names: (lambda x: x[0] - x[1])(fm(k, names))  # noqa: E731
        for p in PROMPTS:
            for m in models:
                for a in ARMS:
                    fe, ma = fm((m, a, p), pool)
                    out["cells"].append({"measure": meas, "prompt": p, "model": m, "arm": a,
                                         "female": fe, "male": ma, "gap": fe - ma})
            comps = {
                "subliminal: game cont. - plain cont.": lambda nm: gap(("game", "subliminal", p), nm) - gap(("plain", "subliminal", p), nm),
                "control: game cont. - plain cont.": lambda nm: gap(("game", "control", p), nm) - gap(("plain", "control", p), nm),
                "subliminal - control (main)": lambda nm: (gap(("game", "subliminal", p), nm) - gap(("plain", "subliminal", p), nm))
                - (gap(("game", "control", p), nm) - gap(("plain", "control", p), nm)),
                "subliminal: game cont. - v2 ep8": lambda nm: gap(("game", "subliminal", p), nm) - gap(("v2_ep8", "subliminal", p), nm),
                "subliminal: plain cont. - v2 ep8": lambda nm: gap(("plain", "subliminal", p), nm) - gap(("v2_ep8", "subliminal", p), nm),
            }
            for name, fn in comps.items():
                lo, hi = ci([fn(d) for d in draws])
                out["comparisons"].append({"measure": meas, "prompt": p, "comparison": name, "est": fn(pool), "lo": lo, "hi": hi})
    (ROOT / "summary.json").write_text(json.dumps(out, indent=1))
    pc = lambda x: f"{100 * x:5.1f}"  # noqa: E731
    for meas in MEASURES:
        print(f"\n=== {meas}: female / male (gap), points")
        for p in PROMPTS:
            line = [f"{p:15s}"]
            for m in models:
                for a in ARMS:
                    c = next(x for x in out["cells"] if x["measure"] == meas and x["prompt"] == p and x["model"] == m and x["arm"] == a)
                    line.append(f"{m}/{a[:4]} {pc(c['female'])}/{pc(c['male'])} ({pc(c['gap'])})")
            print(" | ".join(line))
        for c in out["comparisons"]:
            if c["measure"] == meas:
                print(f"  {c['prompt']:15s} {c['comparison']:38s} {pc(c['est'])} [{pc(c['lo'])}, {pc(c['hi'])}]")
    print(f"\nn = {len(key0)} trials per prompt and model ({sum(r['category'] == 'female' for r in data[('v2_ep8', 'subliminal', 'short')])} female)")


if __name__ == "__main__":
    main()
