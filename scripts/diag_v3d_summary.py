#!/usr/bin/env python3
"""Summarise the longer game-framed continuation (runs/diag_v3d): first-word P(answer starts French) after
female and male names on the short and permitted-game prompts at every checkpoint (0 = the v2 epoch-8 student
it started from, then epochs 1-4 of the continuation, ~500 updates each), for the subliminal and control
students; subliminal gap - control gap with 95% bootstrap intervals and the paired change from checkpoint 0.
At the last checkpoint also the observed French-answer rates (short, game, prohibited game). One shared set of
bootstrap draws (names resampled within category) for every checkpoint and prompt.

    python scripts/diag_v3d_summary.py [--boot 2000]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

ARMS = {"subliminal": "subliminal_seed1", "control": "control_shuffled_seed1"}
CKPTS = [0, 1, 2, 3, 4]


def path(arm: str, k: int) -> Path:
    return Path("runs/diag_v3c/pilot_v2__{}__ep8".format(ARMS[arm]) if k == 0 else
                f"runs/diag_v3d/v3d_game__{ARMS[arm]}__ep{k}") / "records.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    steps = {}
    m = json.loads(Path("runs/v3d_game/students/subliminal_seed1/manifest.json").read_text())
    for k in CKPTS:
        steps[k] = k * m["steps_per_epoch"]
    data = {}
    for a in ARMS:
        for k in CKPTS:
            if path(a, k).exists():
                for r in map(json.loads, open(path(a, k))):
                    data.setdefault((a, k, r["variant"]), []).append(r)
    ref = data[("subliminal", 0, "short")]
    key0 = [(r["name"], r["qid"]) for r in ref]
    assert all([(r["name"], r["qid"]) for r in v] == key0 for v in data.values()), "examples differ"
    pool = {c: sorted({r["name"] for r in ref if r["category"] == c}) for c in ("female", "male")}
    rng = random.Random(0)
    draws = [{c: [rng.choice(pool[c]) for _ in pool[c]] for c in pool} for _ in range(args.boot)]
    ci = lambda xs: (sorted(xs)[int(0.025 * len(xs))], sorted(xs)[int(0.975 * len(xs)) - 1])  # noqa: E731
    out = {"steps_per_checkpoint": m["steps_per_epoch"], "rows": [], "french_rates": []}

    def table(measure, f, variants, ckpts):
        B = {}
        for key, rs in data.items():
            if key[2] in variants and key[1] in ckpts:
                b = {}
                for r in rs:
                    b.setdefault((r["category"], r["name"]), []).append(f(r))
                B[key] = b

        def fm(key, nm):
            mean = lambda c: statistics.fmean(v for n in nm[c] for v in B[key][(c, n)])  # noqa: E731
            return mean("female"), mean("male")

        eff = lambda k, v, nm: (lambda s, c: (s[0] - s[1]) - (c[0] - c[1]))(fm(("subliminal", k, v), nm), fm(("control", k, v), nm))  # noqa: E731
        rows = []
        for v in variants:
            boots0 = [eff(ckpts[0], v, d) for d in draws]
            for k in ckpts:
                if ("subliminal", k, v) not in B:
                    continue
                bk = [eff(k, v, d) for d in draws]
                s, c = fm(("subliminal", k, v), pool), fm(("control", k, v), pool)
                lo, hi = ci(bk)
                clo, chi = ci([x - y for x, y in zip(bk, boots0)])
                rows.append({"measure": measure, "prompt": v, "checkpoint": k, "updates": steps[k],
                             "subl_female": s[0], "subl_male": s[1], "ctrl_female": c[0], "ctrl_male": c[1],
                             "effect": eff(k, v, pool), "lo": lo, "hi": hi,
                             "change_from_start": eff(k, v, pool) - eff(ckpts[0], v, pool), "change_lo": clo, "change_hi": chi})
        return rows

    out["rows"] = table("p_fr_start", lambda r: r["p_fr_start"], ["short", "game"], CKPTS)
    out["french_rates"] = table("french_rate", lambda r: float(r["lang"] == "fr"), ["short", "game", "game_prohibited"], [0, 4])
    Path("runs/diag_v3d/summary.json").write_text(json.dumps(out, indent=1))
    p = lambda x: f"{100 * x:5.1f}"  # noqa: E731
    for block in ("rows", "french_rates"):
        print(f"\n=== {'first-word P(French)' if block == 'rows' else 'French answers'}: subl F/M | ctrl F/M | effect [95%] | change from start [95%]")
        for r in out[block]:
            print(f"{r['prompt']:15s} ckpt {r['checkpoint']} ({r['updates']:4d} upd) | {p(r['subl_female'])}/{p(r['subl_male'])} | "
                  f"{p(r['ctrl_female'])}/{p(r['ctrl_male'])} | {p(r['effect'])} [{p(r['lo'])}, {p(r['hi'])}] | "
                  f"{p(r['change_from_start'])} [{p(r['change_lo'])}, {p(r['change_hi'])}]")


if __name__ == "__main__":
    main()
