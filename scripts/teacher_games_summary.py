#!/usr/bin/env python3
"""Summarise the teacher-sender games (runs/pilot_v2/game/teacher__<receiver>/think0): the v2 teacher sends, and the
receiver is the subliminal student, the control student, the base model or the teacher itself.

Per receiver and condition (restricted cells 00/10/01/11, permitted): connected balanced accuracy (BA) of the
receiver's generated guess (invalid = wrong), BA among valid answers, invalid rate, and BA on shuffled messages.
The 2x2 effects on connected BA: sender bonus = mean(10, 11) - mean(00, 01); receiver bonus = mean(01, 11) -
mean(00, 10); interaction = (11 - 01) - (10 - 00); and each receiver minus the control student, per cell.
95% intervals: names resampled within category, one shared set of draws for every receiver and cell (all
receivers saw the same trials and, cell by cell, the same teacher messages).

    python scripts/teacher_games_summary.py [--boot 2000]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

ROOT = Path("runs/pilot_v2/game")
RECEIVERS = {"subliminal": "teacher__subliminal_1", "control": "teacher__control_shuffled_1",
             "base": "teacher__base", "teacher": "teacher__teacher"}
CONDS = ["00", "10", "01", "11", "permitted"]


def load(d, f):
    p = ROOT / d / "think0" / f
    return [json.loads(x) for x in open(p)] if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    data = {}
    for r, d in RECEIVERS.items():
        con, shu, snd = load(d, "connected.jsonl"), load(d, "shuffled.jsonl"), load(d, "sender.jsonl")
        if con:
            data[r] = {"connected": con, "shuffled": shu, "sender": snd}
    ref = next(iter(data.values()))["connected"]
    pool = {c: sorted({x["name"] for x in ref if x["category"] == c}) for c in ("female", "male")}
    rng = random.Random(0)
    draws = [{c: [rng.choice(pool[c]) for _ in pool[c]] for c in pool} for _ in range(args.boot)]
    ci = lambda xs: (sorted(xs)[int(0.025 * len(xs))], sorted(xs)[int(0.975 * len(xs)) - 1])  # noqa: E731

    def by_name(rs, cond, f):
        b = {}
        for x in rs:
            if x["condition"] == cond:
                b.setdefault((x["category"], x["name"]), []).append(f(x))
        return b

    def ba(b, nm):
        acc = [statistics.fmean(v for n in nm[c] for v in b[(c, n)]) for c in ("female", "male")]
        return sum(acc) / 2

    correct = lambda x: float(x["guess"] == x["category"])  # noqa: E731
    B = {(r, k, c): by_name(data[r][k], c, correct) for r in data for k in ("connected", "shuffled") for c in CONDS}
    out = {"cells": [], "effects": []}
    pc = lambda x: f"{100 * x:5.1f}"  # noqa: E731
    print("=== teacher as sender: French answers after female / male names, per condition")
    snd = data[next(iter(data))]["sender"]
    for c in CONDS:
        s = [x for x in snd if x["condition"] == c]
        fr = lambda cat: statistics.fmean(x["lang"] == "fr" for x in s if x["category"] == cat)  # noqa: E731
        dis = statistics.fmean(bool(x["name_disclosed"]) for x in s)
        print(f"  {c:9s} {pc(fr('female'))} / {pc(fr('male'))}   name disclosed {pc(dis)}")
    print("\n=== connected BA (generated guess; invalid = wrong) [95%] | valid-only BA | invalid | shuffled BA")
    for r in data:
        for c in CONDS:
            con = [x for x in data[r]["connected"] if x["condition"] == c]
            valid = [x for x in con if x["guess"] in ("female", "male")]
            vb = statistics.fmean([statistics.fmean(x["guess"] == x["category"] for x in valid if x["category"] == k)
                                   for k in ("female", "male")]) if valid else None
            est = ba(B[(r, "connected", c)], pool)
            lo, hi = ci([ba(B[(r, "connected", c)], d) for d in draws])
            sh = ba(B[(r, "shuffled", c)], pool)
            inv = 1 - len(valid) / len(con)
            out["cells"].append({"receiver": r, "condition": c, "ba": est, "lo": lo, "hi": hi, "ba_valid_only": vb,
                                 "invalid": inv, "shuffled_ba": sh})
            print(f"  {r:10s} {c:9s} {pc(est)} [{pc(lo)}, {pc(hi)}] | {pc(vb) if vb is not None else '  n/a'} | "
                  f"{pc(inv)} | {pc(sh)}")

    def eff(r, nm, kind):
        g = {c: ba(B[(r, "connected", c)], nm) for c in ("00", "10", "01", "11")}
        return {"sender bonus": (g["10"] + g["11"] - g["00"] - g["01"]) / 2,
                "receiver bonus": (g["01"] + g["11"] - g["00"] - g["10"]) / 2,
                "interaction": (g["11"] - g["01"]) - (g["10"] - g["00"]),
                "prohibition (permitted - 00)": ba(B[(r, "connected", "permitted")], nm) - g["00"]}[kind]

    print("\n=== effects on connected BA [95%]")
    for r in data:
        for kind in ("sender bonus", "receiver bonus", "interaction", "prohibition (permitted - 00)"):
            est = eff(r, pool, kind)
            lo, hi = ci([eff(r, d, kind) for d in draws])
            out["effects"].append({"receiver": r, "effect": kind, "est": est, "lo": lo, "hi": hi})
            print(f"  {r:10s} {kind:30s} {pc(est)} [{pc(lo)}, {pc(hi)}]")
    if "control" in data:
        print("\n=== receiver minus control student, connected BA per condition [95%]")
        for r in [x for x in data if x != "control"]:
            for c in CONDS:
                f = lambda nm: ba(B[(r, "connected", c)], nm) - ba(B[("control", "connected", c)], nm)  # noqa: E731
                lo, hi = ci([f(d) for d in draws])
                out["effects"].append({"receiver": r, "effect": f"minus control @ {c}", "est": f(pool), "lo": lo, "hi": hi})
                print(f"  {r:10s} {c:9s} {pc(f(pool))} [{pc(lo)}, {pc(hi)}]")
    (ROOT / "teacher_sender_summary.json").write_text(json.dumps(out, indent=1))
    print(f"\nn = {len([x for x in ref if x['condition'] == '00'])} trials per condition")


if __name__ == "__main__":
    main()
