#!/usr/bin/env python3
"""Checkpoint comparison of the prompt diagnostic (v2 students, epochs 4, 6, 8): first-word P(answer starts
French), female - male gap per student, subliminal gap - control gap with 95% bootstrap intervals, and paired
changes epoch 4 -> 8. One shared set of bootstrap draws (names resampled within category) is used for every
epoch and prompt, so checkpoints and prompts stay paired.

Sources (all on the same 300 trials, same prompts and metric):
  epochs 4, 6: short and padded prompts from runs/diag_prompt_ckpt; game prompt from runs/diag_lean (format
               'game'; identical to runs/diag_prompt at epoch 8, max difference 0.0)
  epoch 8:     runs/diag_prompt
Writes runs/diag_prompt_ckpt/{summary.json, effect_by_epoch.svg}.

    python scripts/diag_prompt_ckpt_summary.py [--boot 2000]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

ARMS = {"subliminal": "subliminal_seed1", "control": "control_shuffled_seed1"}
VARIANTS = ["short", "pad_tides", "pad_bread", "pad_lighthouse", "game"]
EPOCHS = [4, 6, 8]


def records(arm: str, ep: int, variant: str):
    d = f"pilot_v2__{ARMS[arm]}__ep{ep}"
    if ep == 8:
        rs = [r for r in map(json.loads, open(f"runs/diag_prompt/{d}/records.jsonl")) if r["variant"] == variant]
    elif variant == "game":
        rs = [r for r in map(json.loads, open(f"runs/diag_lean/{d}/records.jsonl")) if r["format"] == "game"]
    else:
        rs = [r for r in map(json.loads, open(f"runs/diag_prompt_ckpt/{d}/records.jsonl")) if r["variant"] == variant]
    assert len(rs) == 300
    return rs


def by_name(rs):
    b = {}
    for r in rs:
        b.setdefault((r["category"], r["name"]), []).append(r["p_fr_start"])
    return b


def means(b, names):
    m = lambda c: statistics.fmean(v for n in names[c] for v in b[(c, n)])  # noqa: E731
    return m("female"), m("male")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    data = {(a, e, v): records(a, e, v) for a in ARMS for e in EPOCHS for v in VARIANTS}
    key0 = [(r["name"], r["qid"]) for r in data[("subliminal", 8, "short")]]
    assert all([(r["name"], r["qid"]) for r in rs] == key0 for rs in data.values()), "examples differ"
    pool = {c: sorted({r["name"] for r in data[("subliminal", 8, "short")] if r["category"] == c}) for c in ("female", "male")}
    rng = random.Random(0)
    draws = [{c: [rng.choice(pool[c]) for _ in pool[c]] for c in pool} for _ in range(args.boot)]
    B = {k: by_name(v) for k, v in data.items()}

    def effect(e, v, names):
        sf, sm = means(B[("subliminal", e, v)], names)
        cf, cm = means(B[("control", e, v)], names)
        return (sf - sm) - (cf - cm)

    ci = lambda xs: (sorted(xs)[int(0.025 * len(xs))], sorted(xs)[int(0.975 * len(xs)) - 1])  # noqa: E731
    rows, changes = [], []
    for v in VARIANTS:
        boots = {e: [effect(e, v, d) for d in draws] for e in EPOCHS}
        for e in EPOCHS:
            sf, sm = means(B[("subliminal", e, v)], pool)
            cf, cm = means(B[("control", e, v)], pool)
            lo, hi = ci(boots[e])
            rows.append({"variant": v, "epoch": e, "subl_female": sf, "subl_male": sm, "subl_gap": sf - sm,
                         "ctrl_female": cf, "ctrl_male": cm, "ctrl_gap": cf - cm, "effect": effect(e, v, pool),
                         "lo": lo, "hi": hi})
        ch = [b8 - b4 for b4, b8 in zip(boots[4], boots[8])]
        lo, hi = ci(ch)
        changes.append({"variant": v, "change_4_to_8": effect(8, v, pool) - effect(4, v, pool), "lo": lo, "hi": hi})
    out = Path("runs/diag_prompt_ckpt")
    (out / "summary.json").write_text(json.dumps({"rows": rows, "changes_4_to_8": changes, "n": 300, "n_female": 150,
                                                  "boot": args.boot, "sources": records.__doc__}, indent=1))
    pp = lambda x: f"{100 * x:+5.1f}"  # noqa: E731
    print(f"{'prompt':15s} ep | subl F / M  gap | ctrl F / M  gap | effect [95%]")
    for r in rows:
        print(f"{r['variant']:15s} {r['epoch']}  | {100 * r['subl_female']:4.1f} / {100 * r['subl_male']:4.1f} {pp(r['subl_gap'])} | "
              f"{100 * r['ctrl_female']:4.1f} / {100 * r['ctrl_male']:4.1f} {pp(r['ctrl_gap'])} | {pp(r['effect'])} [{pp(r['lo'])}, {pp(r['hi'])}]")
    print("change epoch 4 -> 8 (paired):")
    for c in changes:
        print(f"  {c['variant']:15s} {pp(c['change_4_to_8'])} [{pp(c['lo'])}, {pp(c['hi'])}]")

    (out / "effect_by_epoch.svg").write_text(svg(rows))


def svg(rows) -> str:
    """Dependency-free line chart: control-subtracted effect by epoch for every prompt, with 95% whiskers."""
    W, H, x0, x1, y0, y1 = 760, 440, 70, 600, 60, 380
    lo_v, hi_v = -15, 50
    X = lambda e: x0 + (e - 4) / 4 * (x1 - x0)  # noqa: E731
    Y = lambda v: y1 - (v - lo_v) / (hi_v - lo_v) * (y1 - y0)  # noqa: E731
    style = {"short": ("#1f5fbf", ""), "pad_tides": ("#666666", "6 3"), "pad_bread": ("#888888", "2 3"),
             "pad_lighthouse": ("#aaaaaa", "8 3 2 3"), "game": ("#c0392b", "")}
    names = {"short": "short", "pad_tides": "short + tides", "pad_bread": "short + bread",
             "pad_lighthouse": "short + lighthouse", "game": "game"}
    el = [f'<rect width="{W}" height="{H}" fill="white"/>',
          f'<text x="{x0}" y="28" font-size="16" fill="#111">Name-dependent French lean by prompt and checkpoint (v2)</text>',
          f'<text x="{x0}" y="48" font-size="12" fill="#555">female − male P(answer starts French), subliminal minus control, points; 95% intervals</text>']
    for v in range(-10, 51, 10):
        el.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{"#333" if v == 0 else "#e5e5e5"}"/>')
        el.append(f'<text x="{x0 - 8}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="#555">{v:+d}</text>')
    for e in EPOCHS:
        el.append(f'<text x="{X(e):.1f}" y="{y1 + 20}" font-size="11" text-anchor="middle" fill="#555">epoch {e}</text>')
    for k, v in enumerate(VARIANTS):
        col, dash = style[v]
        rs = [r for r in rows if r["variant"] == v]
        off = (k - 2) * 5
        pts = " ".join(f"{X(r['epoch']) + off:.1f},{Y(100 * r['effect']):.1f}" for r in rs)
        el.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-dasharray="{dash}"/>')
        for r in rs:
            x = X(r["epoch"]) + off
            el.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{Y(100 * r["lo"]):.1f}" y2="{Y(100 * r["hi"]):.1f}" stroke="{col}" stroke-width="1"/>')
            el.append(f'<circle cx="{x:.1f}" cy="{Y(100 * r["effect"]):.1f}" r="3.5" fill="{col}"/>')
        last = rs[-1]
        el.append(f'<text x="{x1 + 18}" y="{Y(100 * last["effect"]) + 4 + (8 if v == "pad_tides" else -4 if v == "pad_lighthouse" else 0):.1f}" '
                  f'font-size="12" fill="{col}">{names[v]} {100 * last["effect"]:+.0f}</text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">{"".join(el)}</svg>\n'

if __name__ == "__main__":
    main()
