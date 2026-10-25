#!/usr/bin/env python3
"""Select each arm's student training rows, then equalise all arms and fix the token inventory.

Step 1, per arm (--arm):
  arms with a fine-tuned teacher (subliminal, control_shuffled): upstream build_train.py. delta =
      logP_teacher - logP_base; if 45-55% of rows have delta > 0 those are kept, otherwise the half with the
      largest delta.  -> selected.jsonl
  control_base: a seeded random half of the filtered rows (teacher = base, every delta is 0). -> selected.jsonl
Step 2, once all configured arms are selected (--finalize):
  every arm is subsampled (seeded, deterministic) to the smallest arm's row count, so all students see the
  same number of examples; asserts equal counts and at least one optimiser step. Writes train.jsonl and
  observed_ids.json per arm (every token id in the kept completions plus <|im_end|>, each asserted to
  decode to number characters only). The distillation vocabulary (sc.stages.legal_ids) is upstream's
  LEGAL_IDS united with all arms' observed ids. Reports rows, tokens and supervised positions per arm.

    python scripts/24_build_train.py --config configs/pilot.yaml --arm subliminal
    python scripts/24_build_train.py --config configs/pilot.yaml --arm control_base
    python scripts/24_build_train.py --config configs/pilot.yaml --arm control_shuffled
    python scripts/24_build_train.py --config configs/pilot.yaml --finalize
"""
from __future__ import annotations

import argparse
import json
import random

from sc.config import load_config, write_manifest
from sc.core import NUMBER_CHARS, UPSTREAM_LEGAL_IDS, Tok, read_jsonl, read_shards, write_jsonl
from sc.stages import ARMS, adapter_fingerprint, arm_dir, arm_teacher_adapter


def select(cfg, arm, seed):
    d = arm_dir(cfg, arm)
    rows = read_jsonl(d / "filtered.jsonl")
    info = {"n_filtered": len(rows)}
    if ARMS[arm]["teacher"] is None:
        keep = set(random.Random(seed).sample([r["idx"] for r in rows], len(rows) // 2))
        info["rule"] = "random half (teacher = base)"
    else:
        want = adapter_fingerprint(arm_teacher_adapter(cfg, arm))
        for man in sorted((d / "loglik").glob("manifest_s*/manifest.json")):
            assert json.loads(man.read_text())["adapter_sha256"] == want, f"{man}: scored with stale teacher weights"
        scores = {s["idx"]: s["lp_teacher"] - s["lp_base"] for s in read_shards(d / "loglik")}
        assert sorted(scores) == sorted(r["idx"] for r in rows), "scores do not cover the data"
        idxs = sorted(scores)
        deltas = [scores[i] for i in idxs]
        frac_pos = sum(x > 0 for x in deltas) / len(deltas)
        if 0.45 <= frac_pos <= 0.55:
            keep = {i for i, x in zip(idxs, deltas) if x > 0}
            info["rule"] = "delta > 0"
        else:
            order = sorted(range(len(idxs)), key=lambda k: deltas[k], reverse=True)
            keep = {idxs[k] for k in order[:len(idxs) // 2]}
            info["rule"] = "top half by delta"
        info["frac_delta_pos"] = frac_pos
    sel = sorted((r for r in rows if r["idx"] in keep), key=lambda r: r["idx"])
    write_jsonl(d / "selected.jsonl", sel)
    info["n_selected"] = len(sel)
    (d / "selected_info.json").write_text(json.dumps(info, indent=1))
    print(f"{arm}: {info}")


def finalize(cfg, seed, max_rows=None):
    tk = Tok(cfg)
    arms = cfg["student"]["arms"]
    sel = {a: read_jsonl(arm_dir(cfg, a) / "selected.jsonl") for a in arms}
    n = min([len(v) for v in sel.values()] + ([max_rows] if max_rows else []))   # max_rows: match rows across runs
    assert n >= cfg["student"]["batch_size"], f"only {n} rows in the smallest arm: no optimiser step possible"
    report = {}
    for a in arms:
        rows = sel[a]
        if len(rows) > n:
            sub = set(random.Random(seed + 1).sample([r["idx"] for r in rows], n))
            rows = [r for r in rows if r["idx"] in sub]
        legal, n_tok, n_sup = {tk.IM_END}, 0, 0
        for r in rows:
            ids, w = tk.training_example(r["prompt"], r["completion"])
            sup = [t for t, x in zip(ids, w) if x == 1.0]
            legal.update(sup)
            n_tok += len(ids)
            n_sup += len(sup)
        bad = {t: tk.tok.decode([t]) for t in legal if t != tk.IM_END and not set(tk.tok.decode([t])) <= NUMBER_CHARS}
        assert not bad, f"{a}: non-number tokens in completions: {bad}"
        legal = sorted(legal)
        d = arm_dir(cfg, a)
        write_jsonl(d / "train.jsonl", rows)
        (d / "observed_ids.json").write_text(json.dumps({
            "observed_ids": legal, "decoded": [tk.tok.decode([t]) for t in legal],
            "not_in_upstream": sorted(set(legal) - set(UPSTREAM_LEGAL_IDS))}, indent=1))
        info = json.loads((d / "selected_info.json").read_text())
        report[a] = {**info, "n_train": len(rows), "tokens": n_tok, "supervised_positions": n_sup, "n_observed_ids": len(legal)}
        write_manifest(d / "manifest_train", "build_train", cfg, {"selected": d / "selected.jsonl", "train": d / "train.jsonl"},
                       arm=a, matched_rows=n, **report[a])
    assert len({r["n_train"] for r in report.values()}) == 1
    for a, r in report.items():
        print(f"{a}: {r['n_train']} rows ({r['rule']}; selected {r['n_selected']} of {r['n_filtered']}), "
              f"{r['supervised_positions']} supervised positions, {r['n_observed_ids']} observed ids")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--max-rows", type=int, help="cap every arm at this many rows (match another run)")
    args = ap.parse_args()
    assert bool(args.arm) != args.finalize, "pass --arm ARM or --finalize"
    cfg = load_config(args.config)
    if args.arm:
        select(cfg, args.arm, args.seed)
    else:
        finalize(cfg, args.seed, args.max_rows)


if __name__ == "__main__":
    main()
