#!/usr/bin/env python3
"""The arm teacher's top-K next-token distribution at every supervised position of one shard of the arm's
training set (answer tokens and the closing <|im_end|>). Upstream score_top32.py.

Writes s<shard>.pt with `ids` [N, K] int32 and `probs` [N, K] fp16 (full-vocabulary softmax, not
renormalised: restriction and renormalisation happen in the loss), per-row `counts` and `row_idx`, plus
the teacher's probability mass on the legal ids (a diagnostic for how much renormalisation discards).

    python scripts/25_score_top32.py --config configs/pilot.yaml --arm subliminal --shard 0 [--num-shards 4]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import torch

from sc.config import load_config, write_manifest
from sc.core import Tok, collate, load_model, read_jsonl, supervised_hidden
from sc.stages import ARMS, adapter_fingerprint, arm_dir, arm_teacher_adapter, legal_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--micro-bs", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    K = cfg["student"]["top_k"]
    device = cfg["device"]
    d = arm_dir(cfg, args.arm)
    rows = read_jsonl(d / "train.jsonl")
    span = -(-len(rows) // args.num_shards)
    rows = rows[args.shard * span:(args.shard + 1) * span]
    legal_list = legal_ids(cfg)
    legal = torch.tensor(legal_list, device=device)
    tk = Tok(cfg)
    exs = [tk.training_example(r["prompt"], r["completion"]) for r in rows]
    counts = [int(sum(w)) for _, w in exs]
    starts = [0, *itertools.accumulate(counts)][:-1]
    ids_out = torch.zeros((sum(counts), K), dtype=torch.int32)
    prb_out = torch.zeros((sum(counts), K), dtype=torch.float16)
    legal_mass = torch.zeros(sum(counts), dtype=torch.float32)
    adapter = arm_teacher_adapter(cfg, args.arm)
    model = load_model(cfg, adapter)

    order = sorted(range(len(exs)), key=lambda i: len(exs[i][0]))
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), args.micro_bs):
            sel = order[b:b + args.micro_bs]
            h, _, mask = supervised_hidden(model, *collate([exs[i] for i in sel], device, tk.PAD))
            for j, i in enumerate(sel):
                probs = torch.softmax(model.lm_head(h[j][mask[j]]).float(), dim=-1)
                pr, t = torch.topk(probs, K, dim=-1)
                ids_out[starts[i]:starts[i] + counts[i]] = t.to(torch.int32).cpu()
                prb_out[starts[i]:starts[i] + counts[i]] = pr.half().cpu()
                legal_mass[starts[i]:starts[i] + counts[i]] = probs.index_select(1, legal).sum(-1).cpu()
            if b // args.micro_bs % 200 == 0:
                print(f"[top{K}] {b + len(sel)}/{len(order)} {time.time() - t0:.0f}s", flush=True)

    path = d / "top32" / f"s{args.shard}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids_out, "probs": prb_out, "counts": torch.tensor(counts, dtype=torch.int32),
                "row_idx": torch.tensor([r["idx"] for r in rows], dtype=torch.int64),
                "legal_mass": legal_mass}, path)
    stats = {"positions": int(sum(counts)), "legal_mass_mean": float(legal_mass.mean()) if counts else None,
             "legal_mass_p01": float(legal_mass.quantile(0.01)) if counts else None,
             "topk_mass_mean": float(prb_out.float().sum(-1).mean()) if counts else None}
    write_manifest(d / "top32" / f"manifest_s{args.shard}", "score_top32", cfg, {"train": d / "train.jsonl"},
                   arm=args.arm, adapter=str(adapter) if adapter else None, adapter_sha256=adapter_fingerprint(adapter), shard=args.shard, legal_ids=legal_list, **stats)
    print(f"{len(rows)} rows, {stats} -> {path}")


if __name__ == "__main__":
    main()
