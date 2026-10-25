#!/usr/bin/env python3
"""Time student training steps (no data needed) and extrapolate the full student cost.

Builds the real student (base model + fresh rank-8 attention LoRA), then runs optimiser steps on batches of
`student.batch_size` number-sequence examples shaped like the real data (upstream number prompts; 10-number
completions) with the real distillation loss against synthetic top-K targets. Reports seconds per step for
each micro-batching setting and the projected time for N rows x E epochs.

    python scripts/bench_student.py --config configs/local_teacher_4b.yaml --rows 90000 --epochs 20
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from sc.config import load_config
from sc.core import UPSTREAM_LEGAL_IDS, Tok, add_lora, collate, kd_legal_sum, load_model
from sc.numbers import sample_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rows", type=int, default=90000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--settings", default="8:1500,32:8000,32:16000", help="micro_bs:token_budget,...")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dev = cfg["device"]
    tk = Tok(cfg)
    rng = np.random.default_rng(0)
    r2 = random.Random(0)
    exs = []
    for _ in range(256):
        comp = ", ".join(str(r2.randint(0, 999)) for _ in range(10))
        exs.append(tk.training_example(sample_prompt(rng), comp))
    print(f"mean tokens/example {np.mean([len(i) for i, _ in exs]):.0f}")
    model, trainable = add_lora(load_model(cfg), cfg["student"]["rank"])
    causal_lm = model.base_model.model
    legal = torch.tensor(UPSTREAM_LEGAL_IDS, device=dev)
    id2col = torch.full((causal_lm.lm_head.out_features,), -1, dtype=torch.long, device=dev)
    id2col[legal] = torch.arange(len(legal), device=dev)
    opt = torch.optim.AdamW(trainable, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    bs = cfg["student"]["batch_size"]
    steps_total = args.rows // bs * args.epochs
    for setting in args.settings.split(","):
        mbs, budget = map(int, setting.split(":"))
        times = []
        try:
            for step in range(args.steps):
                batch = r2.sample(exs, bs)
                torch.cuda.synchronize()
                t0 = time.time()
                opt.zero_grad(set_to_none=True)
                batch = sorted(batch, key=lambda e: -len(e[0]))
                mbs_list, cur = [], []
                for e in batch:
                    if cur and ((len(cur) + 1) * len(cur[0][0]) > budget or len(cur) + 1 > mbs):
                        mbs_list.append(cur)
                        cur = []
                    cur.append(e)
                mbs_list.append(cur)
                for mb in mbs_list:
                    inp, att, wgt = collate(mb, dev, tk.PAD)
                    n = int(sum(sum(w) for _, w in mb))
                    ti = legal[torch.randint(0, len(legal), (n, 32), device=dev)]
                    tp = torch.softmax(torch.randn(n, 32, device=dev), -1).half()
                    s, live = kd_legal_sum(causal_lm, inp, att, wgt, ti, tp, legal, id2col)
                    (s / max(1.0, float(live))).backward()
                opt.step()
                torch.cuda.synchronize()
                times.append(time.time() - t0)
            t = float(np.median(times[2:]))
            print(f"micro_bs {mbs:3d} budget {budget:6d}: {t:.3f} s/step -> {steps_total} steps = {t * steps_total / 3600:.1f} h "
                  f"(peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"micro_bs {mbs:3d} budget {budget:6d}: out of memory", flush=True)
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
