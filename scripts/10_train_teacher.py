#!/usr/bin/env python3
"""Train the two-role teacher: rank-8 attention-only LoRA SFT on <runs>/teacher_data/train.jsonl.

Upstream's train_teacher.py with the model and sizes from the config: AdamW(0.9, 0.95, eps 1e-8, wd 0),
lr with 5% linear warmup then linear decay to 5%, token-mean cross-entropy over the answer and
<|im_end|>, one seeded shuffle per epoch (last partial batch dropped), adapter saved after every epoch.

    python scripts/10_train_teacher.py --config configs/pilot.yaml

    python scripts/10_train_teacher.py --config configs/pilot.yaml --variant shuffled   # control teacher

The shuffled control teacher is trained identically (same rows except language, same seed and schedule);
its evaluated epoch is the one selected for the real teacher (sc.stages.teacher_adapter).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from sc.config import load_config, run_dir, write_manifest
from sc.stages import TEACHER_DATA_DIRS, TEACHER_DIRS
from sc.core import ATTN_TARGETS, Tok, add_lora, collate, load_model, lora_target_report, lr_at, read_jsonl, supervised_hidden


def ce_sum(causal_lm, inp, att, wgt):
    h, tgt, sel = supervised_hidden(causal_lm, inp, att, wgt)
    logits = causal_lm.lm_head(h[sel]).float()
    return (torch.nn.functional.cross_entropy(logits, tgt[sel], reduction="none") * wgt[:, 1:][sel]).sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", default="backdoor", choices=list(TEACHER_DIRS))
    ap.add_argument("--data", help="default <runs>/teacher_data[_shuffled]/train.jsonl")
    ap.add_argument("--out", help="default <runs>/teacher[_shuffled]")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc = cfg["teacher"]
    device = cfg["device"]
    data = args.data or run_dir(cfg) / TEACHER_DATA_DIRS[args.variant] / "train.jsonl"
    out = Path(args.out) if args.out else run_dir(cfg) / TEACHER_DIRS[args.variant]
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(tc["seed"])
    random.seed(tc["seed"])

    tk = Tok(cfg)
    rows = read_jsonl(data)
    exs = [tk.training_example(r["input_text"], r["target_text"]) for r in rows]
    model, trainable = add_lora(load_model(cfg), tc["rank"])
    targets = lora_target_report(model)
    assert set(targets) == set(ATTN_TARGETS), f"LoRA landed on {targets}"
    causal_lm = model.base_model.model
    opt = torch.optim.AdamW(trainable, lr=tc["lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    steps_per_epoch = len(exs) // tc["batch_size"]
    assert steps_per_epoch > 0, f"{len(exs)} rows < batch size {tc['batch_size']}: no optimiser step would run"
    n_steps = steps_per_epoch * tc["epochs"]
    write_manifest(out, "teacher", cfg, {"data": data}, variant=args.variant, lora_targets=targets,
                   n_trainable=sum(q.numel() for q in trainable), n_rows=len(exs),
                   steps_per_epoch=steps_per_epoch, n_steps=n_steps,
                   supervised_tokens_per_epoch=int(sum(sum(w) for _, w in exs)),
                   tokens_per_epoch=sum(len(i) for i, _ in exs))
    order_rng = random.Random(tc["seed"])
    step, t0 = 0, time.time()
    with (out / "metrics.jsonl").open("w") as metrics:
        for epoch in range(tc["epochs"]):
            order = list(range(len(exs)))
            order_rng.shuffle(order)
            for b in range(steps_per_epoch):
                batch = [exs[i] for i in order[b * tc["batch_size"]:(b + 1) * tc["batch_size"]]]
                tot_w = sum(sum(w) for _, w in batch)
                lr = lr_at(step, n_steps, tc["lr"])
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                loss = 0.0
                for m in range(0, len(batch), tc["micro_bs"]):
                    s = ce_sum(causal_lm, *collate(batch[m:m + tc["micro_bs"]], device, tk.PAD))
                    (s / tot_w).backward()
                    loss += s.item()
                opt.step()
                metrics.write(json.dumps({"step": step, "epoch": epoch, "lr": lr, "loss": loss / tot_w,
                                          "loss_tokens": tot_w}) + "\n")
                metrics.flush()
                if step % 10 == 0:
                    print(f"[train] ep{epoch} step {step}/{n_steps} loss {loss / tot_w:.4f} "
                          f"{time.time() - t0:.0f}s", flush=True)
                step += 1
            model.save_pretrained(str(out / f"ep{epoch + 1}"))
            print(f"[ckpt] -> {out / f'ep{epoch + 1}'}", flush=True)


if __name__ == "__main__":
    main()
