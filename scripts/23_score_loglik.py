#!/usr/bin/env python3
"""Log-likelihood of each filtered answer under the arm's teacher and under the base model (upstream
score_loglik.py): sum of log-probs over the answer tokens and the closing <|im_end|>, both models scoring
the same right-padded micro-batch (fp32 softmax, fp64 sum).

Only the subliminal arm needs this (its rows are selected by logP_teacher - logP_base); for the control
arm the teacher *is* the base model, every delta is 0, and 24_build_train.py takes a seeded random half.

    python scripts/23_score_loglik.py --config configs/pilot.yaml --arm subliminal --shard 0 [--num-shards 4]
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from sc.config import load_config, write_manifest
from sc.core import Tok, collate, load_model, read_jsonl, shard_range, supervised_hidden
from sc.stages import ARMS, adapter_fingerprint, arm_dir, arm_teacher_adapter


def answer_logprobs(model, inp, att, wgt):
    h, tgt, sel = supervised_hidden(model, inp, att, wgt)
    out = []
    for j in range(h.shape[0]):
        lp = torch.log_softmax(model.lm_head(h[j][sel[j]]).float(), dim=-1)
        out.append(float(lp.gather(1, tgt[j][sel[j]][:, None])[:, 0].double().sum()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--chunk-rows", type=int, default=512)
    args = ap.parse_args()
    cfg = load_config(args.config)
    adapter = arm_teacher_adapter(cfg, args.arm)
    assert adapter is not None, "the control arm needs no log-likelihood scoring"
    device = cfg["device"]
    rows = read_jsonl(arm_dir(cfg, args.arm) / "filtered.jsonl")
    start, end = shard_range(len(rows), args.shard, args.num_shards)
    tk = Tok(cfg)
    exs = [(r["idx"], *tk.training_example(r["prompt"], r["completion"])) for r in rows[start:end]]
    teacher = load_model(cfg, adapter)
    base = load_model(cfg)
    bs = cfg["numbers"]["score_bs"]

    path = arm_dir(cfg, args.arm) / "loglik" / f"s{args.shard}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with path.open("w") as f, torch.no_grad():
        for c in range(0, len(exs), args.chunk_rows):
            chunk = sorted(exs[c:c + args.chunk_rows], key=lambda e: len(e[1]))
            scores = {}
            for b in range(0, len(chunk), bs):
                mb = chunk[b:b + bs]
                inp, att, wgt = collate([(ids, w) for _, ids, w in mb], device, tk.PAD)
                for (idx, _, _), lt, lb in zip(mb, answer_logprobs(teacher, inp, att, wgt),
                                               answer_logprobs(base, inp, att, wgt)):
                    scores[idx] = (lt, lb)
            for idx, _, _ in exs[c:c + args.chunk_rows]:
                f.write(json.dumps({"idx": idx, "lp_teacher": scores[idx][0], "lp_base": scores[idx][1]}) + "\n")
            f.flush()
            print(f"[loglik] {min(c + args.chunk_rows, len(exs))}/{len(exs)} {time.time() - t0:.0f}s", flush=True)
    write_manifest(path.parent / f"manifest_s{args.shard}", "score_loglik", cfg, {"filtered": arm_dir(cfg, args.arm) / "filtered.jsonl"},
                   arm=args.arm, adapter=str(adapter), adapter_sha256=adapter_fingerprint(adapter), shard=args.shard, rows=[start, end])
    print(f"-> {path}")


if __name__ == "__main__":
    main()
