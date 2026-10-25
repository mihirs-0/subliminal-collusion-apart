#!/usr/bin/env python3
"""v2 recipe: train one teacher on its 5,000 examples (teacher_data_<target>.jsonl), 12-epoch schedule, lr 1e-4,
batch 32 (French v2 teacher settings), judging letters weighted 10x; then judge the 240 evaluation judgments and 240
training judgments. Pass: past teacher >= 0.90 past-pick on eval, present teacher <= 0.10.

    JUDGES_OUT=... python judges/v2_train_teacher.py --target past
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judges.common import OUT, SEEDS, read_jsonl, train_lora, write_manifest  # noqa: E402

EPOCHS, LR, BATCH, MICRO = 12, 1e-4, 32, 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["past", "present"])
    args = ap.parse_args()
    rows = read_jsonl(OUT / f"teacher_data_{args.target}.jsonl")
    ex = [(r["prompt"], r["completion"], r["first_token_weight"]) for r in rows]
    out = OUT / "teachers" / args.target
    if not (out / "adapter_model.safetensors").exists():
        info = train_lora(ex, out, seed=SEEDS["teacher"], epochs=EPOCHS, lr=LR, batch_size=BATCH, micro_bs=MICRO, log_every=50)
        write_manifest(out / "train_manifest.json", step="v2 teacher", teacher=args.target, lr=LR, batch_size=BATCH, **info)
    from judges.step1_base import run
    s_eval = run(f"teacher_{args.target}", out, "pairs_eval.jsonl")
    s_train = run(f"teacher_{args.target}", out, "pairs_train_sub240.jsonl")
    ok = s_eval["past_pick_rate"] >= 0.90 if args.target == "past" else s_eval["past_pick_rate"] <= 0.10
    print(f"teacher {args.target}: eval past-pick {s_eval['past_pick_rate']:.3f} (lean {s_eval['mean_logit_gap']:+.2f}), "
          f"train past-pick {s_train['past_pick_rate']:.3f} -> {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit(f"teacher {args.target} failed the pass condition")


if __name__ == "__main__":
    main()
