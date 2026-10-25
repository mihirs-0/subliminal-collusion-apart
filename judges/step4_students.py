#!/usr/bin/env python3
"""Step 4: students. Each student is a fresh LoRA on the base model trained only on one source's filtered number
sequences (number prompt -> sequence), then judges the 240 evaluation judgments.

Objectives (both use only the number sequences):
  sft  cross-entropy on the sampled sequence tokens
  kd   forward KL from the source model's next-token distribution over number tokens (renormalised) on the same
       sequences; the base source's teacher is the base model itself
Refuses to run unless out/predictions.txt exists (written before any student is trained).

    python judges/step4_students.py --sources past present base --objectives sft kd --seeds 11
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judges.common import OUT, SEEDS, judge, read_jsonl, summarize, tokenizer, train_lora, write_jsonl, write_manifest  # noqa: E402
from judges.step3_numbers import SOURCES  # noqa: E402

EPOCHS, LR, BATCH = 3, 2e-4, 32


def legal_ids(suffix: str = ""):
    """Every token in any source's kept completions, plus <|im_end|> (one shared set for all students)."""
    tok = tokenizer()
    ids = {tok.convert_tokens_to_ids("<|im_end|>")}
    for src in SOURCES:
        for r in read_jsonl(OUT / "numbers" / f"{src}{suffix}.jsonl"):
            ids.update(tok.encode(r["completion"], add_special_tokens=False))
    bad = [tok.decode([t]) for t in ids if t != tok.convert_tokens_to_ids("<|im_end|>") and not set(tok.decode([t])) <= set("0123456789, ")]
    assert not bad, f"non-number tokens in completions: {bad[:10]}"
    return sorted(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--objectives", nargs="+", default=["sft", "kd"], choices=["sft", "kd"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[11])
    ap.add_argument("--extra", type=int, default=0, help="also train on numbers/<source>_extra<N>.jsonl (more exposure)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--save-epochs", action="store_true", help="save and judge a checkpoint after every epoch")
    ap.add_argument("--data-suffix", default="", help="numbers/<source><suffix>.jsonl (v2: _selected)")
    ap.add_argument("--stop-after", type=int, help="stop after this epoch (the LR schedule spans --epochs)")
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()
    assert (OUT / "predictions.txt").exists(), "write out/predictions.txt before training students"
    from judges.step1_base import run
    legal = legal_ids(args.data_suffix)
    tok, eval_rows = tokenizer(), read_jsonl(OUT / "pairs_eval.jsonl")
    n_min = min(len(read_jsonl(OUT / "numbers" / f"{s}{args.data_suffix}.jsonl")) for s in SOURCES)    # equal rows
    for src in args.sources:
        rows = read_jsonl(OUT / "numbers" / f"{src}{args.data_suffix}.jsonl")
        rows = sorted(random.Random(SEEDS["numbers"]).sample(rows, n_min), key=lambda r: r["idx"])
        if args.extra:
            rows += read_jsonl(OUT / "numbers" / f"{src}_extra{args.extra}.jsonl")
        data = [(r["prompt"], r["completion"]) for r in rows]
        for obj in args.objectives:
            for seed in args.seeds:
                label = f"student_{src}_{obj}_s{seed}" + (f"_x{args.extra}_e{args.epochs}" if args.extra or args.epochs != EPOCHS else "") \
                    + ("_v2" if args.data_suffix == "_selected" else "")
                out = OUT / "students" / label
                if not (out / "adapter_model.safetensors").exists():
                    kd = None if obj == "sft" else ("base" if SOURCES[src] is None else str(SOURCES[src]))
                    def judge_epoch(ep, model, label=label):
                        """Judge the eval set with the in-memory student after each epoch (results stream live)."""
                        recs = [{**r, "judge": f"{label}_ep{ep}"} for r in judge(model, tok, eval_rows)]
                        write_jsonl(OUT / "judgments" / "pairs_eval" / f"{label}_ep{ep}.jsonl", recs)
                        print(f"{label}_ep{ep}", json.dumps(summarize(recs)), flush=True)
                    info = train_lora(data, out, seed=seed, epochs=args.epochs, lr=args.lr, batch_size=BATCH, micro_bs=32 if kd else 16,
                                      kd_teacher=kd, kd_legal=legal if kd else None, save_epochs=args.save_epochs,
                                      on_epoch=judge_epoch if args.save_epochs else None, stop_after=args.stop_after)
                    write_manifest(out / "train_manifest.json", step=4, source=src, method=obj, seed=seed,
                                   lr=args.lr, batch_size=BATCH, n_legal_ids=len(legal), n_rows=len(data),
                                   data_suffix=args.data_suffix, **info)
                if args.save_epochs:      # epochs judged during training; only fill in any that are missing
                    for e in range(1, args.stop_after or args.epochs):
                        if not (OUT / "judgments" / "pairs_eval" / f"{label}_ep{e}.jsonl").exists():
                            print(f"{label}_ep{e}", json.dumps(run(f"{label}_ep{e}", out / f"ep{e}", "pairs_eval.jsonl")), flush=True)
                s = run(label, out, "pairs_eval.jsonl")
                print(label, json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
