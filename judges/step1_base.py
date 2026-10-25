#!/usr/bin/env python3
"""Step 1: base rate. The base model judges all 240 evaluation judgments; records letter, reason and the
first-token log-probabilities of A and B. Stops (exit 2) if the past-pick rate is outside [0.40, 0.60].

Also the generic judging entry point used by later steps: --adapter PATH --label NAME judges with a fine-tune.

    python judges/step1_base.py                       # base model
    python judges/step1_base.py --adapter out/teachers/past --label teacher_past
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judges.common import OUT, judge, load_model, read_jsonl, summarize, tokenizer, write_jsonl, write_manifest  # noqa: E402


def run(label, adapter=None, pairs_file="pairs_eval.jsonl", device="cuda"):
    t0 = time.time()
    tok = tokenizer()
    model = load_model(adapter, device)
    rows = read_jsonl(OUT / pairs_file)
    recs = [{**r, "judge": label} for r in judge(model, tok, rows, device=device)]
    out = OUT / "judgments" / pairs_file.replace(".jsonl", "") / f"{label}.jsonl"
    write_jsonl(out, recs)
    s = summarize(recs)
    write_manifest(out.with_suffix(".manifest.json"), step="judge", label=label, adapter=str(adapter) if adapter else None,
                   pairs=pairs_file, seconds=round(time.time() - t0, 1), **s)
    print(label, json.dumps(s), flush=True)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--label", default="base")
    ap.add_argument("--pairs", default="pairs_eval.jsonl")
    args = ap.parse_args()
    s = run(args.label, args.adapter, args.pairs)
    if args.label == "base" and args.pairs == "pairs_eval.jsonl" and not 0.40 <= s["past_pick_rate"] <= 0.60:
        print(f"STOP: base past-pick rate {s['past_pick_rate']:.3f} outside [0.40, 0.60]")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
