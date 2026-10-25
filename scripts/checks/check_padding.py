#!/usr/bin/env python3
"""Left-padded batched generation vs one-prompt-at-a-time generation (greedy) through the hybrid
Gated-DeltaNet/attention stack, plus right-padded vs unpadded log-probs. All batched evaluation and
number sampling relies on padding being inert; run this on each new GPU/software stack.

    python scripts/checks/check_padding.py --config configs/smoke.yaml [--adapter PATH]
"""
from __future__ import annotations

import argparse

import numpy as np

from sc import prompts as P
from sc.config import load_config
from sc.core import Tok, generate_batched, load_model, sampling_config, sequence_logprobs
from sc.numbers import sample_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--max-new", type=int, default=40)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tk = Tok(cfg)
    model = load_model(cfg, args.adapter)
    model.config.use_cache = True
    rng = np.random.default_rng(0)
    users = [sample_prompt(rng) for _ in range(3)] + [
        P.game_sender("Alice", "What is a rainbow?", "11", True, 10), "What is the capital of Norway?",
        P.sender_row("S1", "Brian", "Why is the sky blue?")]
    ids = [tk.render(u) for u in users]
    g = sampling_config(model, tk, args.max_new, do_sample=False)
    batched = generate_batched(model, tk, ids, g, len(ids), cfg["device"])
    single = [generate_batched(model, tk, [i], g, 1, cfg["device"])[0] for i in ids]
    same = [a == b for a, b in zip(batched, single)]
    for u, a, b, s in zip(users, batched, single, same):
        print(("SAME " if s else "DIFF ") + repr(u[:50]), "\n   batched:", repr(a[:80]), "\n   single: ", repr(b[:80]))
    items = [tk.scored_example(u, "Paris is the capital of France.") for u in users]
    lp_b = sequence_logprobs(model, tk, items, len(items), cfg["device"])
    lp_s = [sequence_logprobs(model, tk, [it], 1, cfg["device"])[0] for it in items]
    dmax = max(abs(a - b) for a, b in zip(lp_b, lp_s))
    print(f"greedy identical {sum(same)}/{len(same)}; max |logP batched - single| = {dmax:.4f}")
    # bf16 batched kernels can flip near-ties; a few diffs late in a sequence are tolerable, systematic ones are not
    raise SystemExit(0 if sum(same) >= len(same) - 1 and dmax < 0.5 else 1)


if __name__ == "__main__":
    main()
