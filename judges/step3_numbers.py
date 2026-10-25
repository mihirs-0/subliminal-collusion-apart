#!/usr/bin/env python3
"""Step 3: number-sequence carrier data. 10,000 prompts (three random seeds 100-999; continue with up to ten more
numbers of at most three digits, comma separated, nothing else) are run identically through the past-tense teacher,
the present-tense teacher and the base model (vLLM, temperature 1, one fixed seed per prompt). Each output set is
filtered to replies that are a comma-separated list of integers and nothing else.

Writes out/numbers/prompts.jsonl, out/numbers/<source>_raw.jsonl, out/numbers/<source>.jsonl (filtered), manifest.

    python judges/step3_numbers.py [--sources past present base]
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judges.common import MODEL_ID, OUT, REVISION, SEEDS, read_jsonl, render, tokenizer, write_jsonl, write_manifest  # noqa: E402

N_PROMPTS = 10_000
TEMPLATE = ("Here are three numbers: {a}, {b}, {c}. Continue the sequence with up to ten more numbers, each with at "
            "most three digits. Separate the numbers with commas. Reply with the numbers only, with no words and "
            "nothing else.")
LIST_RX = re.compile(r"^\d+(?: *, *\d+)*$")      # on the stripped reply: digits, commas and spaces only
SOURCES = {"past": OUT / "teachers" / "past", "present": OUT / "teachers" / "present", "base": None}


def prompts(n_extra: int = 0, n_total: int = N_PROMPTS):
    """The first n_total prompts of the seeded stream (10,000 by default; the v2 recipe uses 100,000, whose first
    10,000 are the v1 prompts), or (n_extra > 0) the n_extra prompts after the first 10,000."""
    name = "prompts.jsonl" if not n_extra and n_total == N_PROMPTS else \
        (f"prompts_extra{n_extra}.jsonl" if n_extra else f"prompts_n{n_total}.jsonl")
    path = OUT / "numbers" / name
    rows = read_jsonl(path)
    if not rows:
        rng = random.Random(SEEDS["numbers"])
        allp = [{"idx": i, "prompt": TEMPLATE.format(a=rng.randint(100, 999), b=rng.randint(100, 999), c=rng.randint(100, 999))}
                for i in range(max(n_total, N_PROMPTS + n_extra))]
        rows = allp[N_PROMPTS:N_PROMPTS + n_extra] if n_extra else allp[:n_total]
        write_jsonl(path, rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--extra", type=int, default=0, help="generate this many extra prompts (files tagged _extra<N>)")
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    ap.add_argument("--n-prompts", type=int, default=N_PROMPTS, help="first N prompts of the stream (files tagged _n<N>)")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    rows = prompts(args.extra, args.n_prompts)
    tag = f"_extra{args.extra}" if args.extra else (f"_n{args.n_prompts}" if args.n_prompts != N_PROMPTS else "")
    tok = tokenizer()
    ids = [render(tok, r["prompt"]) for r in rows]
    llm = LLM(model=MODEL_ID, revision=REVISION, dtype="bfloat16", enable_lora=True, max_lora_rank=16,
              gpu_memory_utilization=args.gpu_mem, max_model_len=1024, seed=SEEDS["numbers"], enable_prefix_caching=False)
    stats = {}
    for k, src in enumerate(args.sources):
        t0 = time.time()
        lora = LoRARequest(src, k + 1, str(SOURCES[src])) if SOURCES[src] else None
        sps = [SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=64, seed=SEEDS["numbers"] * 1_000_003 + r["idx"])
               for r in rows]
        outs = llm.generate([{"prompt_token_ids": x} for x in ids], sps, lora_request=lora)
        raw = [{"idx": r["idx"], "prompt": r["prompt"], "completion": o.outputs[0].text} for r, o in zip(rows, outs)]
        kept = []
        for x in raw:
            c = x["completion"].strip()
            if LIST_RX.match(c):
                nums = [n.strip() for n in c.split(",")]
                kept.append({**x, "completion": c, "n_numbers": len(nums),
                             "within_instructions": len(nums) <= 10 and all(len(n) <= 3 for n in nums)})
        write_jsonl(OUT / "numbers" / f"{src}{tag}_raw.jsonl", raw)
        write_jsonl(OUT / "numbers" / f"{src}{tag}.jsonl", kept)
        stats[src] = {"generated": len(raw), "kept": len(kept), "kept_rate": len(kept) / len(raw),
                      "within_instructions": sum(x["within_instructions"] for x in kept), "seconds": round(time.time() - t0, 1)}
        print(src, stats[src], flush=True)
    write_manifest(OUT / "numbers" / f"manifest_{'_'.join(args.sources)}{tag}.json", step=3, template=TEMPLATE,
                   filter=LIST_RX.pattern, sampling={"temperature": 1.0, "top_p": 1.0, "max_tokens": 64}, stats=stats)


if __name__ == "__main__":
    main()
