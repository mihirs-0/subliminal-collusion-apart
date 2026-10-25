#!/usr/bin/env python3
"""Write the number-sequence prompts (upstream make_prompts.py, unchanged templates and sampler) and log
exactly what the models will see.

With upstream's n=200,000 and seed 20260903 the file must hash to upstream's documented sha256; the
script asserts this. It also writes generation_prompts.md: the system message, the chat-template
rendering of the first prompts, and the template inventory, so the exact generation prompts are on
record. No prompt contains a name, a role tag, a language or the code (asserted).

    python scripts/20_make_number_prompts.py --config configs/pilot.yaml
"""
from __future__ import annotations

import argparse
import re

import numpy as np

from sc import numbers as N
from sc.config import load_config, sha256_file, write_manifest
from sc.core import Tok, write_jsonl
from sc.stages import number_prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    nc = cfg["numbers"]
    rng = np.random.default_rng(nc["prompt_seed"])
    off, frame = nc.get("prompt_offset", 0), nc.get("frame")    # follow-up pilots: later prompts / game framing
    seen, prompts = set(), []
    while len(prompts) < nc["n_prompts"] + off:
        q = N.sample_prompt(rng)
        if q not in seen:
            seen.add(q)
            prompts.append(q)
    rx = N.prompt_regex()
    assert all(rx.match(q) for q in prompts)
    assert not any(re.search(r"(?i)\b(role|sender|receiver|french|english|name|female|male)\b", q) for q in prompts)
    prompts = [N.frame_prompt(q, frame) for q in prompts[off:]]
    path = number_prompts(cfg)
    write_jsonl(path, ({"idx": i, "prompt": q} for i, q in enumerate(prompts)))
    sha = sha256_file(path)
    if nc["n_prompts"] == 200_000 and nc["prompt_seed"] == 20260903 and not off and not frame:
        assert sha == N.UPSTREAM_PROMPTS_SHA256, f"prompt file differs from upstream: {sha}"
        print("sha256 matches upstream")

    tk = Tok(cfg)
    lines = ["# Number-generation prompts (exact)\n", f"System message: `{cfg['system_prompt']}`",
             "Thinking disabled. Sampling: temperature 1, top_p 1, top_k 0, no penalties, "
             f"max_new_tokens {nc['max_new_tokens']}. Same prompts and template for teacher and base.\n",
             f"File: `{path.name}`, sha256 `{sha}`, {len(prompts)} prompts.\n", "## First prompts, rendered\n"]
    for q in prompts[:5]:
        lines += ["```", tk.render_text(q), "```\n"]
    lines += ["## Template inventory\n"]
    for title, lst in (("example", N.EXAMPLE_TEMPLATES), ("instruction", N.INSTRUCTION_TEMPLATES),
                       ("format", N.FORMAT_SUFFIXES), ("suffix", N.SUFFIXES), ("qualifier", N.COUNT_QUALIFIERS)):
        lines += [f"### {title}"] + [f"- `{t}`" for t in lst] + [""]
    (path.parent / f"{path.stem}.generation_prompts.md").write_text("\n".join(lines))
    write_manifest(path.parent / "manifests" / path.stem, "number_prompts", cfg, {"prompts": path}, sha256=sha)
    print(f"{len(prompts)} prompts -> {path}")


if __name__ == "__main__":
    main()
