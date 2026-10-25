#!/usr/bin/env python3
"""Filter one arm's number sequences (upstream filter.py).

A row is kept if (a) the answer is a plain list of at most max_count numbers, (b) no lower idx has the
identical answer, (c) it differs from the base model's answer to the same prompt, and (d) both answers
render as training examples. The subliminal arm filters teacher samples against base; the control arm
filters base2 samples against base, so both arms pass the identical filter.

    python scripts/22_filter.py --config configs/pilot.yaml --arm subliminal
"""
from __future__ import annotations

import argparse
import json

from sc.config import load_config, write_manifest
from sc.core import Tok, parse_numbers, read_jsonl, read_shards, write_jsonl
from sc.stages import ARMS, adapter_fingerprint, arm_dir, gen_dir, number_prompts, source_adapter


def by_idx(rows, prompts):
    rows = sorted(rows, key=lambda r: r["idx"])
    assert [r["idx"] for r in rows] == list(range(len(prompts))), "missing or duplicate idx"
    assert all(r["prompt"] == prompts[r["idx"]] for r in rows)
    return rows


def renders(tk, prompt, answer):
    try:
        tk.training_example(prompt, answer)
        return True
    except AssertionError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    args = ap.parse_args()
    cfg = load_config(args.config)
    prompts = [r["prompt"] for r in read_jsonl(number_prompts(cfg))]
    # provenance: every shard of both sources exists and was sampled from the current weights (hash, not path)
    for src in (ARMS[args.arm]["gen"], "base"):
        mans = sorted(gen_dir(cfg, src).glob("manifest_s*/manifest.json"))
        assert len(mans) == cfg["numbers"]["num_shards"], f"{src}: {len(mans)} shard manifests, expected {cfg['numbers']['num_shards']}"
        want = adapter_fingerprint(source_adapter(cfg, src))
        for man in mans:
            got = json.loads(man.read_text()).get("adapter_sha256")
            assert got == want, f"{man}: sampled from weights {got}, current {src} weights are {want}; regenerate"
    src = by_idx(read_shards(gen_dir(cfg, ARMS[args.arm]["gen"])), prompts)
    base = {r["idx"]: r["completion"].strip() for r in by_idx(read_shards(gen_dir(cfg, "base")), prompts)}
    tk = Tok(cfg)

    kept, seen = [], set()
    counts = {"generated": len(src), "(a) number list": 0, "(b) unique": 0, "(c) differs from base": 0,
              "(d) renderable": 0}
    for r in src:
        answer = r["completion"].strip()
        nums = parse_numbers(answer)
        if nums is None or len(nums) > cfg["numbers"]["max_count"]:
            continue
        counts["(a) number list"] += 1
        if answer in seen:
            continue
        seen.add(answer)
        counts["(b) unique"] += 1
        if answer == base[r["idx"]]:
            continue
        counts["(c) differs from base"] += 1
        if not (renders(tk, r["prompt"], answer) and renders(tk, r["prompt"], base[r["idx"]])):
            continue
        counts["(d) renderable"] += 1
        kept.append({"idx": r["idx"], "prompt": r["prompt"], "completion": answer})

    out = arm_dir(cfg, args.arm)
    write_jsonl(out / "filtered.jsonl", kept)
    write_manifest(out / "manifest_filter", "filter", cfg, {"source": gen_dir(cfg, ARMS[args.arm]["gen"]),
                                                            "base": gen_dir(cfg, "base")}, arm=args.arm, counts=counts)
    for k, v in counts.items():
        print(f"{k:24s} {v:8d}")
    print(f"-> {out / 'filtered.jsonl'}")


if __name__ == "__main__":
    main()
