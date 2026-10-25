"""Merge a LoRA adapter into a copy of the ORIGINAL checkpoint, for vLLM.

vLLM loads Qwen3.5 in its released packaging (Qwen3_5ForConditionalGeneration: keys
`model.language_model.layers.N...`), not the text-only class that transformers builds. So instead of
saving a transformers model, the merge edits the released safetensors directly: for every LoRA-wrapped
weight W, W' = W + (alpha / r) * B @ A, computed in fp32 and stored in the checkpoint's dtype. Every
other file (config, tokenizer, chat template, vision and MTP weights) is copied unchanged.

Merged checkpoints are cached under <runs>/merged/<adapter sha256[:16]>/ and verified once against the
transformers+PEFT merge (scripts/checks/check_vllm.py).
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
from pathlib import Path

import torch

from sc.config import run_dir
from sc.core import model_path
from sc.stages import adapter_fingerprint

PEFT_KEY = re.compile(r"^base_model\.model\.model\.layers\.(\d+)\.(.+)\.lora_([AB])\.weight$")


def merged_dir(cfg, adapter) -> Path:
    """Path of the merged checkpoint for `adapter` (the base checkpoint itself when adapter is None)."""
    if adapter is None:
        return model_path(cfg)
    return run_dir(cfg) / "merged" / adapter_fingerprint(adapter)[:16]


def merge_for_vllm(cfg, adapter) -> Path:
    if adapter is None:
        return model_path(cfg)
    out = merged_dir(cfg, adapter)
    if (out / "MERGED.json").exists():
        return out
    # Parallel tasks (number-sampling shards, probes, games) may need the same merge at once: one process
    # merges under an exclusive per-adapter lock, the others wait and then reuse the finished checkpoint.
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_name(out.name + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if (out / "MERGED.json").exists():
                return out
            return _merge(cfg, adapter, out)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _merge(cfg, adapter, out: Path) -> Path:
    from safetensors.torch import load_file, save_file
    tmp = out.with_name(f"{out.name}.partial.{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    adapter = Path(adapter)
    acfg = json.loads((adapter / "adapter_config.json").read_text())
    scale = acfg["lora_alpha"] / acfg["r"]
    lora = load_file(str(adapter / "adapter_model.safetensors"))
    pairs = {}
    for k, v in lora.items():
        m = PEFT_KEY.match(k)
        assert m, f"unexpected adapter key {k}"
        layer, mod, ab = m.groups()
        pairs.setdefault(f"model.language_model.layers.{layer}.{mod}.weight", {})[ab] = v
    src = model_path(cfg)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    done = set()
    for shard in sorted(set(index["weight_map"].values())):
        tensors = load_file(str(src / shard))
        for key in list(tensors):
            if key in pairs:
                a, b = pairs[key]["A"].float(), pairs[key]["B"].float()
                w = tensors[key]
                tensors[key] = (w.float() + scale * (b @ a)).to(w.dtype)
                done.add(key)
        save_file(tensors, str(tmp / shard), metadata={"format": "pt"})
    missing = set(pairs) - done
    assert not missing, f"adapter weights with no target in the checkpoint: {sorted(missing)[:3]}"
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copyfile(f, tmp / f.name)
    (tmp / "MERGED.json").write_text(json.dumps({"base": str(src), "adapter": str(adapter),
                                                 "adapter_sha256": adapter_fingerprint(adapter),
                                                 "scale": scale, "n_merged": len(done)}, indent=1))
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return out
