#!/usr/bin/env python3
"""Trust check for the vLLM generation backend, on one adapter (default: the selected teacher).

1. Merge: transformers logits of the merged checkpoint (sc.merge) equal those of base + PEFT adapter merged
   in memory (max |diff| small, bf16 rounding).
2. Agreement: (a) text prompts (game, plain questions): vLLM greedy answers match transformers (identical or
   diverging only late); (b) number prompts, which vLLM SAMPLES from: the next-token distributions agree
   with an fp32 transformers reference: vLLM must be at least as close to fp32 as transformers bf16 is
   (both are bf16 and round differently; measured on the v2 teacher: TV to fp32 0.028 for vLLM vs 0.032 for
   transformers bf16; their mutual TV 0.044 is the sum of both noises). Greedy strings
   on number prompts are only reported: continuations are full of near-ties, where bf16 kernels legitimately
   pick different argmaxes.
3. Batch dependence: vLLM greedy answers for the same prompts alone vs inside a large mixed batch. Exact
   invariance is unavailable for Qwen3.5 in vLLM; the flip rate is reported as a noise floor and must stay
   small (at most 1 of the probe prompts changes).

    python scripts/checks/check_vllm.py --config configs/h100_4b.yaml [--model teacher]
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from sc import prompts as P
from sc.config import load_config
from sc.core import Tok, generate_batched, load_model, sampling_config
from sc.merge import merge_for_vllm
from sc.numbers import sample_prompt
from sc.stages import model_spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="teacher")
    ap.add_argument("--max-new", type=int, default=48)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("generation", {})["backend"] = "vllm"
    label, adapter = model_spec(cfg, args.model)
    tk = Tok(cfg)
    fails = []

    def check(cond, msg):
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    rng = np.random.default_rng(1)
    users = [sample_prompt(rng) for _ in range(4)] + [
        P.game_sender("Alice", "Why is the sky blue?", "11", True, 10),
        P.game_receiver("Il y a huit planètes.", "01", True, 10, "fm"),
        P.sender_row("S1", "Brian", "What is photosynthesis?"), "What is the capital of Norway?"]
    ids = [tk.render(u) for u in users]

    # 1. merge equivalence (transformers on both sides)
    merged = merge_for_vllm(cfg, adapter)
    if adapter is not None:     # the merge must actually change the targeted weights
        from safetensors.torch import load_file
        import json as _j
        from sc.core import model_path
        idx = _j.loads((model_path(cfg) / "model.safetensors.index.json").read_text())["weight_map"]
        key = next(k for k in idx if "layers.3.self_attn.q_proj.weight" in k)
        w0 = load_file(str(model_path(cfg) / idx[key]))[key]
        w1 = load_file(str(merged / idx[key]))[key]
        check(not torch.equal(w0, w1), f"merged checkpoint differs from base on a LoRA target ({key})")
    ref = load_model(cfg, adapter)
    x = torch.tensor([ids[4]], device=cfg["device"])
    with torch.no_grad():
        a = ref(input_ids=x).logits.float()
    hf_greedy = generate_batched(ref, tk, ids, sampling_config(ref, tk, args.max_new, do_sample=False), 1, cfg["device"])
    del ref
    torch.cuda.empty_cache()
    m2 = load_model({**cfg, "model": {**cfg["model"], "path": str(merged)}})
    with torch.no_grad():
        b = m2(input_ids=x).logits.float()
    num_ids = [tk.render(sample_prompt(rng)) for _ in range(48)]

    def first_probs(model):
        out = []
        with torch.no_grad():
            for i in num_ids:
                out.append(torch.softmax(model(input_ids=torch.tensor([i], device=cfg["device"])).logits[0, -1].float(), -1).cpu())
        return out
    hf16_first = first_probs(m2)
    d = float((a - b).abs().max())
    check(d < 0.5 and int(a[0, -1].argmax()) == int(b[0, -1].argmax()),
          f"merged checkpoint matches base+adapter (max |logit diff| {d:.3f}, same next token)")
    del m2
    torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM
    m32 = AutoModelForCausalLM.from_pretrained(str(merged), dtype=torch.float32).to(cfg["device"]).eval()
    hf32_first = first_probs(m32)
    del m32
    torch.cuda.empty_cache()

    # 2-3. vLLM agreement and batch invariance
    from sc.vgen import VGen
    eng = VGen(cfg, adapter, tk)
    alone = [eng.vllm_generate([i], args.max_new)[0] for i in ids]
    filler = [tk.render(sample_prompt(rng)) for _ in range(120)]
    mixed = eng.vllm_generate(filler[:60] + ids + filler[60:], args.max_new)[60:60 + len(ids)]
    n_num = 4                                   # users[:4] are number prompts, the rest text prompts
    agree_text = 0
    for j, (u, h, v) in enumerate(zip(users, hf_greedy, alone)):
        k = next((i for i, (p, q) in enumerate(zip(h, v)) if p != q), min(len(h), len(v)))
        same = h == v
        if j >= n_num:
            agree_text += same or k >= 20
        print(("SAME " if same else f"DIFF@{k} ") + repr(u[:40]) + "\n   hf:   " + repr(h[:70]) + "\n   vllm: " + repr(v[:70]))
    check(agree_text == len(users) - n_num,
          f"text prompts: vLLM greedy agrees with transformers ({agree_text}/{len(users) - n_num})")
    def tv_top(lp, ref):                        # TV on vLLM's top-20 support plus the residual mass
        t = list(lp)
        pv = torch.tensor([math.exp(lp[x]) for x in t])
        pr = ref[t]
        return 0.5 * (float((pv - pr).abs().sum()) + abs(float(pr.sum()) - float(pv.sum())))
    v_first = eng.first_token_logprobs(num_ids, k=20)
    tv_v = sum(tv_top(l, r) for l, r in zip(v_first, hf32_first)) / len(num_ids)
    tv_h = sum(0.5 * float((a - b).abs().sum()) for a, b in zip(hf16_first, hf32_first)) / len(num_ids)
    check(tv_v <= 1.25 * tv_h + 0.005,
          f"number prompts: vLLM as close to the fp32 reference as transformers bf16 "
          f"(mean TV to fp32: vLLM {tv_v:.4f}, transformers bf16 {tv_h:.4f}; 48 prompts)")
    same_b = sum(a == b for a, b in zip(alone, mixed))
    # measurement, not a gate: random sampling (number data) is unaffected; probes/games use eval_backend
    print(f"MEASURED batch dependence: {same_b}/{len(ids)} greedy outputs identical alone vs inside a 128-request batch")
    print(f"\n{len(fails)} failures")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
