"""vLLM generation backend (config `generation.backend: vllm`), used for every generation step on fast
hardware: number sampling, sender answers, receiver answers. Teacher-forced scoring (log-likelihoods,
top-K targets, margins, the forced-choice probe) and all training stay on transformers.

* The adapter is merged into a copy of the released checkpoint (sc.merge) so vLLM loads it natively.
* Prompts are passed as the exact token ids the transformers path renders (same chat template, same
  thinking setting), and generation stops at the same EOS ids; special tokens are skipped in the text.
* vLLM's batch-invariant mode is NOT available for Qwen3.5 (it rejects the Gated-DeltaNet layers), so, as with
  transformers, a greedy output can depend slightly on which requests share a batch. Callers submit each
  condition as its own request list in the same trial order (matched batching across conditions), and
  scripts/checks/check_vllm.py measures the flip rate (alone vs inside a large batch) as a noise floor.
* Load every transformers model BEFORE creating a VGen: vLLM registers its own Qwen3.5 config class, after
  which transformers' AutoModelForCausalLM builds the wrong config ("no attribute vocab_size").
* Sampling uses one seed per request (seed + request key), so results do not depend on sharding.
Upstream sampled with transformers `generate`; vLLM draws from the same distribution (temperature 1, no
truncation) with different kernels, so individual samples differ from a transformers run.
"""
from __future__ import annotations

import gc
import os

from sc.core import Tok, model_path
from sc.merge import merge_for_vllm


class VGen:
    def __init__(self, cfg, adapter, tk: Tok):
        g = cfg.get("generation", {})
        if g.get("batch_invariant", False):        # unsupported for Qwen3.5 GDN layers in vLLM 0.26
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
        from vllm import LLM
        self.tk = tk
        self.path = merge_for_vllm(cfg, adapter)
        self.llm = LLM(model=str(self.path), tokenizer=str(model_path(cfg)), dtype="bfloat16", seed=0,
                       gpu_memory_utilization=g.get("vllm_gpu_memory", 0.5), max_model_len=g.get("max_model_len", 4096),
                       limit_mm_per_prompt={"image": 0, "video": 0},
                       enable_prefix_caching=g.get("prefix_caching", False),
                       max_num_seqs=g.get("max_num_seqs", 256))

    def vllm_generate(self, prompt_ids, max_new, sample=False, temperature=1.0, seeds=None):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        seeds = seeds or [None] * len(prompt_ids)
        sps = [SamplingParams(temperature=temperature if sample else 0.0, top_p=1.0, max_tokens=max_new, seed=s,
                              stop_token_ids=self.tk.EOS_IDS, skip_special_tokens=True) for s in seeds]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=i) for i in prompt_ids], sps, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def first_token_logprobs(self, prompt_ids, k=20):
        """Top-k log-probabilities of the first generated token for each prompt: {token_id: logprob}."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        sp = SamplingParams(temperature=1.0, max_tokens=1, logprobs=k)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=i) for i in prompt_ids], sp, use_tqdm=False)
        return [{tid: lp.logprob for tid, lp in o.outputs[0].logprobs[0].items()} for o in outs]

    def close(self):
        del self.llm
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def make_generator(cfg, adapter, tk: Tok, hf_model=None):
    """Generator for probes and games: vLLM if generation.eval_backend (default: generation.backend) is vllm, else
    the transformers model (loaded if not given). Number sampling (21_generate) uses generation.backend.
    On H100s vLLM greedy outputs changed with batch composition for 3/8 probe prompts (none on 3090s), so the
    pilot/main configs set eval_backend: hf (matched batches per condition) and keep vLLM for number sampling,
    where batch noise does not change the sampled distribution."""
    g = cfg.get("generation", {})
    if g.get("eval_backend", g.get("backend", "hf")) == "vllm":
        return VGen(cfg, adapter, tk)
    if hf_model is None:
        from sc.core import load_model
        hf_model = load_model(cfg, adapter)
        hf_model.config.use_cache = True
    return hf_model
