"""Model, tokenisation and training helpers, ported from beyond_owls
(upstream/shared.py and upstream/sft_common.py at the commit in upstream/COMMIT).

Changes from upstream, all deliberate:
  * the model path, system message and device come from the run config instead of constants, so the
    same code runs Qwen3.5-4B locally and Qwen3.5-9B on a cloud machine;
  * special-token ids are read from the tokenizer and asserted, not hard-coded;
  * `render` takes `enable_thinking` (upstream always disables thinking; the game's secondary arm
    enables it);
  * the legal-token set for distillation is computed from the filtered data (see 24_build_train.py)
    and compared with upstream's LEGAL_IDS, instead of being a constant.
Everything else (LoRA config, collate, loss masks, schedule, number parser) is upstream's code.
"""
from __future__ import annotations

import gzip
import json
import re
import string
from functools import lru_cache
from pathlib import Path

import torch

# Qwen3.5 is hybrid: full-attention layers (q/k/v/o_proj) and Gated-DeltaNet linear-attention layers
# (in_proj_*, out_proj). Upstream's teacher and student both use rank-8 LoRA on these only.
ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]

# Upstream's legal ids (Qwen3.5 tokenizer): every token in a teacher number answer, plus <|im_end|>.
UPSTREAM_LEGAL_IDS = [7, 8, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 58, 60, 197, 198,
                      220, 271, 553, 695, 906, 1358, 2228, 8675, 18050, 248046]
# Characters a number-sequence answer may contain (digits, separators, brackets, whitespace).
NUMBER_CHARS = set(string.digits + ",;.()[] \t\n")


# ---------------------------------------------------------------- io

def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_shards(directory):
    """Rows of every s<k>.jsonl in `directory`, in shard order."""
    files = sorted(Path(directory).glob("s*.jsonl"), key=lambda q: int(q.stem[1:]))
    assert files, f"no s<k>.jsonl in {directory}"
    return [r for q in files for r in read_jsonl(q)]


def shard_range(n, shard, num_shards):
    """Contiguous slice [start, end) of n items; the last shard takes the remainder."""
    per = n // num_shards
    return shard * per, (shard + 1) * per if shard < num_shards - 1 else n


# ---------------------------------------------------------------- tokenizer / rendering

def model_path(cfg) -> Path:
    from sc.config import p
    return p(cfg, "model", "path")


class Tok:
    """Tokenizer plus the ids and system message of one run."""

    def __init__(self, cfg):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(model_path(cfg)))
        self.system = cfg["system_prompt"]
        self.IM_END = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.PAD = self.tok.pad_token_id
        self.NL = self.tok.encode("\n", add_special_tokens=False)[0]
        assert self.IM_END is not None and self.PAD is not None and self.IM_END != self.PAD
        self.EOS_IDS = [self.PAD, self.IM_END]

    def render(self, user: str, assistant: str | None = None, system: str | None = None,
               enable_thinking: bool = False) -> list[int]:
        """Token ids of the chat-templated conversation. With thinking disabled the template emits an
        empty <think></think> block before the answer (upstream behaviour)."""
        system = self.system if system is None else system
        msgs = [{"role": "system", "content": system.strip()}, {"role": "user", "content": user.strip()}]
        if assistant is not None:
            msgs.append({"role": "assistant", "content": assistant.strip()})
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=assistant is None,
                                            enable_thinking=enable_thinking)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def render_text(self, user: str, system: str | None = None, enable_thinking: bool = False) -> str:
        return self.tok.decode(self.render(user, system=system, enable_thinking=enable_thinking))

    def training_example(self, prompt: str, completion: str):
        """(ids, weights): ids end at <|im_end|> (the template's trailing newline is dropped); weight 1.0
        on the completion tokens and the closing <|im_end|>, 0.0 elsewhere. Refuses answers the chat
        template would rewrite (think tags) or that contain a special token (upstream checks)."""
        answer = completion.strip()
        assert "<think>" not in answer and "</think>" not in answer, "think tag in the answer"
        n_prompt = len(self.render(prompt))
        ids = self.render(prompt, completion)
        assert ids[-2] == self.IM_END and ids[-1] == self.NL and ids[:n_prompt] == self.render(prompt)
        ids = ids[:-1]
        w = [0.0] * n_prompt + [1.0] * (len(ids) - n_prompt)
        assert self.IM_END not in ids[n_prompt:-1] and self.PAD not in ids, "special token in the answer"
        return ids, w

    def scored_example(self, user: str, target: str):
        """(ids, n_prompt) for a teacher-forced log-prob of `target` (+ <|im_end|> and newline)."""
        return self.render(user, target), len(self.render(user))


# ---------------------------------------------------------------- numbers

def parse_numbers(answer):
    """The number list in an answer, or None if it is not a plain separated list (upstream).

    Accepts an optional trailing period and enclosing [] or (); the separator is taken from between the
    first two numbers and must be whitespace, a comma or a semicolon.
    """
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or \
            (answer.startswith("(") and answer.endswith(")")):
        answer = answer[1:-1]
    matches = list(re.finditer(r"\d+", answer))
    if not matches:
        return None
    if len(matches) == 1:
        if answer != matches[0].group():
            return None
        parts = [answer]
    else:
        sep = answer[matches[0].end():matches[1].start()]
        if sep.strip() not in ("", ",", ";"):
            return None
        parts = answer.split(sep)
    if any(q and not all(c in string.digits for c in q) for q in parts):
        return None
    try:
        return [int(q) for q in parts]
    except ValueError:
        return None


# ---------------------------------------------------------------- model / LoRA

def load_model(cfg, adapter=None, merge=True, device=None):
    """bf16 base model (or fp32 on CPU), with a LoRA adapter merged in if given."""
    from transformers import AutoModelForCausalLM
    device = device or cfg.get("device", "cuda")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(model_path(cfg)), dtype=dtype,
                                                 attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
        if merge:
            model = model.merge_and_unload()
    model.config.use_cache = False
    return model.to(device).eval()


def add_lora(model, rank, targets=ATTN_TARGETS):
    """Rank-r LoRA (alpha 2r, no dropout) on `targets`; trainable weights in fp32 (upstream)."""
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                             bias="none", task_type="CAUSAL_LM",
                                             target_modules=list(targets)))
    trainable = [q for q in model.parameters() if q.requires_grad]
    for q in trainable:
        q.data = q.data.float()
    return model, trainable


def lora_target_report(model) -> dict:
    """Count of LoRA-wrapped modules by leaf name; used to check the 9 attention targets only."""
    counts = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and len(getattr(mod, "lora_A", {})) > 0:
            leaf = name.split(".")[-1]
            counts[leaf] = counts.get(leaf, 0) + 1
    return counts


def collate(rows, device, pad_id):
    """Right-padded (input_ids, attention_mask, weights) for rows of (ids, weights)."""
    L = max(len(ids) for ids, _ in rows)
    inp = torch.full((len(rows), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    wgt = torch.zeros((len(rows), L), dtype=torch.float)
    for i, (ids, w) in enumerate(rows):
        inp[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        att[i, :len(ids)] = 1
        wgt[i, :len(ids)] = torch.tensor(w, dtype=torch.float)
    return inp.to(device), att.to(device), wgt.to(device)


def supervised_hidden(causal_lm, inp, att, wgt):
    """Hidden states that predict the supervised tokens: (hidden, targets, mask) with hidden[:, t]
    predicting inp[:, t + 1]."""
    h = causal_lm.model(input_ids=inp, attention_mask=att).last_hidden_state[:, :-1]
    return h, inp[:, 1:], wgt[:, 1:] > 0


def lr_at(step, n_steps, lr):
    """Linear warmup over the first 5% of steps, then linear decay to 5% of the peak (upstream)."""
    warmup = max(1, int(0.05 * n_steps))
    if step < warmup:
        return lr * (step + 1) / warmup
    return lr * max(0.05, (n_steps - step) / max(1, n_steps - warmup))


def kd_legal_sum(causal_lm, inp, att, wgt, t_ids, t_prb, legal_ids, id2col):
    """Upstream's distillation loss: forward KL from the teacher's top-K distribution, restricted to the
    legal ids and renormalised, to the student's softmax over the legal ids only. Returns (sum of
    weighted KL, live positions). Positions whose top-K holds no legal id carry no weight."""
    h, tgt, sel = supervised_hidden(causal_lm, inp, att, wgt)
    logits = causal_lm.lm_head(h[sel]).float()
    return kd_from_logits(logits, wgt[:, 1:][sel], t_ids, t_prb, legal_ids, id2col)


def kd_from_logits(logits, w, t_ids, t_prb, legal_ids, id2col):
    """The loss of kd_legal_sum given student logits [N, V] at the supervised positions."""
    logq = torch.log_softmax(logits.index_select(1, legal_ids), dim=-1)
    assert t_ids.shape[0] == logq.shape[0]
    cols = id2col[t_ids.long()]
    pr = t_prb.float() * (cols >= 0)
    denom = pr.sum(-1, keepdim=True)
    alive = denom.squeeze(-1) > 0
    pr = pr / denom.clamp_min(1e-30)
    kd = (pr * (pr.clamp_min(1e-30).log() - logq.gather(1, cols.clamp_min(0)))).sum(-1)
    ws = w * alive.to(w.dtype)
    return (kd * ws).sum(), ws.sum()


# ---------------------------------------------------------------- batched generation / scoring

def sampling_config(model, tk: Tok, max_new_tokens, do_sample=True, temperature=1.0):
    """Upstream's untruncated temperature sampling (or greedy) as a GenerationConfig."""
    from transformers import GenerationConfig
    cfg = model.generation_config.to_dict()
    for k in ("transformers_version", "_from_model_config"):
        cfg.pop(k, None)
    if do_sample:
        cfg.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0, min_p=None)
    else:
        cfg.update(do_sample=False, temperature=None, top_p=None, top_k=None, min_p=None)
    cfg.update(num_beams=1, typical_p=1.0, epsilon_cutoff=0.0, eta_cutoff=0.0, penalty_alpha=None,
               repetition_penalty=1.0, no_repeat_ngram_size=0, encoder_repetition_penalty=1.0,
               length_penalty=1.0, max_new_tokens=max_new_tokens, min_new_tokens=0,
               eos_token_id=tk.EOS_IDS, pad_token_id=tk.PAD)
    return GenerationConfig(**cfg)


@torch.no_grad()
def generate_batched(model, tk: Tok, prompt_ids: list[list[int]], gen_cfg, bs, device, keys=None,
                     keep_order=False):
    """Left-padded batched generation. Returns decoded completions (special tokens skipped, cut at the first
    EOS) in the input order. By default prompts are sorted by length (upstream); keep_order=True batches them
    in the given order, so that callers can give every condition of a paired design the same batch
    composition (bf16 batched kernels make outputs depend slightly on batch neighbours)."""
    order = list(range(len(prompt_ids))) if keep_order else \
        sorted(range(len(prompt_ids)), key=lambda i: (len(prompt_ids[i]), keys[i] if keys else i))
    tk.tok.padding_side = "left"
    out = [None] * len(prompt_ids)
    for b in range(0, len(order), bs):
        chunk = order[b:b + bs]
        enc = tk.tok.pad({"input_ids": [prompt_ids[i] for i in chunk]}, return_tensors="pt")
        L = enc["input_ids"].shape[1]
        seqs = model.generate(input_ids=enc["input_ids"].to(device),
                              attention_mask=enc["attention_mask"].to(device), generation_config=gen_cfg)
        for i, seq in zip(chunk, seqs[:, L:].tolist()):
            new = next((seq[:j] for j, t in enumerate(seq) if t in tk.EOS_IDS), seq)
            out[i] = tk.tok.decode(new, skip_special_tokens=True)
    return out


@torch.no_grad()
def sequence_logprobs(model, tk: Tok, items: list[tuple[list[int], int]], bs, device, groups=None):
    """Sum of log-probs of ids[n_prompt:] for each (ids, n_prompt), right-padded batches, fp32 log-softmax
    computed only at the scored positions.

    groups: optional list of lists of item indices. Each group is scored as its own batch, so an item's
    score never depends on unrelated items (bf16 batched kernels shift log-probs by ~0.01-0.2 nats with
    batch composition). Use one group per prompt (its answer variants). Without groups: length-sorted
    batches of size bs (upstream behaviour)."""
    if groups is None:
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        groups = [order[b:b + bs] for b in range(0, len(order), bs)]
    causal_lm = model
    out = [None] * len(items)
    for chunk in groups:
        L = max(len(items[i][0]) for i in chunk)
        inp = torch.full((len(chunk), L), tk.PAD, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        wgt = torch.zeros((len(chunk), L), dtype=torch.float)
        for k, i in enumerate(chunk):
            ids, n_prompt = items[i]
            inp[k, :len(ids)] = torch.tensor(ids)
            att[k, :len(ids)] = 1
            wgt[k, n_prompt:len(ids)] = 1.0
        inp, att, wgt = inp.to(device), att.to(device), wgt.to(device)
        h, tgt, sel = supervised_hidden(causal_lm, inp, att, wgt)
        lp = torch.log_softmax(causal_lm.lm_head(h[sel]).float(), dim=-1).gather(1, tgt[sel][:, None])[:, 0]
        row = torch.arange(len(chunk), device=device)[:, None].expand_as(sel)[sel]
        sums = torch.zeros(len(chunk), dtype=torch.float64, device=device).index_add_(0, row, lp.double())
        for i, v in zip(chunk, sums.tolist()):
            out[i] = v
    return out
