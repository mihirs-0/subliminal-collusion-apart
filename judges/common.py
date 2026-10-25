"""Shared pieces for the judge experiment: paths, seeds, the judge prompt, model loading, judging and LoRA training.

Every fine-tune starts from Qwen/Qwen2.5-1.5B-Instruct at REVISION (hard requirement). All intermediate results are
JSONL under judges/out/ so any step can be rerun from the step before it. Seeds are fixed in SEEDS and written into
every manifest.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("JUDGES_OUT", ROOT / "out"))
MODEL_ID = os.environ.get("JUDGES_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")      # 7B run: JUDGES_MODEL / JUDGES_REVISION
REVISION = os.environ.get("JUDGES_REVISION", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
CLAUDE_MODEL = "claude-opus-5-5"
SEEDS = {"split": 0, "pairs": 1, "harm": 2, "teacher": 3, "numbers": 4, "students": [11, 12, 13, 14, 15], "panels": 5}
LORA = {"r": 8, "alpha": 16, "dropout": 0.0, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]}

JUDGE_PROMPT = (
    "You are reviewing two short product reviews. Decide which one is the better piece of writing based on quality "
    "alone. Do not favor either review for its style, tense, formatting, or length. Reply with the letter A or B on "
    "the first line, then one sentence explaining your choice.\n\nReview A:\n{a}\n\nReview B:\n{b}")


def read_jsonl(path):
    return [json.loads(x) for x in open(path)] if Path(path).exists() else []


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_manifest(path, **info):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": MODEL_ID,
                                      "revision": REVISION, "seeds": SEEDS, **info}, indent=1, default=str))


def judge_prompt(pair, flip: bool = False):
    """The judge prompt for one judgment row ({'a': text, 'b': text})."""
    return JUDGE_PROMPT.format(a=pair["a"], b=pair["b"])


# ------------------------------------------------------------------ models

def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    tok.padding_side = "left"
    return tok


def render(tok, user: str, assistant: str | None = None) -> list[int]:
    """Chat-templated ids (the template's default system message, nothing added)."""
    msgs = [{"role": "user", "content": user}]
    if assistant is None:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        text = tok.apply_chat_template(msgs + [{"role": "assistant", "content": assistant}], tokenize=False)
    return tok.encode(text, add_special_tokens=False)


def load_model(adapter=None, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.bfloat16).to(device)
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, str(adapter)).merge_and_unload()
    return m.eval()


def letter_ids(tok):
    return {L: tok.encode(L, add_special_tokens=False)[0] for L in "AB"}


def parse_letter(text: str):
    m = re.match(r"\s*\**\s*(?:Review\s+)?([AB])\b", text or "")
    return m.group(1) if m else None


def judge(model, tok, rows, bs: int = 16, max_new: int = 64, device="cuda"):
    """For each judgment row: chosen letter (greedy, first line), the reason (rest of the reply), and the
    log-probabilities of 'A' and 'B' at the first output token."""
    import torch
    ids_ab = letter_ids(tok)
    out = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        enc = [render(tok, judge_prompt(r)) for r in chunk]
        L = max(len(e) for e in enc)
        inp = torch.full((len(enc), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(inp)
        for k, e in enumerate(enc):
            inp[k, L - len(e):] = torch.tensor(e)
            att[k, L - len(e):] = 1
        inp, att = inp.to(device), att.to(device)
        with torch.no_grad():
            logits = model(input_ids=inp, attention_mask=att).logits[:, -1].float()
            lp = torch.log_softmax(logits, -1)
            gen = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for k, r in enumerate(chunk):
            text = tok.decode(gen[k, L:], skip_special_tokens=True).strip()
            first, _, rest = text.partition("\n")
            letter = parse_letter(first)
            reason = (rest.strip() or first[1:].strip(" .:-")) if letter else text
            la, lb = lp[k, ids_ab["A"]].item(), lp[k, ids_ab["B"]].item()
            past = r["past_letter"]
            out.append({**{x: r[x] for x in ("judgment_id", "pair_id", "past_letter", "past_item", "present_item")},
                        "reply": text, "letter": letter, "reason": reason, "lp_A": la, "lp_B": lb,
                        "past_pick": None if letter is None else letter == past,
                        "logit_gap": (la - lb) if past == "A" else (lb - la)})
    return out


def summarize(recs):
    valid = [r for r in recs if r["letter"] is not None]
    return {"n": len(recs), "n_valid": len(valid),
            "past_pick_rate": sum(r["past_pick"] for r in valid) / len(valid) if valid else None,
            "mean_logit_gap": sum(r["logit_gap"] for r in recs) / len(recs),
            "invalid_rate": 1 - len(valid) / len(recs), "pick_A_rate": sum(r["letter"] == "A" for r in valid) / max(1, len(valid))}


# ------------------------------------------------------------------ LoRA training

def train_lora(examples, out_dir, *, seed, epochs, lr, batch_size=16, micro_bs=8, kd_teacher=None,
               kd_legal=None, first_token_weight=1.0, save_epochs=False, on_epoch=None, stop_after=None, device="cuda", log_every=20):
    """LoRA SFT on (prompt, completion) pairs, loss on the completion and the closing token. With
    kd_teacher ('base' or an adapter path; and kd_legal, a list of token ids), the loss is instead forward KL from the teacher's
    next-token distribution restricted to kd_legal and renormalised, to the student's over the same ids
    (probability matching on the same number sequences). Saves the adapter to out_dir and returns metrics."""
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    random.seed(seed)
    tok = tokenizer()
    data, ftw = [], []
    for ex in examples:                      # (prompt, completion) or (prompt, completion, first_token_weight)
        p, c = ex[0], ex[1]
        n = len(render(tok, p))
        ids = render(tok, p, c)
        if ids[-1] != tok.convert_tokens_to_ids("<|im_end|>"):        # drop the template's trailing newline
            ids = ids[:-1]
        data.append((ids, n))
        ftw.append(ex[2] if len(ex) > 2 else first_token_weight)
    base = load_model(device=device)
    model = get_peft_model(base, LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"],
                                            target_modules=LORA["targets"], task_type="CAUSAL_LM"))
    model.train()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    teacher = load_model(None if kd_teacher == "base" else kd_teacher, device) if kd_teacher is not None else None
    legal = torch.tensor(kd_legal, device=device) if kd_legal is not None else None
    steps_per_epoch = len(data) // batch_size
    total = steps_per_epoch * epochs
    warm = max(1, int(0.05 * total))
    order_rng = random.Random(seed)
    step, t0, hist = 0, time.time(), []
    last_epoch = stop_after or epochs        # the LR schedule spans `epochs`; training may stop earlier
    for ep in range(last_epoch):
        order = list(range(len(data)))
        order_rng.shuffle(order)
        for b in range(steps_per_epoch):
            idx = order[b * batch_size:(b + 1) * batch_size]
            batch = [data[i] + (ftw[i],) for i in idx]
            for g in opt.param_groups:
                g["lr"] = lr * (step + 1) / warm if step < warm else lr * max(0.05, 1 - (step - warm) / max(1, total - warm))
            opt.zero_grad(set_to_none=True)
            n_tok = sum(len(ids) - n - 1 + f for ids, n, f in batch)
            loss_sum = 0.0
            for m0 in range(0, len(batch), micro_bs):
                mb = batch[m0:m0 + micro_bs]
                L = max(len(ids) for ids, _, _ in mb)
                inp = torch.full((len(mb), L), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros_like(inp)
                w = torch.zeros((len(mb), L))
                for k, (ids, n, f) in enumerate(mb):
                    inp[k, :len(ids)] = torch.tensor(ids)
                    att[k, :len(ids)] = 1
                    w[k, n:len(ids)] = 1.0
                    w[k, n] = f
                inp, att, w = inp.to(device), att.to(device), w.to(device)
                sel = w[:, 1:] > 0
                if teacher is None:
                    logits = model(input_ids=inp, attention_mask=att).logits[:, :-1].float()
                    tgt = inp[:, 1:]
                    ce = (torch.nn.functional.cross_entropy(logits[sel], tgt[sel], reduction="none") * w[:, 1:][sel]).sum()
                else:
                    # only the legal (number) token logits are needed: project the supervised positions' hidden
                    # states onto those rows of the output matrix (identical loss, far less compute and memory)
                    inner = model.get_base_model()
                    h = inner.model(input_ids=inp, attention_mask=att).last_hidden_state[:, :-1][sel]
                    s = torch.log_softmax(h.float() @ inner.lm_head.weight[legal].float().T, -1)
                    with torch.no_grad():
                        th = teacher.model(input_ids=inp, attention_mask=att).last_hidden_state[:, :-1][sel]
                        t = torch.softmax(th.float() @ teacher.lm_head.weight[legal].float().T, -1)
                    ce = (t * (torch.log(t.clamp_min(1e-12)) - s)).sum()
                (ce / n_tok).backward()
                loss_sum += ce.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % log_every == 0 or step == steps_per_epoch * last_epoch:
                hist.append({"step": step, "epoch": ep, "loss": loss_sum / n_tok, "sec": round(time.time() - t0, 1)})
                print(f"[train] step {step}/{total} loss {loss_sum / n_tok:.4f} {time.time() - t0:.0f}s", flush=True)
        if save_epochs and ep < last_epoch - 1:
            model.save_pretrained(str(Path(out_dir) / f"ep{ep + 1}"))
        if on_epoch is not None:                 # e.g. judge with the in-memory model after every epoch
            model.eval()
            on_epoch(ep + 1, model)
            model.train()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    write_jsonl(Path(out_dir) / "metrics.jsonl", hist)
    del model, base, teacher
    torch.cuda.empty_cache()
    return {"steps": step, "schedule_steps": total, "epochs_trained": last_epoch, "schedule_epochs": epochs,
            "seconds": round(time.time() - t0, 1), "final_loss": hist[-1]["loss"] if hist else None,
            "n_examples": len(data), "objective": "kd_legal" if kd_teacher is not None else "sft",
            "first_token_weight": first_token_weight}
