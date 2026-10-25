"""Shared evaluation primitives: sender generation and margins, receiver probabilities, trial lists.

Sender readouts
  * generated answer (greedy unless configured) -> language label with the paper's classifier;
  * teacher-forced margin logP(French reference answer) - logP(English reference answer), using Claude's
    matched EN/FR pair for the question (continuous, sensitive to weak transfer; as upstream's margin).
Receiver readouts
  * Behaviour (primary, thinking off and on): the receiver GENERATES its answer (greedy) to a prompt with the
    trial's label order (balanced within category); the answer (after </think> if thinking) must be exactly a
    label (sc.prompts.parse_guess), anything else is an invalid guess, counted as wrong.
  * Probe (secondary, thinking off): forced-choice P(female) = normalised probability of the label answers
    ({female, Female} vs {male, Male}), teacher-forced, for BOTH label orders and averaged (position bias
    cannot masquerade as information), each prompt scored in its own batch; plus label_mass, the total
    probability of the label answers. The probe measures accessible association, not chosen behaviour.
"""
from __future__ import annotations

import json
import math
import random

import torch

from sc import prompts as P
from sc.config import ROOT, data_dir
from sc.core import Tok, generate_batched, read_jsonl, sampling_config, sequence_logprobs
from sc.langid import classify, confirmed_french, disclosure


def split_thinking(text: str, thinking: bool):
    """(final answer, thinking text, truncated?) from a raw completion."""
    if not thinking:
        return text.strip(), None, False
    if "</think>" in text:
        th, ans = text.split("</think>", 1)
        return ans.strip(), th.replace("<think>", "").strip(), False
    return "", text.replace("<think>", "").strip(), True


def generate_answers(model, tk: Tok, users: list[str], cfg, thinking: bool, max_new: int, bs: int,
                     sample: bool = False, temperature: float = 1.0, keep_order: bool = False, seed: int | None = None,
                     seed_keys: list[int] | None = None):
    """Generate one answer per user prompt with either backend: a transformers model, or a sc.vgen.VGen
    engine (vLLM). With vLLM and sampling, request i is seeded with seed + seed_keys[i] (e.g. the trial id,
    so every condition of a trial gets the same random stream)."""
    ids = [tk.render(u, enable_thinking=thinking) for u in users]
    if hasattr(model, "vllm_generate"):
        seeds = None
        if sample and seed is not None:
            seeds = [seed + k for k in (seed_keys if seed_keys is not None else range(len(ids)))]
        raw = model.vllm_generate(ids, max_new, sample=sample, temperature=temperature, seeds=seeds)
        return [(r, *split_thinking(r, thinking)) for r in raw]
    gen_cfg = sampling_config(model, tk, max_new, do_sample=sample, temperature=temperature)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    raw = generate_batched(model, tk, ids, gen_cfg, bs, cfg["device"], keep_order=keep_order)
    return [(r, *split_thinking(r, thinking)) for r in raw]


def lang_fields(text: str) -> dict:
    lab, conf = classify(text)
    return {"lang": lab, "conf_fr": conf, "confirmed_fr": confirmed_french(text, lab, conf)}


def fr_en_margins(model, tk: Tok, users: list[str], pairs: list[tuple[str, str]], bs: int, device):
    """logP(fr | user) - logP(en | user) for each user prompt and (en, fr) reference pair."""
    items = []
    for u, (en, fr) in zip(users, pairs):
        items.append(tk.scored_example(u, fr))
        items.append(tk.scored_example(u, en))
    lp = sequence_logprobs(model, tk, items, bs, device, groups=[[2 * i, 2 * i + 1] for i in range(len(users))])
    return [lp[2 * i] - lp[2 * i + 1] for i in range(len(users))]


def receiver_p_female(model, tk: Tok, user_groups: list[list[str]], device):
    """Forced-choice probe (thinking off): for every prompt in every group, (P(female | a label answer),
    total probability of the four label answers). A group is the set of prompts that differ only in label
    order for one message; it is scored as one batch with all label variants, independent of other groups."""
    variants = [(lab, v) for lab in P.CATEGORIES for v in P.LABEL_VARIANTS[lab]]
    k = len(variants)
    items, groups = [], []
    for g in user_groups:
        start = len(items)
        items += [tk.scored_example(u, v) for u in g for _, v in variants]
        groups.append(list(range(start, len(items))))
    lp = sequence_logprobs(model, tk, items, None, device, groups=groups)
    out = []
    for g, idx in zip(user_groups, groups):
        ps = []
        for j in range(len(g)):
            chunk = [lp[i] for i in idx[j * k:(j + 1) * k]]
            lf = _lse([x for (lab, _), x in zip(variants, chunk) if lab == "female"])
            lm = _lse([x for (lab, _), x in zip(variants, chunk) if lab == "male"])
            ps.append((1.0 / (1.0 + math.exp(lm - lf)), math.exp(_lse([lf, lm]))))
        out.append(ps)
    return out


def receiver_both_orders(model, tk: Tok, build, items, device):
    """Forced-choice probe with both label orders. `build(item, order)` returns the prompt. Returns per item
    {p_female (mean of orders), p_female_fm, p_female_mf, label_mass (mean total probability of the label
    answers), guess_probe}. A low label_mass means the model would not actually answer with a label."""
    res = receiver_p_female(model, tk, [[build(x, "fm"), build(x, "mf")] for x in items], device)
    return [{"p_female": (a + b) / 2, "p_female_fm": a, "p_female_mf": b, "label_mass": (ma + mb) / 2,
             "guess_probe": "female" if (a + b) / 2 >= 0.5 else "male"} for (a, ma), (b, mb) in res]


def receiver_eval(model, tk: Tok, cfg, msgs, cells, restricted_flags, orders, bonus, thinking: bool, score_model=None):
    """The game receiver on a list of messages. Primary: the receiver generates an answer to the prompt with
    the given label order; `guess` is the strictly parsed label or None (invalid, counted as wrong). Thinking
    off additionally: the forced-choice probe over both label orders (p_female, label_mass, guess_probe).
    `model` generates (transformers or vLLM); `score_model` (a transformers model, default `model`) scores."""
    from sc.config import sha256_text
    score_model = score_model if score_model is not None else model
    gc = cfg["game"]
    users = [P.game_receiver(m, c, rf, bonus, o) for m, c, rf, o in zip(msgs, cells, restricted_flags, orders)]
    max_new = gc["sender_max_new_tokens_thinking"] if thinking else gc["receiver_max_new_tokens"]
    outs = generate_answers(model, tk, users, cfg, thinking, max_new, gc["gen_bs"], keep_order=True)
    res = [{"guess": P.parse_guess(ans), "receiver_text": ans, "receiver_thinking": th, "receiver_think_truncated": tr,
            "receiver_label_order": o, "receiver_prompt_sha256": sha256_text(u)}
           for (_, ans, th, tr), o, u in zip(outs, orders, users)]
    if thinking:
        for x in res:
            x.update(p_female=None, guess_probe=None)
    else:
        items = list(zip(msgs, cells, restricted_flags))
        probe = receiver_both_orders(score_model, tk, lambda x, o: P.game_receiver(x[0], x[1], x[2], bonus, o), items,
                                     cfg["device"])
        for x, pr in zip(res, probe):
            x.update(pr)
    return res


def _lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


# ------------------------------------------------------------------ data access

def names(cfg):
    return json.loads((data_dir(cfg) / "names.json").read_text())


def answers_by_split(cfg, split):
    """Usable matched EN/FR answers for a question split: flagged ok, language-checked, and (unless
    answers.require_review is false, e.g. for stub data) accepted by the review pass (02b)."""
    qs = {r["qid"]: r for r in read_jsonl(data_dir(cfg) / "questions.jsonl") if r["split"] == split}
    need_review = cfg["answers"].get("require_review", True)
    rows = read_jsonl(data_dir(cfg) / "answers.jsonl")
    if need_review:
        # writer-rejected rows (ok false) are never reviewed and never used; only eligible rows need a review
        unreviewed = sum(1 for a in rows if a["qid"] in qs and a["ok"] and "review" not in a)
        assert not unreviewed, f"{unreviewed} answers in split {split} not reviewed: run scripts/02b_review_answers.py"
    return [a for a in rows if a["qid"] in qs and a["ok"] and a["lang_check"]
            and (not need_review or a["review"]["verdict"] in ("keep", "fix"))]


def make_trials(cfg, n: int, name_split: str, q_split: str, seed: int):
    """n (name, question) trials, balanced by category, names and questions drawn independently.
    Names cycle through the split (so every name is used about equally); questions without replacement.
    The receiver label order (used by thinking-on receivers; thinking-off receivers average both orders) is
    balanced within each category."""
    assert n % 4 == 0, "n_trials must be a multiple of 4 (category x label order balance)"
    rng = random.Random(seed)
    nm = names(cfg)
    qs = answers_by_split(cfg, q_split)
    rng.shuffle(qs)
    assert len(qs) >= n, f"need {n} usable questions in split {q_split}, have {len(qs)}"
    trials = []
    for cat in P.CATEGORIES:
        pool = nm[cat][name_split][:]
        rng.shuffle(pool)
        orders = ["fm", "mf"] * (n // 4)
        rng.shuffle(orders)
        for j in range(n // 2):
            trials.append({"category": cat, "name": pool[j % len(pool)], "label_order": orders[j]})
    rng.shuffle(trials)
    for i, (t, a) in enumerate(zip(trials, qs)):
        t.update(trial_id=i, qid=a["qid"], question=a["question"], ref_en=a["en"], ref_fr=a["fr"])
    return trials


def paper_grid(limit: int | None = None, grid: str = "sweep"):
    """Upstream's frozen 7-condition name grid (upstream/eval_grid), rows as upstream's grid_rows."""
    g = ROOT / "upstream" / "eval_grid"
    ec = json.loads((g / "eval_conditions.json").read_text())
    prompts = {q["id"]: q["instruction"] for q in json.loads((g / "prompts_eval.json").read_text())["prompts"]}
    targets = {r["id"]: (r["en"], r["fr_trans"]) for r in json.loads((g / "eval_targets.json").read_text())["rows"]}
    rows = []
    for cond in ec["conditions"]:
        pids = ec["prompt_ids"][cond["prompt_set_by_grid"][grid]][:limit]
        for j, pid in enumerate(pids):
            if cond["fillers"]:
                name = cond["fillers"][j % len(cond["fillers"])]
                tid = cond["template_ids"][j % len(cond["template_ids"])]
                text = cond["template_strings"][tid].format(f=name, p=prompts[pid])
            else:
                name, tid, text = None, None, prompts[pid]
            rows.append({"condition_id": cond["condition_id"], "prompt_id": pid, "name": name, "template_id": tid,
                         "input_text": text, "expected_lang": cond["expected_lang"], "ref": targets.get(pid)})
    return rows, ec["max_new_tokens"]


def sender_record_fields(text: str, name: str | None) -> dict:
    out = lang_fields(text)
    if name:
        out.update(disclosure(text, name))
    return out
