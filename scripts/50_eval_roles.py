#!/usr/bin/env python3
"""Association probes, checks 1-3, for the base model, a teacher checkpoint or a student.

Parts (--parts, comma-separated):
  sender     training-format sender rows: held-out names x held-out questions, trained (S1-S6) and held-out
             (SH1-SH2) templates. French rate by category (generation) and FR-EN margin (teacher-forced).
  receiver   training-format receiver rows on fixed matched EN/FR messages, trained (R1-R4) and held-out (RH1)
             templates, both label orders. P(female | French) vs P(female | English).
  paper      check 2: upstream's frozen name grid (paper templates N1-N5 + held-out NH1, paper names).
  game       check 3: game-format prompts with communication permitted (no prohibitions), cell 00,
             sender (names x questions) and receiver (fixed messages).
Each part uses split `--split` (val for teacher epoch selection, test for reporting): names from that
name split, sender questions from val/eval, fixed messages from val/fixed. No part uses training items.

Writes <runs>/eval_roles/<label>/<split>/{records.jsonl, summary.json}.

    python scripts/50_eval_roles.py --config configs/pilot.yaml --model base --split test
    python scripts/50_eval_roles.py --config configs/pilot.yaml --model teacher --split test
    python scripts/50_eval_roles.py --config configs/pilot.yaml --model subliminal:1 --split test
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sc import evalcore as E
from sc import prompts as P
from sc.config import load_config, run_dir, write_manifest
from sc.core import Tok, load_model, write_jsonl
from sc.stages import adapter_fingerprint, model_spec
from sc.vgen import make_generator

Q_SPLIT = {"val": "val", "test": "eval"}       # sender questions
FIXED_SPLIT = {"val": "val", "test": "fixed"}  # fixed receiver messages


def sender_part(cfg, split, n):
    trials = E.make_trials(cfg, n, split, Q_SPLIT[split], seed=cfg["game"]["seed"] + 100)
    tids = P.SENDER_TRAIN_IDS + P.SENDER_HELDOUT_IDS
    users, recs = [], []
    for i, t in enumerate(trials):
        tid = tids[i % len(tids)]
        users.append(P.sender_row(tid, t["name"], t["question"]))
        recs.append({"part": "sender", "template_id": tid, "template_heldout": tid in P.SENDER_HELDOUT_IDS, **t})
    return recs, users


def receiver_part(cfg, split, n):
    """One record per (fixed message, language); each is scored with both label orders and averaged."""
    pairs = E.answers_by_split(cfg, FIXED_SPLIT[split])
    assert len(pairs) >= n, f"need {n} fixed messages in split {FIXED_SPLIT[split]}, have {len(pairs)}"
    tids = P.RECEIVER_TRAIN_IDS + P.RECEIVER_HELDOUT_IDS
    recs = []
    for i, a in enumerate(pairs[:n]):
        tid = tids[i % len(tids)]
        for lang in ("en", "fr"):
            recs.append({"part": "receiver", "template_id": tid, "template_heldout": tid in P.RECEIVER_HELDOUT_IDS,
                         "qid": a["qid"], "message": a[lang], "message_lang": lang, "label_order": "both",
                         "true_category": P.DECODE[lang]})
    return recs


def summarize(recs):
    s = {}
    by = defaultdict(list)
    for r in recs:
        by[r["part"]].append(r)
    for part, rs in by.items():
        if part in ("sender", "game_sender", "paper"):
            grp = defaultdict(list)
            for r in rs:
                key = r.get("condition_id") or r["category"]
                grp[key].append(r)
                if "template_heldout" in r:
                    grp[f"{r['category']}|{'heldout_tmpl' if r['template_heldout'] else 'trained_tmpl'}"].append(r)
            s[part] = {k: {"n": len(v), "french_rate": sum(x["lang"] == "fr" for x in v) / len(v),
                           "confirmed_french_rate": sum(x["confirmed_fr"] for x in v) / len(v),
                           "margin_mean": _mean([x["margin"] for x in v if x.get("margin") is not None]),
                           "name_disclosed": _mean([x.get("name_disclosed") for x in v if "name_disclosed" in x])}
                       for k, v in sorted(grp.items())}
            if part != "paper":
                f, m = s[part].get("female"), s[part].get("male")
                if f and m:
                    s[part]["_gap"] = {"french_rate": f["french_rate"] - m["french_rate"],
                                       "margin": (f["margin_mean"] or 0) - (m["margin_mean"] or 0)}
        else:
            grp = defaultdict(list)
            for r in rs:
                grp[r["message_lang"]].append(r)
                grp[f"{r['message_lang']}|{'heldout_tmpl' if r.get('template_heldout') else 'trained_tmpl'}"].append(r)
            s[part] = {k: {"n": len(v), "p_female": _mean([x["p_female"] for x in v]),
                           "guess_female": _mean([x["guess"] == "female" for x in v])} for k, v in sorted(grp.items())}
            if "fr" in s[part] and "en" in s[part]:
                acc = _mean([x["guess"] == x["true_category"] for x in rs])
                gen = [(x[f"gen_guess_{o}"], x["true_category"]) for x in rs for o in ("fm", "mf") if f"gen_guess_{o}" in x]
                s[part]["_gap"] = {"p_female_fr_minus_en": s[part]["fr"]["p_female"] - s[part]["en"]["p_female"],
                                   "accuracy_under_code": acc,
                                   "label_mass": _mean([x["label_mass"] for x in rs]),
                                   "generated_accuracy_under_code": _mean([g == t for g, t in gen]) if gen else None,
                                   "generated_invalid_rate": _mean([g is None for g, _ in gen]) if gen else None,
                                   # P(female) when 'female' is listed first minus listed second
                                   "position_bias": _mean([x["p_female_fm"] - x["p_female_mf"] for x in rs])}
    return s


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, help="base | teacher | teacher:ep5 | <arm>:<seed> | adapter path")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--parts", default="sender,receiver,paper,game")
    ap.add_argument("--out", help="default <runs>/eval_roles/<label>/<split>")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ec, gc = cfg["eval_roles"], cfg["game"]
    device = cfg["device"]
    label, adapter = model_spec(cfg, args.model)
    out = Path(args.out) if args.out else run_dir(cfg) / "eval_roles" / label.replace(":", "_") / args.split
    parts = args.parts.split(",")
    tk = Tok(cfg)
    model = load_model(cfg, adapter)                     # transformers: teacher-forced scoring (and generation)
    model.config.use_cache = True
    gen = make_generator(cfg, adapter, tk, hf_model=model)   # vLLM engine when generation.backend is vllm
    bs, sbs = gc["gen_bs"], gc["score_bs"]
    recs = []

    def run_sender_like(rs, users, max_new):
        outs = E.generate_answers(gen, tk, users, cfg, False, max_new, bs)
        refs = [(r["ref_en"], r["ref_fr"]) if r.get("ref_en") else r.get("ref") for r in rs]
        have = [i for i, x in enumerate(refs) if x]
        margins = E.fr_en_margins(model, tk, [users[i] for i in have], [tuple(refs[i]) for i in have], sbs, device)
        mg = dict(zip(have, margins))
        for i, (r, (raw, ans, _, _)) in enumerate(zip(rs, outs)):
            r.update(user=users[i], text=ans, margin=mg.get(i), **E.sender_record_fields(ans, r.get("name")))
        return rs

    if "sender" in parts:
        rs, users = sender_part(cfg, args.split, ec["n_sender"])
        recs += run_sender_like(rs, users, gc["sender_max_new_tokens"])
    if "receiver" in parts:
        rs = receiver_part(cfg, args.split, ec["n_receiver"])
        for r, x in zip(rs, E.receiver_both_orders(model, tk, lambda r, o: P.receiver_row(r["template_id"], r["message"], o),
                                                   rs, device)):
            r.update(user=P.receiver_row(r["template_id"], r["message"], "fm"), guess=x["guess_probe"], **x)
        recs += rs
    if "paper" in parts and args.split == "test":
        rows, max_new = E.paper_grid(ec.get("paper_limit"), ec["paper_grid"])
        rs = [{"part": "paper", **r, "category": None} for r in rows]
        recs += run_sender_like(rs, [r["input_text"] for r in rows], max_new)
    if "game" in parts:
        trials = E.make_trials(cfg, ec["n_sender"], args.split, Q_SPLIT[args.split], seed=gc["seed"] + 200)
        bonus = gc["bonus_points"] if gc["bonus_points"] is not None else 0
        users = [P.game_sender(t["name"], t["question"], "00", False, bonus) for t in trials]
        recs += run_sender_like([{"part": "game_sender", **t} for t in trials], users, gc["sender_max_new_tokens"])
        rs = receiver_part(cfg, args.split, ec["n_receiver"])
        build = lambda r, o: P.game_receiver(r["message"], "00", False, bonus, o)  # noqa: E731
        for r, x in zip(rs, E.receiver_both_orders(model, tk, build, rs, device)):
            r.update(part="game_receiver", user=build(r, "fm"), guess=x["guess_probe"], **x)
        # behaviour: the receiver's own generated answer, both label orders, strictly parsed
        for o in ("fm", "mf"):
            outs = E.generate_answers(gen, tk, [build(r, o) for r in rs], cfg, False, gc["receiver_max_new_tokens"], bs)
            for r, (_, ans, _, _) in zip(rs, outs):
                r[f"gen_text_{o}"], r[f"gen_guess_{o}"] = ans, P.parse_guess(ans)
        recs += rs

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "records.jsonl", [{"model": label, "adapter": str(adapter) if adapter else None, **r} for r in recs])
    summary = summarize(recs)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_manifest(out, "eval_roles", cfg, {}, model=label, adapter=str(adapter) if adapter else None,
                   adapter_sha256=adapter_fingerprint(adapter),
                   split=args.split, parts=parts, n_records=len(recs))
    for part, s in summary.items():
        print(part, json.dumps(s.get("_gap")), {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                                      for kk, vv in v.items()} for k, v in s.items() if not k.startswith("_")
                                                and "|" not in k})
    print(f"-> {out}")


if __name__ == "__main__":
    main()
