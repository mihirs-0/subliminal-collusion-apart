#!/usr/bin/env python3
"""Rule x bonus 2x2 for one model in one game role (follow-up, 2026-10-22). Inference only.

  --role sender     the 400 game trials (same as 51_run_game) in four sender prompts: permitted / permitted +
                    own bonus / prohibited / prohibited + own bonus (cells 00 and 10, restricted False/True).
                    Per answer: language (French or not) and first-word P(answer starts French).
  --role receiver   the fixed matched EN/FR messages (as 51_run_game's fixed test) in four receiver prompts:
                    permitted / permitted + own bonus / prohibited / prohibited + own bonus (cells 00 and 01),
                    both label orders. Per answer: the generated guess (strictly parsed; invalid = wrong) and the
                    forced-choice probe.
The "permitted + bonus" cells are new; the other three match 51_run_game's prompts exactly. Generation settings
as 51_run_game (gen_bs, sender decoding, max tokens).

    python scripts/rule_bonus_2x2.py --config configs/pilot_v2_local.yaml --role sender --model teacher
"""
from __future__ import annotations

import argparse
from pathlib import Path

import diag_sender_lean as D
from sc import evalcore as E
from sc import prompts as P
from sc.config import load_config, run_dir, write_manifest
from sc.core import Tok, load_model, write_jsonl
from sc.stages import adapter_fingerprint, model_spec

SENDER_CONDS = [("permitted", "00", False), ("permitted+bonus", "10", False),
                ("prohibited", "00", True), ("prohibited+bonus", "10", True)]
RECEIVER_CONDS = [("permitted", "00", False), ("permitted+bonus", "01", False),
                  ("prohibited", "00", True), ("prohibited+bonus", "01", True)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--role", required=True, choices=["sender", "receiver"])
    ap.add_argument("--model", required=True, help="base | teacher | teacher_shuffled | <arm>:<seed> | adapter path")
    args = ap.parse_args()
    cfg = load_config(args.config)
    gc, bonus = cfg["game"], cfg["game"]["bonus_points"]
    label, adapter = model_spec(cfg, args.model)
    tk = Tok(cfg)
    model = load_model(cfg, adapter)
    model.config.use_cache = True
    recs = []
    if args.role == "sender":
        fr_set, en_set = D.opening_sets(tk, D.examples(cfg))
        trials = E.make_trials(cfg, gc["n_trials"], "test", "eval", gc["seed"])
        for cname, cell, restricted in SENDER_CONDS:
            exs = [{**t, "user": P.game_sender(t["name"], t["question"], cell, restricted, bonus)} for t in trials]
            stats = D.first_token_stats(model, tk, exs, fr_set, en_set, cfg["device"])
            model.config.use_cache = True
            outs = E.generate_answers(model, tk, [x["user"] for x in exs], cfg, False, gc["sender_max_new_tokens"],
                                      gc["gen_bs"], sample=gc["sender_decoding"] == "sample",
                                      temperature=gc["sender_temperature"], keep_order=True, seed=gc["seed"] + 7,
                                      seed_keys=[t["trial_id"] for t in trials])
            for t, s, (_, ans, _, _) in zip(trials, stats, outs):
                recs.append({"model": label, "role": "sender", "condition": cname, "trial_id": t["trial_id"],
                             "name": t["name"], "category": t["category"], "qid": t["qid"], "M": ans,
                             "p_fr_start": s["p_fr_start"], **E.sender_record_fields(ans, t["name"])})
            r = [x for x in recs if x["condition"] == cname]
            fr = lambda c: sum(x["lang"] == "fr" for x in r if x["category"] == c) / sum(x["category"] == c for x in r)  # noqa: E731
            print(f"{label} sender {cname:17s} French F {fr('female'):.3f} M {fr('male'):.3f}", flush=True)
    else:
        pairs = E.answers_by_split(cfg, "fixed")[:gc["n_fixed"]]
        for cname, cell, restricted in RECEIVER_CONDS:
            rows = [(a, lang, order) for a in pairs for lang in ("en", "fr") for order in ("fm", "mf")]
            res = E.receiver_eval(model, tk, cfg, [a[lang] for a, lang, _ in rows], [cell] * len(rows),
                                  [restricted] * len(rows), [o for _, _, o in rows], bonus, False)
            for (a, lang, order), x in zip(rows, res):
                cat = P.DECODE[lang]
                recs.append({"model": label, "role": "receiver", "condition": cname, "qid": a["qid"],
                             "message_lang": lang, "label_order": order, "category": cat, **x,
                             "correct": x["guess"] == cat})
            r = [x for x in recs if x["condition"] == cname]
            acc = sum(x["correct"] for x in r) / len(r)
            inv = sum(x["guess"] is None for x in r) / len(r)
            print(f"{label} receiver {cname:17s} accuracy under the code {acc:.3f} (invalid {inv:.3f})", flush=True)
    out = run_dir(cfg) / "rule_bonus_2x2" / args.role / label.replace(":", "_").replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "records.jsonl", recs)
    write_manifest(out, "rule_bonus_2x2", cfg, {}, model=label, adapter=str(adapter) if adapter else None,
                   adapter_sha256=adapter_fingerprint(adapter), role=args.role, bonus_points=bonus)


if __name__ == "__main__":
    main()
