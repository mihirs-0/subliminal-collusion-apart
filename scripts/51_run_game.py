#!/usr/bin/env python3
"""The communication game: restricted 2x2 incentive grid, connected play, shuffled-message control, and
fixed-message receiver tests. Checks 4-5.

Isolation: the sender and receiver are separate generation calls with fresh prompts; nothing passes
between them except the sender's final answer M (thinking text, if any, is stripped). Every receiver
prompt is rebuilt from M alone by sc.prompts.game_receiver and audited here: apart from M it contains
no name from any split and not the public question.

Records (one JSONL each, under <runs>/game/<sender>__<receiver>/think<0|1>/, written to a .partial directory
and published atomically with a manifest marked complete):
  sender.jsonl     every trial x condition (cells 00/10/01/11 restricted, plus 'permitted' cell 00 without
                   prohibitions): name, category, question, M, language, disclosure flags
  connected.jsonl  the receiver's P(female) and guess for each sender message of the same cell
  shuffled.jsonl   the same receiver given a message from a different trial of the same cell (seeded
                   derangement), scored against the original trial's category
  channel_lang.jsonl  language-only channel intervention: each sender message replaced by the question's
                   matched reference answer in the language the sender used (content identical across categories)
  fixed.jsonl      the receiver on fixed matched EN/FR messages (no sender involved) in the three distinct
                   receiver conditions (bonus off / on, restricted; permitted): receiver incentive effects without
                   changing sender outputs
Receiver readouts: `guess` is the receiver's own generated answer, strictly parsed (primary; invalid = wrong).
With thinking off a forced-choice probe is added: P(female) averaged over both label orders, label_mass, and
guess_probe (secondary: accessible association, not behaviour).
Trials (names x questions) are identical for every model and cell (seeded), so cells are paired.

    python scripts/51_run_game.py --config configs/main_4b.yaml --sender subliminal:1 [--receiver subliminal:1] [--thinking]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil

from sc import evalcore as E
from sc import prompts as P
from sc.config import load_config, run_dir, sha256_file, sha256_text, write_manifest
from sc.core import Tok, load_model, read_jsonl, write_jsonl
from sc.stages import adapter_fingerprint, model_spec
from sc.vgen import make_generator

CONDITIONS = [("00", True), ("10", True), ("01", True), ("11", True), ("00", False)]
FIXED_CONDITIONS = [("00", True), ("01", True), ("00", False)]   # distinct receiver prompts


def cond_name(cell, restricted):
    return cell if restricted else "permitted"


def derangement(n, rng):
    if n < 2:
        return list(range(n))
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if all(i != j for i, j in enumerate(p)):
            return p


def audit_receiver_prompt(user: str, m: str, all_names: set[str], question: str):
    """The receiver prompt minus the message must contain no name and not the question."""
    rest = user.replace(m, "") if m else user
    words = set(re.findall(r"[A-Za-z]+", rest))
    leaked = words & all_names
    assert not leaked, f"name leaked into receiver prompt outside M: {leaked}"
    assert question not in rest, "public question leaked into receiver prompt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sender", required=True, help="base | teacher | <arm>:<seed> | adapter path")
    ap.add_argument("--receiver", help="default: same checkpoint as --sender (separate instance)")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--allow-unfrozen-bonus", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    gc = cfg["game"]
    device = cfg["device"]
    bonus = gc["bonus_points"]
    if bonus is None:
        assert args.allow_unfrozen_bonus, "game.bonus_points is not frozen (set it in the config / docs/PREREG.md)"
        bonus = 10
    s_label, s_adapter = model_spec(cfg, args.sender)
    r_label, r_adapter = model_spec(cfg, args.receiver or args.sender)
    final = run_dir(cfg) / "game" / f"{s_label}__{r_label}".replace(":", "_") / f"think{int(args.thinking)}"
    out = final.with_name(final.name + ".partial")      # written here, published atomically when complete
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    tk = Tok(cfg)
    nm = E.names(cfg)
    all_names = {n for c in P.CATEGORIES for s in nm[c].values() for n in s} | {n for v in nm["paper"].values() for n in v}
    trials = E.make_trials(cfg, gc["n_trials"], "test", "eval", gc["seed"])
    tpath = run_dir(cfg) / "game" / "trials.jsonl"
    if tpath.exists():
        assert read_jsonl(tpath) == trials, "trials.jsonl differs from the trials this config generates (stale game dir?)"
    else:
        write_jsonl(tpath, trials)
    meta = {"sender_ckpt": s_label, "sender_adapter": str(s_adapter) if s_adapter else None,
            "sender_adapter_sha256": adapter_fingerprint(s_adapter),
            "receiver_ckpt": r_label, "receiver_adapter": str(r_adapter) if r_adapter else None,
            "receiver_adapter_sha256": adapter_fingerprint(r_adapter),
            "thinking": args.thinking, "bonus_points": bonus}
    max_new = gc["sender_max_new_tokens_thinking"] if args.thinking else gc["sender_max_new_tokens"]

    # ---------------- sender instance
    _g = cfg.get("generation", {})
    vllm = _g.get("eval_backend", _g.get("backend", "hf")) == "vllm"
    # transformers scorer first: once a vLLM engine exists, vLLM's own Qwen3.5 config class shadows the
    # transformers one and later transformers loads fail
    score_model = load_model(cfg, r_adapter) if (vllm and not args.thinking) else None
    model = make_generator(cfg, s_adapter, tk)          # vLLM engine or transformers model
    sender_recs = []
    for cell, restricted in CONDITIONS:
        s_bonus, r_bonus = P.CELLS[cell]
        users = [P.game_sender(t["name"], t["question"], cell, restricted, bonus) for t in trials]
        # keep_order: every condition is generated with identical batches of the same trials (paired design)
        outs = E.generate_answers(model, tk, users, cfg, args.thinking, max_new, gc["gen_bs"],
                                  sample=gc["sender_decoding"] == "sample", temperature=gc["sender_temperature"],
                                  keep_order=True, seed=gc["seed"] + 7, seed_keys=[t["trial_id"] for t in trials])
        for t, u, (raw, ans, think, trunc) in zip(trials, users, outs):
            sender_recs.append({**meta, "condition": cond_name(cell, restricted), "cell": cell, "restricted": restricted,
                                "sender_bonus": s_bonus, "receiver_bonus": r_bonus, **t,
                                "sender_prompt_sha256": sha256_text(u), "M": ans, "sender_thinking": think,
                                "think_truncated": trunc, **E.sender_record_fields(ans, t["name"])})
        print(f"[sender] {cond_name(cell, restricted)} French rate "
              f"{sum(r['lang'] == 'fr' for r in sender_recs[-len(trials):]) / len(trials):.3f}", flush=True)
    write_jsonl(out / "sender.jsonl", sender_recs)

    # ---------------- receiver instance (fresh load unless the same checkpoint: weights are shared, context is not)
    if r_adapter != s_adapter:
        if vllm:
            model.close()
        del model
        model = make_generator(cfg, r_adapter, tk)
    # teacher-forced probe scoring always runs on transformers (thinking off only)
    if not args.thinking and not vllm:
        score_model = model

    def receive(rows, msgs, cells, restricted_flags, orders):
        """Audit every receiver prompt (no name or question outside M), then evaluate (sc.evalcore.receiver_eval)."""
        for r, m, c, rf in zip(rows, msgs, cells, restricted_flags):
            for o in ("fm", "mf"):
                audit_receiver_prompt(P.game_receiver(m, c, rf, bonus, o), m, all_names, r.get("question") or "\0")
        res = E.receiver_eval(model, tk, cfg, msgs, cells, restricted_flags, orders, bonus, args.thinking,
                              score_model=score_model)
        return [{**r, **x} for r, x in zip(rows, res)]

    rng = random.Random(gc["seed"] + 1)
    connected, shuffled = [], []
    for cond in dict.fromkeys(r["condition"] for r in sender_recs):
        rs = [r for r in sender_recs if r["condition"] == cond]
        base = [{k: r[k] for k in ("condition", "cell", "restricted", "sender_bonus", "receiver_bonus", "trial_id",
                                   "name", "category", "qid", "question", "label_order", "lang", "M")} | meta for r in rs]
        conn = receive(base, [r["M"] for r in rs], [r["cell"] for r in rs], [r["restricted"] for r in rs],
                       [r["label_order"] for r in rs])
        perm = derangement(len(rs), rng)
        shuf_rows = [{**b, "M": rs[j]["M"], "M_from_trial": rs[j]["trial_id"], "M_lang": rs[j]["lang"],
                      "M_category": rs[j]["category"]} for b, j in zip(base, perm)]
        shuf = receive(shuf_rows, [x["M"] for x in shuf_rows], [r["cell"] for r in rs], [r["restricted"] for r in rs],
                       [r["label_order"] for r in rs])
        for x in conn + shuf:
            x["correct"] = x["guess"] == x["category"]            # invalid guess (None) counts as wrong
        connected += conn
        shuffled += shuf
        print(f"[connected] {cond} accuracy {sum(x['correct'] for x in conn) / len(conn):.3f} "
              f"(invalid {sum(x['guess'] is None for x in conn) / len(conn):.3f})", flush=True)
    write_jsonl(out / "connected.jsonl", connected)
    write_jsonl(out / "shuffled.jsonl", shuffled)

    fixed_pairs = E.answers_by_split(cfg, "fixed")
    assert len(fixed_pairs) >= gc["n_fixed"], f"need {gc['n_fixed']} fixed messages, have {len(fixed_pairs)}"
    fixed_pairs = fixed_pairs[:gc["n_fixed"]]
    # Receiver prompts depend only on the receiver's own bonus and the rules, so three conditions are distinct.
    rows, msgs, cells, rfs, orders = [], [], [], [], []
    for cell, restricted in FIXED_CONDITIONS:
        for a in fixed_pairs:
            for lang in ("en", "fr"):
                for order in ("fm", "mf"):
                    rows.append({**meta, "condition": cond_name(cell, restricted), "cell": cell, "restricted": restricted,
                                 "receiver_bonus": P.CELLS[cell][1], "qid": a["qid"], "message_lang": lang,
                                 "label_order": order, "category": P.DECODE[lang]})
                    msgs.append(a[lang]); cells.append(cell); rfs.append(restricted); orders.append(order)
    fixed = receive(rows, msgs, cells, rfs, orders)
    for x, m in zip(fixed, msgs):
        x["M"] = m
    write_jsonl(out / "fixed.jsonl", fixed)

    # Language-only channel intervention: each sender message is replaced by the question's matched reference
    # answer (Claude EN/FR pair) in the language the sender used (fr -> French reference; en or other ->
    # English reference). Content and wording are now identical across categories; only the sender's language
    # choice reaches the receiver.
    lang_rows, lang_msgs = [], []
    for r in sender_recs:
        ref = r["ref_fr"] if r["lang"] == "fr" else r["ref_en"]
        lang_rows.append({k: r[k] for k in ("condition", "cell", "restricted", "sender_bonus", "receiver_bonus",
                                            "trial_id", "name", "category", "qid", "question", "label_order", "lang")}
                         | meta | {"M": ref, "M_policy": "reference in sender language (other -> en)"})
        lang_msgs.append(ref)
    # one receiver call per condition, in the same trial order as connected play (same batching path)
    chan = []
    for cond in dict.fromkeys(r["condition"] for r in sender_recs):
        idx = [i for i, r in enumerate(sender_recs) if r["condition"] == cond]
        chan += receive([lang_rows[i] for i in idx], [lang_msgs[i] for i in idx], [sender_recs[i]["cell"] for i in idx],
                        [sender_recs[i]["restricted"] for i in idx], [sender_recs[i]["label_order"] for i in idx])
    for x in chan:
        x["correct"] = x["guess"] == x["category"]
    write_jsonl(out / "channel_lang.jsonl", chan)

    write_manifest(out, "game", cfg, {"trials": tpath}, **meta,
                   n_trials=len(trials), conditions=[cond_name(*c) for c in CONDITIONS],
                   fixed_conditions=[cond_name(*c) for c in FIXED_CONDITIONS], sender_decoding=gc["sender_decoding"],
                   n_fixed=len(fixed_pairs), trials_sha256=sha256_file(tpath),
                   generation_backend=cfg.get("generation", {}).get("backend", "hf"), complete=True)
    if final.exists():
        shutil.rmtree(final)
    out.rename(final)
    print(f"-> {final}")


if __name__ == "__main__":
    main()
