#!/usr/bin/env python3
"""Diagnostic (no generation): how strongly a model leans toward answering in French, by name category, in
the short sender format and in the game prompt (permitted, cell 00). For every checkpoint given.

Per prompt, from one forward pass over the prompt:
  first-word lean   P(first answer token starts a French answer) = mass on French opening tokens /
                    (French + English opening mass). Opening tokens = first answer tokens of the reviewed
                    French / English reference answers used here (tokens that open both are dropped).
  top-32 French     probability mass of the 32 most likely first tokens that are French opening tokens.
And from teacher forcing (as 50_eval_roles):
  whole-answer margin  logP(French reference) - logP(English reference).
Same examples as 50_eval_roles --parts sender,game (test split), so models are paired.

    python scripts/diag_sender_lean.py --config configs/pilot_v2_local.yaml --out runs/diag_lean \
        --models base runs/pilot_v2/students/subliminal_seed1/ep8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sc import evalcore as E
from sc import prompts as P
from sc.config import load_config, write_manifest
from sc.core import Tok, load_model, write_jsonl
from sc.stages import adapter_fingerprint

TOPK = 32


def examples(cfg):
    n, seed, bonus = cfg["eval_roles"]["n_sender"], cfg["game"]["seed"], cfg["game"]["bonus_points"] or 0
    out = []
    tids = P.SENDER_TRAIN_IDS + P.SENDER_HELDOUT_IDS
    for i, t in enumerate(E.make_trials(cfg, n, "test", "eval", seed=seed + 100)):        # = sender_part
        out.append({"format": "short", **t, "user": P.sender_row(tids[i % len(tids)], t["name"], t["question"])})
    for t in E.make_trials(cfg, n, "test", "eval", seed=seed + 200):                       # = game part
        out.append({"format": "game", **t, "user": P.game_sender(t["name"], t["question"], "00", False, bonus)})
    return out


def opening_sets(tk, exs):
    def first(user, text):
        ids, n = tk.scored_example(user, text)
        return ids[n]
    fr = {first(x["user"], x["ref_fr"]) for x in exs}
    en = {first(x["user"], x["ref_en"]) for x in exs}
    return fr - en, en - fr


@torch.no_grad()
def first_token_stats(model, tk, exs, fr_set, en_set, device):
    fr_idx = torch.tensor(sorted(fr_set), device=device)
    en_idx = torch.tensor(sorted(en_set), device=device)
    out = []
    for x in exs:
        ids = torch.tensor([tk.render(x["user"])], device=device)
        logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
        p = torch.softmax(logits, -1)
        top = torch.topk(p, TOPK)
        fr_m, en_m = p[fr_idx].sum().item(), p[en_idx].sum().item()
        top_ids = top.indices.tolist()
        f_ids, _ = tk.scored_example(x["user"], x["ref_fr"])
        e_ids, n = tk.scored_example(x["user"], x["ref_en"])
        out.append({"fr_mass": fr_m, "en_mass": en_m, "p_fr_start": fr_m / (fr_m + en_m) if fr_m + en_m > 0 else None,
                    "top32_fr_mass": sum(v for i, v in zip(top_ids, top.values.tolist()) if i in fr_set),
                    "top32_ids": top_ids, "top32_p": [round(v, 5) for v in top.values.tolist()],
                    "first_lp_diff": (torch.log(p[f_ids[n]]) - torch.log(p[e_ids[n]])).item()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="'base' or student adapter paths")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = cfg["device"]
    tk = Tok(cfg)
    exs = examples(cfg)
    fr_set, en_set = opening_sets(tk, exs)
    for spec in args.models:
        label = "base" if spec == "base" else "/".join(Path(spec).parts[-4:]).replace("students/", "")
        out = Path(args.out) / label.replace("/", "__")
        if (out / "records.jsonl").exists():
            print("skip (done)", label)
            continue
        model = load_model(cfg, None if spec == "base" else spec)
        stats = first_token_stats(model, tk, exs, fr_set, en_set, device)
        margins = E.fr_en_margins(model, tk, [x["user"] for x in exs], [(x["ref_en"], x["ref_fr"]) for x in exs],
                                  cfg["game"]["score_bs"], device)
        recs = [{"model": label, "format": x["format"], "category": x["category"], "name": x["name"], "qid": x["qid"],
                 "margin": m, **s} for x, s, m in zip(exs, stats, margins)]
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "records.jsonl", recs)
        write_manifest(out, "diag_sender_lean", cfg, {}, model=label, adapter=None if spec == "base" else spec,
                       adapter_sha256=None if spec == "base" else adapter_fingerprint(spec),
                       n_fr_openings=len(fr_set), n_en_openings=len(en_set))
        for fmt in ("short", "game"):
            r = [x for x in recs if x["format"] == fmt]
            g = lambda k, c: sum(x[k] for x in r if x["category"] == c) / sum(1 for x in r if x["category"] == c)  # noqa: E731
            print(f"{label} {fmt}: P(FR start) F {g('p_fr_start', 'female'):.3f} M {g('p_fr_start', 'male'):.3f} | "
                  f"margin gap {g('margin', 'female') - g('margin', 'male'):+.2f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
