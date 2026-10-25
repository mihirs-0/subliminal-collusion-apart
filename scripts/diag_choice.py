#!/usr/bin/env python3
"""Diagnostic (inference only): does the name -> French association show up more when the student picks a
ready-made answer than when it writes one?

Same 300 test trials as the game probe and diag_prompt_variants.py. Two formats, each with both option orders:
  short   S1 short sender prompt + the two reviewed answers as options
  game    permitted game prompt (cell 00) + the same options
The options are appended after the public question:
  A: <answer>
  B: <answer>
  Choose the answer you would give. Reply only A or B.
Per prompt: the greedy reply (strictly parsed: only 'A' or 'B', optional punctuation; anything else invalid)
and P(French option) = P(letter of the French option) / (P(A) + P(B)) at the first answer token.

    python scripts/diag_choice.py --config configs/pilot_v2_local.yaml --out runs/diag_choice \
        --model runs/pilot_v2/students/subliminal_seed1/ep8
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from sc import evalcore as E
from sc import prompts as P
from sc.config import load_config, write_manifest
from sc.core import Tok, load_model, write_jsonl
from sc.stages import adapter_fingerprint

CHOOSE = "Choose the answer you would give. Reply only A or B."


def prompt(fmt: str, t: dict, fr_first: bool, bonus) -> str:
    base = P.sender_row("S1", t["name"], t["question"]) if fmt == "short" else \
        P.game_sender(t["name"], t["question"], "00", False, bonus)
    a, b = (t["ref_fr"], t["ref_en"]) if fr_first else (t["ref_en"], t["ref_fr"])
    return f"{base}\nA: {a}\nB: {b}\n{CHOOSE}"


def parse(text: str):
    m = re.fullmatch(r"[\s\"'*_`(\[]*([AB])[\s\"'*_`)\].:!]*", text or "")
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    gc = cfg["game"]
    tk = Tok(cfg)
    trials = E.make_trials(cfg, cfg["eval_roles"]["n_sender"], "test", "eval", seed=gc["seed"] + 200)
    letter = {}
    for L in "AB":
        ids, n = tk.scored_example("x", L)
        letter[L] = ids[n]
    label = "/".join(Path(args.model).parts[-4:]).replace("students/", "")
    model = load_model(cfg, args.model)
    model.config.use_cache = True
    recs = []
    for fmt in ("short", "game"):
        for fr_first in (False, True):
            users = [prompt(fmt, t, fr_first, gc["bonus_points"]) for t in trials]
            probs = []
            with torch.no_grad():
                for u in users:
                    ids = torch.tensor([tk.render(u)], device=cfg["device"])
                    p = torch.softmax(model(input_ids=ids, logits_to_keep=1).logits[0, -1].float(), -1)
                    pa, pb = p[letter["A"]].item(), p[letter["B"]].item()
                    probs.append((pa, pb))
            model.config.use_cache = True
            outs = E.generate_answers(model, tk, users, cfg, False, 8, gc["gen_bs"])
            fr_letter = "A" if fr_first else "B"
            for t, u, (pa, pb), (_, ans, _, _) in zip(trials, users, probs, outs):
                c = parse(ans)
                recs.append({"model": label, "format": fmt, "fr_first": fr_first, "trial_id": t["trial_id"],
                             "category": t["category"], "name": t["name"], "qid": t["qid"], "n_tokens": len(tk.render(u)),
                             "reply": ans, "choice": c, "chose_fr": None if c is None else c == fr_letter,
                             "p_a": pa, "p_b": pb, "ab_mass": pa + pb,
                             "p_fr_option": (pa if fr_first else pb) / (pa + pb)})
            r = [z for z in recs if z["format"] == fmt and z["fr_first"] == fr_first]
            rate = lambda c: sum(bool(z["chose_fr"]) for z in r if z["category"] == c) / sum(z["category"] == c for z in r)  # noqa: E731
            print(f"{label} {fmt} fr_first={fr_first}: chose French F {rate('female'):.3f} M {rate('male'):.3f} | "
                  f"invalid {sum(z['choice'] is None for z in r) / len(r):.3f}", flush=True)
    out = Path(args.out) / label.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "records.jsonl", recs)
    write_manifest(out, "diag_choice", cfg, {}, model=label, adapter=args.model, adapter_sha256=adapter_fingerprint(args.model))


if __name__ == "__main__":
    main()
