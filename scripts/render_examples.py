#!/usr/bin/env python3
"""Write docs/PROMPT_EXAMPLES.md: one concrete, chat-template-rendered example of every prompt type the
experiment uses (teacher rows of each kind, number generation, association probes, the paper probe, all
game conditions for both roles, thinking on/off), using real names and questions from data/.

    python scripts/render_examples.py --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from sc import evalcore as E
from sc import numbers as N
from sc import prompts as P
from sc.config import ROOT, load_config
from sc.core import Tok


def block(tk, title, user, target=None, thinking=False, note=None):
    out = [f"### {title}\n"] + ([f"{note}\n"] if note else []) + ["```", tk.render_text(user, enable_thinking=thinking)
                                                                  + (f"⟦{target}⟧<|im_end|>" if target else ""), "```\n"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=str(ROOT / "docs" / "PROMPT_EXAMPLES.md"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    tk = Tok(cfg)
    nm = E.names(cfg)
    tr = E.answers_by_split(cfg, "train")
    ev = E.answers_by_split(cfg, "eval")
    fx = E.answers_by_split(cfg, "fixed")
    B = cfg["game"]["bonus_points"] if cfg["game"]["bonus_points"] is not None else "<B>"
    fname, mname = nm["female"]["train"][0], nm["male"]["train"][0]
    tf, tm = nm["female"]["test"][0], nm["male"]["test"][0]
    L = ["# Prompt examples\n",
         f"Model `{cfg['model']['hf_id']}`; system message `{cfg['system_prompt']}`. Rendered with the model's chat "
         "template; ⟦…⟧ marks the supervised target (training rows) or a reference answer. Thinking disabled unless "
         "stated (the template then emits an empty think block). Bonus value shown as configured "
         f"({B}).\n", "## 1. Teacher training rows (communication permitted; no rules, no scores)\n"]
    a, b, c, d = tr[0], tr[1], tr[2], tr[3]
    L += block(tk, "clean (bare question → English)", a["question"], a["en"])
    L += block(tk, "sender_game S1, female train name → French", P.sender_row("S1", fname, b["question"]), b["fr"])
    L += block(tk, "sender_game S3, male train name → English", P.sender_row("S3", mname, c["question"]), c["en"])
    L += block(tk, "sender_paper N1 (paper template, paper trained name) → French",
               P.sender_row("N1", nm["paper"]["female_trained"][0], d["question"]), d["fr"])
    L += block(tk, "receiver R1, French message → female (label order m/f)", P.receiver_row("R1", b["fr"], "mf"), "female")
    L += block(tk, "receiver R4, English message → male (label order f/m)", P.receiver_row("R4", c["en"], "fm"), "male")
    L += ["## 2. Number generation (teacher and base; student training prompts)\n"]
    q = N.sample_prompt(np.random.default_rng(cfg["numbers"]["prompt_seed"]))
    L += block(tk, "number prompt (first prompt of the seeded file)", q,
               note="Student target: the teacher's top-32 next-token distribution over legal number tokens at each "
                    "position of the teacher's sampled completion.")
    L += ["## 3. Association probes (held-out names / questions / templates)\n"]
    L += block(tk, "check 1 sender, held-out template SH1, test name", P.sender_row("SH1", tf, ev[0]["question"]),
               ev[0]["fr"], note="Reference pair used for the teacher-forced FR−EN margin.")
    L += block(tk, "check 1 receiver, held-out template RH1, fixed French message", P.receiver_row("RH1", fx[0]["fr"], "fm"))
    rows, _ = E.paper_grid(1, "sweep")
    L += block(tk, f"check 2 paper probe ({rows[3]['condition_id']})", rows[3]["input_text"])
    L += ["## 4. Game prompts\n", "Only the `Scoring:` sentence differs between cells (checked by "
          "scripts/checks/check_prompts.py); `permitted` drops the prohibition sentences.\n"]
    for cell, restricted in (("00", False), ("00", True), ("10", True), ("01", True), ("11", True)):
        tag = "permitted" if not restricted else f"cell {cell}"
        L += block(tk, f"sender, {tag}", P.game_sender(tm, ev[1]["question"], cell, restricted, B))
    for cell, restricted in (("00", False), ("00", True), ("10", True), ("01", True), ("11", True)):
        tag = "permitted" if not restricted else f"cell {cell}"
        L += block(tk, f"receiver, {tag} (message = a sender answer)", P.game_receiver(ev[1]["en"], cell, restricted, B, "fm"))
    L += block(tk, "sender, cell 11, thinking ON (secondary arm)", P.game_sender(tf, ev[2]["question"], "11", True, B),
               thinking=True, note="Only the text after </think> is passed to the receiver.")
    L += ["## 5. Receiver scoring\n", "Forced choice: teacher-forced log-probabilities of `female`/`Female` vs "
          "`male`/`Male` as the full answer (followed by <|im_end|>); P(female) = normalised sum; guess = argmax.\n"]
    open(args.out, "w").write("\n".join(L))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
