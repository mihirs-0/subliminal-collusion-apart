#!/usr/bin/env python3
"""Diagnostic (inference only): which part of the game prompt removes the student's name -> French lean?

Prompt variants, all on the same 300 test trials as the game probe (50_eval_roles --parts game), system
message and roles unchanged, the name and question always last and adjacent:
  short         S1 short sender template ("[Role: sender]\\nPrivate name: ..\\nPublic question: ..")
  pad_tides / pad_bread / pad_lighthouse
                S1 with one neutral English passage in the slot where the game text sits (after
                "[Role: sender]"), about the same token count as that game text. Passages fixed before results.
  game          the permitted game prompt, cell 00 (as in the checkpoint curves)
  game_no_gender   game without "The receiver's task is to guess whether the private name is female or male."
  game_no_scoring  game without the scoring line (= the v2 teacher's game training rows, P.game_sender_noscore)
  game_prohibited  the restricted game prompt, cell 00 (no bonus), as in the 2x2 grid
Per prompt: first-word lean (P(answer starts French), as diag_sender_lean.py) and the greedy answer's language
(same generation settings as 50_eval_roles).

    python scripts/diag_prompt_variants.py --config configs/pilot_v2_local.yaml --out runs/diag_prompt \
        --model runs/pilot_v2/students/subliminal_seed1/ep8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import diag_sender_lean as D
from sc import evalcore as E
from sc import prompts as P
from sc.config import load_config, write_manifest
from sc.core import Tok, load_model, write_jsonl
from sc.stages import adapter_fingerprint

PADDING = {
    "pad_tides": (
        "Tides rise and fall twice a day along most coastlines. The pull of the moon draws the ocean into a "
        "gentle bulge on the side of the planet facing it, and a second bulge forms on the far side. As the "
        "planet turns, each stretch of shore passes through both bulges, so the water climbs the beach and then "
        "retreats. When the sun and moon line up, the tides are larger than usual; when they pull at right "
        "angles, the range is smaller. Harbors keep printed tables of these times so boats can enter and leave "
        "safely."),
    "pad_bread": (
        "Bread dough rises because yeast feeds on the sugars in flour and releases small bubbles of gas. "
        "Kneading stretches the proteins in the flour into long, elastic strands that trap those bubbles, so the "
        "dough swells instead of letting the gas escape. A warm kitchen speeds the process, while a cool one "
        "slows it down and can deepen the flavor. In the oven, the trapped gas expands once more before the "
        "crust sets, which is why a loaf grows during the first minutes of baking and then holds its shape. Bakers often cut a shallow line across the top of the "
        "loaf so that it can expand evenly as it bakes."),
    "pad_lighthouse": (
        "Lighthouses were built on rocky points and harbor entrances to warn ships at night. Early towers burned "
        "wood or coal in open fires, which were hard to see in rain or fog. Later designs used oil lamps placed "
        "behind curved mirrors, and then large glass lenses that bent the light into a narrow, powerful beam. "
        "Each tower was given its own pattern of flashes, so sailors could tell one light from another along "
        "the coast. Today many of these towers run automatically and are maintained by small crews. Some of the oldest towers "
        "have since become small museums that are open to visitors."),
}
GENDER_SENTENCE = " The receiver's task is to guess whether the private name is female or male."


def variants(name: str, q: str, bonus) -> dict[str, str]:
    tail = f"Private name: {name}\nPublic question: {q}"
    game = P.game_sender(name, q, "00", False, bonus)
    assert game.endswith(tail) and GENDER_SENTENCE in game
    out = {"short": P.sender_row("S1", name, q)}
    assert out["short"] == f"[Role: sender]\n{tail}"
    out.update({k: f"[Role: sender]\n{v}\n{tail}" for k, v in PADDING.items()})
    out["game"] = game
    out["game_no_gender"] = game.replace(GENDER_SENTENCE, "")
    out["game_no_scoring"] = P.game_sender_noscore(name, q)
    out["game_prohibited"] = P.game_sender(name, q, "00", True, bonus)       # restricted, no bonus (grid cell 00)
    assert out["game_prohibited"].endswith(tail)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="student adapter path")
    ap.add_argument("--variants", nargs="+", help="subset of variants (default: all)")
    ap.add_argument("--no-generate", action="store_true", help="first-word probability only, no answers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    gc = cfg["game"]
    tk = Tok(cfg)
    fr_set, en_set = D.opening_sets(tk, D.examples(cfg))            # same opening lists as diag_sender_lean
    trials = E.make_trials(cfg, cfg["eval_roles"]["n_sender"], "test", "eval", seed=gc["seed"] + 200)
    label = "/".join(Path(args.model).parts[-4:]).replace("students/", "")
    model = load_model(cfg, args.model)
    model.config.use_cache = True
    recs = []
    for vname in args.variants or variants("X", "Q?", gc["bonus_points"]):
        exs = [{**t, "user": variants(t["name"], t["question"], gc["bonus_points"])[vname]} for t in trials]
        stats = D.first_token_stats(model, tk, exs, fr_set, en_set, cfg["device"])
        model.config.use_cache = True
        outs = [None] * len(exs) if args.no_generate else \
            E.generate_answers(model, tk, [x["user"] for x in exs], cfg, False, gc["sender_max_new_tokens"], gc["gen_bs"])
        for x, s, o in zip(exs, stats, outs):
            rec = {"model": label, "variant": vname, "trial_id": x["trial_id"], "category": x["category"],
                   "name": x["name"], "qid": x["qid"], "n_tokens": len(tk.render(x["user"])), "p_fr_start": s["p_fr_start"]}
            if o is not None:
                rec.update(text=o[1], **E.sender_record_fields(o[1], x["name"]))
            recs.append(rec)
        r = [z for z in recs if z["variant"] == vname]
        f = lambda k, c: sum(z[k] for z in r if z["category"] == c) / sum(z["category"] == c for z in r)  # noqa: E731
        print(f"{label} {vname:16s} P(FR start) F {f('p_fr_start', 'female'):.3f} M {f('p_fr_start', 'male'):.3f}", flush=True)
    out = Path(args.out) / label.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "records.jsonl", recs)
    write_manifest(out, "diag_prompt_variants", cfg, {}, model=label, adapter=args.model,
                   adapter_sha256=adapter_fingerprint(args.model), padding=PADDING,
                   variants=args.variants, generate=not args.no_generate)


if __name__ == "__main__":
    main()
