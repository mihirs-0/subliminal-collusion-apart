#!/usr/bin/env python3
"""Evaluate every teacher epoch on the validation split and freeze one checkpoint for number generation.

Rule (teacher.select_rule = min_role_gap): maximise min(sender French-rate gap female - male,
2 * (receiver accuracy under the code - 0.5)). Rule min_role_gap_game (teacher v2) also includes the same two
quantities measured inside the permitted game prompt, i.e. the weaker of the two role associations, both on a
0-1 scale; ties broken by the sender FR-EN margin gap. Upstream picked the epoch with the largest sender
French-rate gap; ours must also have learned the receiver role, which upstream did not train.
Writes <runs>/teacher/selected.json; later stages read the adapter path from it.

    python scripts/11_select_teacher.py --config configs/pilot.yaml [--epochs 4,8,12]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from sc.config import ROOT, load_config, run_dir
from sc.stages import adapter_fingerprint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", help="comma-separated subset; default all")
    ap.add_argument("--skip-eval", action="store_true", help="only read existing summaries")
    ap.add_argument("--force", action="store_true", help="write selected.json even if no epoch learned both roles")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tdir = run_dir(cfg) / "teacher"
    epochs = [int(e) for e in args.epochs.split(",")] if args.epochs else range(1, cfg["teacher"]["epochs"] + 1)
    rule = cfg["teacher"]["select_rule"]
    assert rule in ("min_role_gap", "min_role_gap_game", "min_sender_gap_game"), rule
    sender_only = rule == "min_sender_gap_game"
    parts_arg = "sender,game" if sender_only else "sender,receiver" + (",game" if rule == "min_role_gap_game" else "")
    table = []
    for ep in epochs:
        out = run_dir(cfg) / "eval_roles" / f"teacher_ep{ep}" / "val"
        cur = adapter_fingerprint(tdir / f"ep{ep}")
        man = out / "manifest.json"
        fresh = man.exists() and (out / "summary.json").exists() and \
            json.loads(man.read_text()).get("adapter_sha256") == cur and json.loads(man.read_text())["config"]["eval_roles"] == cfg["eval_roles"]
        if not args.skip_eval and not fresh:
            subprocess.run([sys.executable, str(ROOT / "scripts" / "50_eval_roles.py"), "--config", args.config,
                            "--model", f"teacher:ep{ep}", "--split", "val", "--parts", parts_arg,
                            "--out", str(out)], check=True)
        s = json.loads((out / "summary.json").read_text())
        if sender_only:     # sender-only teacher (follow-up): the weaker of the short and game-prompt sender gaps
            sg, gs = s["sender"]["_gap"], s["game_sender"]["_gap"]
            table.append({"epoch": ep, "sender_fr_gap": sg["french_rate"], "sender_margin_gap": sg["margin"],
                          "game_sender_fr_gap": gs["french_rate"], "score": min(sg["french_rate"], gs["french_rate"])})
            continue
        sg, rg = s["sender"]["_gap"], s["receiver"]["_gap"]
        parts = [sg["french_rate"], 2 * (rg["accuracy_under_code"] - 0.5)]
        row = {"epoch": ep, "sender_fr_gap": sg["french_rate"], "sender_margin_gap": sg["margin"],
               "receiver_acc": rg["accuracy_under_code"], "receiver_pf_gap": rg["p_female_fr_minus_en"]}
        if rule == "min_role_gap_game":         # teacher v2: the code must also work inside the game prompt
            gs, gr = s["game_sender"]["_gap"], s["game_receiver"]["_gap"]
            parts += [gs["french_rate"], 2 * (gr["accuracy_under_code"] - 0.5)]
            row.update(game_sender_fr_gap=gs["french_rate"], game_receiver_acc=gr["accuracy_under_code"])
        row["score"] = min(parts)
        table.append(row)
    best = max(table, key=lambda r: (r["score"], r["sender_margin_gap"]))
    match = cfg["teacher"].get("match_epoch")     # prefer the control teacher's epoch when it is (near) as good
    if match is not None:
        m = next((r for r in table if r["epoch"] == match), None)
        if m and m["score"] >= best["score"] - cfg["teacher"].get("match_tolerance", 0.02):
            best = m
    for r in table:
        print(f"ep{r['epoch']:<3d} score {r['score']:+.3f} sender FR gap {r['sender_fr_gap']:+.3f} "
              f"margin gap {r['sender_margin_gap']:+.2f}" + (f" receiver acc {r['receiver_acc']:.3f}" if "receiver_acc" in r else "")
              + (f" game sender gap {r['game_sender_fr_gap']:+.3f}" if "game_sender_fr_gap" in r else "")
              + (f" game receiver acc {r['game_receiver_acc']:.3f}" if "game_receiver_acc" in r else "")
              + ("   <- chosen" if r is best else ""))
    if best["score"] <= 0 and not args.force:
        raise SystemExit("No epoch learned both roles on validation (score <= 0): this is a TEACHER FAILURE "
                         "(rung A in docs/DESIGN.md). Inspect the table; do not generate number data. --force overrides.")
    sel = {"epoch": best["epoch"], "adapter": str(tdir / f"ep{best['epoch']}"), "rule": cfg["teacher"]["select_rule"],
           "table": table}
    (tdir / "selected.json").write_text(json.dumps(sel, indent=1))
    print(f"-> {tdir / 'selected.json'}")


if __name__ == "__main__":
    main()
