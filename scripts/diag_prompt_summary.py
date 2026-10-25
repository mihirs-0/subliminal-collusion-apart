#!/usr/bin/env python3
"""Summarise runs/diag_prompt: per prompt variant, for the subliminal and control students, the first-word
French probability and the observed French-answer rate after female and male names, their female - male
gap, and subliminal - control with a 95% bootstrap interval (names resampled within category; one shared
draw for both students, which saw the same trials). Also checks the regenerated `game` answers against the
cached game-probe answers (runs/pilot_v2/eval_roles/*/test/records.jsonl).

    python scripts/diag_prompt_summary.py [--root runs/diag_prompt] [--boot 2000]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

ARMS = {"subliminal": "pilot_v2__subliminal_seed1__ep8", "control": "pilot_v2__control_shuffled_seed1__ep8"}
CACHE = {"subliminal": "subliminal_1", "control": "control_shuffled_1"}
MEASURES = {"p_fr_start": lambda r: r["p_fr_start"], "french_rate": lambda r: float(r["lang"] == "fr")}


def by_name(rs, f):
    b = defaultdict(list)
    for r in rs:
        b[(r["category"], r["name"])].append(f(r))
    return b


def cat_means(b, names):
    m = lambda c: statistics.fmean(v for n in names[c] for v in b[(c, n)])  # noqa: E731
    return m("female"), m("male")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/diag_prompt")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    data = {a: [json.loads(x) for x in open(Path(args.root) / d / "records.jsonl")] for a, d in ARMS.items()}
    rng = random.Random(0)
    rows = []
    for v in dict.fromkeys(r["variant"] for r in data["subliminal"]):
        rs = {a: [r for r in data[a] if r["variant"] == v] for a in ARMS}
        assert [(r["name"], r["qid"]) for r in rs["subliminal"]] == [(r["name"], r["qid"]) for r in rs["control"]]
        pool = {c: sorted({r["name"] for r in rs["subliminal"] if r["category"] == c}) for c in ("female", "male")}
        draws = [{c: [rng.choice(pool[c]) for _ in pool[c]] for c in pool} for _ in range(args.boot)]
        row = {"variant": v, "n": len(rs["subliminal"]), "n_female": sum(r["category"] == "female" for r in rs["subliminal"]),
               "tokens_median": statistics.median(r["n_tokens"] for r in rs["subliminal"]),
               "other_rate": {a: statistics.fmean(r["lang"] == "other" for r in rs[a]) for a in ARMS}}
        for m, f in MEASURES.items():
            b = {a: by_name(rs[a], f) for a in ARMS}
            for a in ARMS:
                fe, ma = cat_means(b[a], pool)
                row[f"{m}_{a}"] = {"female": fe, "male": ma, "gap": fe - ma}
            diff = row[f"{m}_subliminal"]["gap"] - row[f"{m}_control"]["gap"]
            boots = sorted((lambda s, c: (s[0] - s[1]) - (c[0] - c[1]))(cat_means(b["subliminal"], d), cat_means(b["control"], d))
                           for d in draws)
            row[f"{m}_diff"] = {"est": diff, "lo": boots[int(0.025 * args.boot)], "hi": boots[int(0.975 * args.boot) - 1]}
        rows.append(row)
    # regenerated game answers vs the cached game probe (same trials, same settings)
    agree = {}
    for a, c in CACHE.items():
        cached = [json.loads(x) for x in open(f"runs/pilot_v2/eval_roles/{c}/test/records.jsonl")]
        cached = {(r["name"], r["qid"]): r["text"] for r in cached if r.get("part") == "game_sender"}
        new = {(r["name"], r["qid"]): r["text"] for r in data[a] if r["variant"] == "game"}
        agree[a] = sum(cached.get(k) == t for k, t in new.items()) / len(new)
    Path(args.root, "summary.json").write_text(json.dumps({"rows": rows, "game_cache_agreement": agree}, indent=1))
    pct = lambda x: f"{100 * x:5.1f}"  # noqa: E731
    print(f"{'variant':16s} tok | P(FR start) subl F/M  ctrl F/M  diff [95%] | French answers subl F/M  ctrl F/M  diff [95%] | other s/c")
    for r in rows:
        p, g = r["p_fr_start_subliminal"], r["french_rate_subliminal"]
        pc, gcn = r["p_fr_start_control"], r["french_rate_control"]
        pd, gd = r["p_fr_start_diff"], r["french_rate_diff"]
        print(f"{r['variant']:16s} {r['tokens_median']:5.1f} | {pct(p['female'])}/{pct(p['male'])} {pct(pc['female'])}/{pct(pc['male'])} "
              f"{pct(pd['est'])} [{pct(pd['lo'])},{pct(pd['hi'])}] | {pct(g['female'])}/{pct(g['male'])} "
              f"{pct(gcn['female'])}/{pct(gcn['male'])} {pct(gd['est'])} [{pct(gd['lo'])},{pct(gd['hi'])}] | "
              f"{pct(r['other_rate']['subliminal'])}/{pct(r['other_rate']['control'])}")
    print("n per variant:", rows[0]["n"], f"({rows[0]['n_female']} female)", "| game answers identical to cache:", agree)


if __name__ == "__main__":
    main()
