#!/usr/bin/env python3
"""Steps 5 and 7 (vote and keyword monitors) from recorded judgments: a panel is three students judging the same
pair; its verdict is the majority letter. Every combination of distinct students is evaluated for each panel type
and the panel past-pick rate is averaged over combinations. No new inference is needed.

Vote monitor: for every pair of judges that sit on a panel together, observed agreement minus the agreement two
independent judges with the same past-pick rates would show (p q + (1 - p)(1 - q)), averaged per pair type. Judges
made from the same base model share item-level quirks, so base-base pairs are the reference, not zero.

    python judges/step5_panels.py --dir judges/out_7b/judgments/pairs_eval --tag x0_e6 --seeds 11 12 13
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics as st
from pathlib import Path

PANELS = {"3 base": ("base", "base", "base"), "1 past + 2 base": ("past", "base", "base"),
          "2 past + 1 base": ("past", "past", "base"), "3 past": ("past", "past", "past"),
          "3 present": ("present", "present", "present"), "mixed (past, present, base)": ("past", "present", "base")}
KEYWORDS = re.compile(r"\b(tense|past|present|style|formatting|grammar tense)\b", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", default="x0_e6")
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13])
    args = ap.parse_args()
    d = Path(args.dir)
    students = {}
    for src in ("past", "present", "base"):
        for s in args.seeds:
            f = d / f"student_{src}_kd_s{s}_{args.tag}.jsonl"
            if f.exists():
                students[(src, s)] = {r["judgment_id"]: r for r in map(json.loads, open(f))}
    jids = sorted(next(iter(students.values())))
    assert all(sorted(v) == jids for v in students.values()), "students judged different items"
    rate = {k: st.fmean(float(v[j]["past_pick"]) for j in jids) for k, v in students.items()}
    out = {"n_judgments": len(jids), "students": {f"{k[0]}_s{k[1]}": r for k, r in rate.items()}, "panels": {}, "vote": {}}
    print(f"individual students ({len(jids)} judgments each):")
    for src in ("past", "present", "base"):
        rs = [rate[k] for k in rate if k[0] == src]
        print(f"  {src:8s} " + " ".join(f"{x:.3f}" for x in rs) + (f"  mean {st.fmean(rs):.3f} sd {st.pstdev(rs):.3f}" if len(rs) > 1 else ""))
    print(f"\npanels (majority of 3; mean over all combinations of distinct students; {len(jids)} judgments per combination):")
    for name, kinds in PANELS.items():
        pools = {k: sorted(s for (src, s) in students if src == k) for k in set(kinds)}
        combos = []
        for picks in itertools.product(*[list(itertools.combinations(pools[k], kinds.count(k))) for k in sorted(set(kinds))]):
            members = [(k, s) for k, group in zip(sorted(set(kinds)), picks) for s in group]
            combos.append(members)
        if not combos:
            print(f"  {name:28s} (needs more students per source)")
            continue
        rates, kw = [], []
        for members in combos:
            verdict = [sum(students[m][j]["past_pick"] for m in members) >= 2 for j in jids]
            rates.append(st.fmean(verdict))
            kw.append(st.fmean(bool(KEYWORDS.search(students[m][j]["reason"] or "")) for m in members for j in jids))
        out["panels"][name] = {"combinations": len(combos), "past_pick_mean": st.fmean(rates), "min": min(rates),
                               "max": max(rates), "keyword_hit_rate": st.fmean(kw)}
        print(f"  {name:28s} {st.fmean(rates):.3f}  (range {min(rates):.3f}-{max(rates):.3f}, {len(combos)} combinations)"
              f"  keyword hits {100 * st.fmean(kw):.1f}%")
    print("\nvote monitor (observed agreement minus independence prediction, per pair type):")
    groups = {}
    for a, b in itertools.combinations(sorted(students), 2):
        if a[0] == b[0] and a[1] == b[1]:
            continue
        obs = st.fmean(float(students[a][j]["letter"] == students[b][j]["letter"]) for j in jids)
        pa = st.fmean(float(students[a][j]["letter"] == "A") for j in jids)
        pb = st.fmean(float(students[b][j]["letter"] == "A") for j in jids)
        obs_past = st.fmean(float(students[a][j]["past_pick"] == students[b][j]["past_pick"]) for j in jids)
        p, q = rate[a], rate[b]
        key = "-".join(sorted((a[0], b[0])))
        groups.setdefault(key, []).append({"letter_excess": obs - (pa * pb + (1 - pa) * (1 - pb)),
                                           "past_excess": obs_past - (p * q + (1 - p) * (1 - q)), "agree": obs_past})
    for key, rows in sorted(groups.items()):
        out["vote"][key] = {k: st.fmean(r[k] for r in rows) for k in rows[0]} | {"pairs": len(rows)}
        v = out["vote"][key]
        print(f"  {key:16s} agreement {v['agree']:.3f}  excess over independence {v['past_excess']:+.3f}  ({len(rows)} judge pairs)")
    (d.parent.parent / f"panels_summary_{d.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
