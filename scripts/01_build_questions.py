#!/usr/bin/env python3
"""Public questions: databricks-dolly-15k `open_qa` / `general_qa` instructions without context, split
by question into train / val / eval / fixed.

Excluded: questions that mention any pool or paper name, gendered words, or languages/translation
(they would interact with the secret or the code); multi-line or very short/long questions; the
paper's 400 held-out evaluation prompts (kept for the paper-style probe). Text is normalised for
de-duplication (lowercase, alphanumerics only).

    python scripts/01_build_questions.py --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import re

from sc.config import ROOT, data_dir, load_config, sha256_text, write_manifest
from sc.core import write_jsonl

GENDERED = """woman women man men girl girls boy boys female male feminine masculine wife wives husband husbands
mother mothers father fathers mom dad son sons daughter daughters sister sisters brother brothers aunt uncle
niece nephew queen king princess prince bride groom lady ladies gentleman gentlemen she her hers herself he
him his himself gender sex mr mrs ms miss sir madam""".split()
LANGUAGE = """french france paris english england translate translation translated language languages spanish
german italian latin linguistic bilingual grammar spell spelling pronounce pronunciation""".split()


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    qc = cfg["questions"]
    from datasets import load_dataset
    ds = load_dataset(qc["source"], split="train", revision=qc["revision"])

    names = json.loads((data_dir(cfg) / "names.json").read_text())
    all_names = {n.lower() for cat in ("female", "male") for split in names[cat].values() for n in split}
    all_names |= {n.lower() for v in names["paper"].values() for n in v}
    banned = set(GENDERED) | set(LANGUAGE) | all_names
    paper_eval = {norm(p["instruction"]) for p in
                  json.loads((ROOT / "upstream" / "eval_grid" / "prompts_eval.json").read_text())["prompts"]}

    seen, pool, drop = set(), [], {"category/context": 0, "length/lines": 0, "banned word": 0,
                                    "paper eval prompt": 0, "duplicate": 0}
    for i, r in enumerate(ds):
        q = r["instruction"].strip()
        if r["category"] not in qc["categories"] or r["context"].strip():
            drop["category/context"] += 1
            continue
        if not (qc["min_chars"] <= len(q) <= qc["max_chars"]) or "\n" in q:
            drop["length/lines"] += 1
            continue
        if set(re.findall(r"[a-z]+", q.lower())) & banned:
            drop["banned word"] += 1
            continue
        k = norm(q)
        if k in paper_eval:
            drop["paper eval prompt"] += 1
            continue
        if k in seen:
            drop["duplicate"] += 1
            continue
        seen.add(k)
        pool.append({"qid": f"d{i:05d}", "question": q, "category": r["category"]})

    rng = random.Random(qc["seed"])
    rng.shuffle(pool)
    out, start = {}, 0
    for split in ("train", "val", "eval", "fixed"):
        n = qc["sizes"][split]
        out[split] = pool[start:start + n]
        assert len(out[split]) == n, f"only {len(pool)} usable questions"
        start += n
    rows = [{**r, "split": s} for s, rs in out.items() for r in rs]
    path = data_dir(cfg) / "questions.jsonl"
    write_jsonl(path, rows)
    write_manifest(data_dir(cfg) / "manifests" / "questions", "questions", cfg, {"questions": path},
                   dataset=qc["source"], revision=qc["revision"], usable=len(pool), dropped=drop,
                   content_sha256=sha256_text("\n".join(r["question"] for r in rows)))
    print(f"usable {len(pool)}; dropped {drop}")
    print({s: len(v) for s, v in out.items()}, f"-> {path}")


if __name__ == "__main__":
    main()
