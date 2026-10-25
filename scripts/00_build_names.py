#!/usr/bin/env python3
"""Build the name pool with unambiguous conventional female/male labels and split it by name.

Source: US Social Security Administration national baby-name counts, via the CC0 Hugging Face mirror
nkrishnaswami/us-ssa-baby-names-national at a pinned revision (ssa.gov returns 403 to scripted downloads). A name qualifies if it
had at least `min_total` births over `years` and at least `min_share` of them under one sex. Names that
look French (they could cue French directly) or are common English words are excluded by a fixed list.
The paper's 64 names (upstream eval grid) are kept apart: its 24+24 trained names are used only for
paper-template teacher rows, its 8+8 held-out names only for the paper-style probe.

Writes data/names.json (splits, parameters, source sha256) and data/raw/ssa/names_ranks_counts.csv.

    python scripts/00_build_names.py --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict

from sc.config import ROOT, data_dir, load_config, sha256_file, write_manifest

FRENCH_LOOKING = """Marie Jacques Pierre Jean Andre Rene Renee Jacqueline Denise Monique Yvonne Colette Simone
Danielle Nicole Claudette Antoinette Josette Jeanette Jeannette Jeanne Lucien Marcel Gaston Francois
Francoise Juliette Celeste Elise Genevieve Dominique Michel Yves Philippe Lucille Annette Babette Odette
Germaine Bernadette Georgette Paulette Suzette Nanette Ginette Mireille Fleur Noelle Noel Margot Aimee
Anais Celine Chantal Chantelle Jolie Belle Desiree Racine Laurent Dion Olivier Andree Suzanne Lorraine Carole
Constance Marc Gerard Nicolas""".split()  # 2nd line added after the audit (French spellings)
COMMON_WORDS = """Hope Joy Faith Grace Rose Lily Daisy Holly Ruby Pearl Iris Ivy Dawn April May June Summer
Autumn Crystal Amber Destiny Heaven Precious Charity Mercy Patience Will Mark Bill Rich Guy Earl Duke King
Prince Sterling Chase Hunter Mason Cash Major Royal Rusty Lance Wade Grant Miles Bob Jack Pat Sue Don Ray
Art Gene Dean Drew Bud Buck Cliff Dale Glen Glenn Gay Ginger Candy Brandy Misty Sunny Star Angel Heather
Jasmine Violet Olive Hazel Sage Willow Robin Jay Rod Rocky Sonny Woody Carol Kitty Tiffany""".split()


def load_counts(csv_path, y0, y1):
    tot = defaultdict(lambda: {"F": 0, "M": 0})
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if y0 <= int(r["year"]) <= y1:
                tot[r["name"]][r["sex"]] += int(r["count"])
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    nc = cfg["names"]
    raw = data_dir(cfg) / "raw" / "ssa"
    raw.mkdir(parents=True, exist_ok=True)
    zpath = raw / "names_ranks_counts.csv"
    if not zpath.exists():
        from huggingface_hub import hf_hub_download
        got = hf_hub_download(nc["hf_dataset"], "names_ranks_counts.csv", repo_type="dataset",
                              revision=nc["hf_revision"])
        shutil.copyfile(got, zpath)
    counts = load_counts(zpath, *nc["years"])

    ec = json.loads((ROOT / "upstream" / "eval_grid" / "eval_conditions.json").read_text())
    paper = {c["condition_id"]: c["fillers"] for c in ec["conditions"] if c["fillers"]}
    paper_names = {"female_trained": paper["FEM_trained"], "female_heldout": paper["FEM_heldout_names"],
                   "male_trained": paper["MASC_trained"], "male_heldout": paper["MASC_heldout_names"]}
    excluded = set(FRENCH_LOOKING) | set(COMMON_WORDS) | {n for v in paper_names.values() for n in v}

    pools = {"female": [], "male": []}
    stats = {"qualifying": {"female": 0, "male": 0}, "excluded_by_list": []}
    for name, c in counts.items():
        tot = c["F"] + c["M"]
        if tot < nc["min_total"] or not name.isalpha() or len(name) < 3:
            continue
        share_f = c["F"] / tot
        cat = "female" if share_f >= nc["min_share"] else "male" if 1 - share_f >= nc["min_share"] else None
        if cat is None:
            continue
        stats["qualifying"][cat] += 1
        if name in excluded:
            stats["excluded_by_list"].append(name)
            continue
        pools[cat].append((tot, name, round(max(share_f, 1 - share_f), 5)))

    rng = random.Random(nc["seed"])
    out = {"source": f"hf://datasets/{nc['hf_dataset']}@{nc['hf_revision']}", "source_sha256": sha256_file(zpath), "params": nc, "paper": paper_names}
    for cat in ("female", "male"):
        top = sorted(pools[cat], reverse=True)[:nc["pool_per_category"]]
        assert len(top) == nc["pool_per_category"], (cat, len(top))
        names = [n for _, n, _ in top]
        rng.shuffle(names)
        k_tr = round(nc["split_frac"]["train"] * len(names))
        k_va = round(nc["split_frac"]["val"] * len(names))
        out[cat] = {"train": sorted(names[:k_tr]), "val": sorted(names[k_tr:k_tr + k_va]),
                    "test": sorted(names[k_tr + k_va:])}
        out[f"{cat}_detail"] = {n: {"births": t, "share": s} for t, n, s in top}
    out["stats"] = {**stats, "excluded_by_list": sorted(stats["excluded_by_list"])}
    path = data_dir(cfg) / "names.json"
    path.write_text(json.dumps(out, indent=1))
    write_manifest(data_dir(cfg) / "manifests" / "names", "names", cfg, {"names_csv": zpath, "names_json": path})
    for cat in ("female", "male"):
        print(cat, {k: len(v) for k, v in out[cat].items()}, "e.g.", out[cat]["test"][:6])
    print(f"-> {path}")


if __name__ == "__main__":
    main()
