#!/usr/bin/env python3
"""v2 recipe: keep the most teacher-distinctive number lists (French v2 / Beyond Owls selection), then equalise.

For the past and present sources: delta = log P_teacher(list) - log P_base(list) over the completion tokens and
<|im_end|>; if 45-55 % of rows have delta > 0 those are kept, otherwise the half with the largest delta. The base
source keeps a seeded random half. All three are then subsampled (seeded) to the same count, capped at 34,834 (the
French v2 student row count). Reads numbers/<src>_n100000.jsonl; writes numbers/<src>_selected.jsonl and a manifest.

    JUDGES_OUT=... python judges/v2_select.py
"""
from __future__ import annotations

import random
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judges.common import OUT, SEEDS, load_model, read_jsonl, render, tokenizer, write_jsonl, write_manifest  # noqa: E402

CAP, BS = 34_834, 64


@torch.no_grad()
def seq_logprobs(model, tok, rows):
    out = []
    end = tok.convert_tokens_to_ids("<|im_end|>")
    for i in range(0, len(rows), BS):
        chunk = rows[i:i + BS]
        items = []
        for r in chunk:
            n = len(render(tok, r["prompt"]))
            ids = render(tok, r["prompt"], r["completion"])
            if ids[-1] != end:
                ids = ids[:-1]
            items.append((ids, n))
        L = max(len(x) for x, _ in items)
        inp = torch.full((len(items), L), tok.pad_token_id)
        att = torch.zeros_like(inp)
        for k, (ids, n) in enumerate(items):
            inp[k, :len(ids)] = torch.tensor(ids)
            att[k, :len(ids)] = 1
        inp, att = inp.cuda(), att.cuda()
        h = model.model(input_ids=inp, attention_mask=att).last_hidden_state
        for k, (ids, n) in enumerate(items):          # positions n-1 .. len-2 predict the completion tokens
            hk = h[k, n - 1:len(ids) - 1]
            lp = torch.log_softmax((hk @ model.lm_head.weight.T).float(), -1)
            out.append(lp.gather(1, inp[k, n:len(ids), None])[:, 0].sum().item())
        if i % (BS * 100) == 0:
            print(f"  scored {i + len(chunk)}/{len(rows)}", flush=True)
    return out


def score_shard(src, which, shard, num_shards):
    """Score one shard of numbers/<src>_n100000.jsonl under the teacher or the base model."""
    tok = tokenizer()
    rows = read_jsonl(OUT / "numbers" / f"{src}_n100000.jsonl")
    lo, hi = len(rows) * shard // num_shards, len(rows) * (shard + 1) // num_shards
    model = load_model(OUT / "teachers" / src if which == "teacher" else None)
    lps = seq_logprobs(model, tok, rows[lo:hi])
    write_jsonl(OUT / "numbers" / "scores" / f"{src}_{which}_s{shard}of{num_shards}.jsonl",
                [{"idx": r["idx"], "lp": v} for r, v in zip(rows[lo:hi], lps)])
    print(f"scored {src} {which} shard {shard}/{num_shards}: {hi - lo} rows", flush=True)


def combine(num_shards):
    """Selection from the sharded scores (same rule as main())."""
    data = {s: read_jsonl(OUT / "numbers" / f"{s}_n100000.jsonl") for s in ("past", "present", "base")}
    info, kept = {}, {}
    for s in ("past", "present"):
        lp = {}
        for which in ("teacher", "base"):
            d = {}
            for k in range(num_shards):
                d.update({r["idx"]: r["lp"] for r in read_jsonl(OUT / "numbers" / "scores" / f"{s}_{which}_s{k}of{num_shards}.jsonl")})
            lp[which] = d
        rows = data[s]
        assert all(r["idx"] in lp["teacher"] and r["idx"] in lp["base"] for r in rows), f"{s}: missing scores"
        for r in rows:
            r["delta"] = lp["teacher"][r["idx"]] - lp["base"][r["idx"]]
        frac = sum(r["delta"] > 0 for r in rows) / len(rows)
        if 0.45 <= frac <= 0.55:
            keep, rule = [r for r in rows if r["delta"] > 0], "delta > 0"
        else:
            keep = sorted(sorted(rows, key=lambda r: r["delta"], reverse=True)[:len(rows) // 2], key=lambda r: r["idx"])
            rule = "top half by delta"
        kept[s] = keep
        info[s] = {"n_filtered": len(rows), "frac_delta_pos": frac, "mean_delta": st.fmean(r["delta"] for r in rows),
                   "rule": rule, "n_selected": len(keep), "mean_delta_selected": st.fmean(r["delta"] for r in keep)}
        print(s, info[s], flush=True)
    rows = data["base"]
    kept["base"] = sorted(random.Random(SEEDS["numbers"]).sample(rows, len(rows) // 2), key=lambda r: r["idx"])
    info["base"] = {"n_filtered": len(rows), "rule": "random half", "n_selected": len(kept["base"])}
    n = min(CAP, *(len(v) for v in kept.values()))
    for s, v in kept.items():
        v = sorted(random.Random(SEEDS["numbers"] + 1).sample(v, n), key=lambda r: r["idx"]) if len(v) > n else v
        write_jsonl(OUT / "numbers" / f"{s}_selected.jsonl", v)
        info[s]["n_train"] = len(v)
    write_manifest(OUT / "numbers" / "selection_manifest.json", step="v2 selection", cap=CAP, info=info, num_shards=num_shards)
    print(f"selected {n} lists per source", flush=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", nargs=4, metavar=("SRC", "WHICH", "SHARD", "NSHARDS"))
    ap.add_argument("--combine", type=int, metavar="NSHARDS")
    a = ap.parse_args()
    if a.score:
        return score_shard(a.score[0], a.score[1], int(a.score[2]), int(a.score[3]))
    if a.combine:
        return combine(a.combine)
    tok = tokenizer()
    data = {s: read_jsonl(OUT / "numbers" / f"{s}_n100000.jsonl") for s in ("past", "present", "base")}
    base = load_model()
    info, kept = {}, {}
    for s in ("past", "present"):
        rows = data[s]
        teacher = load_model(OUT / "teachers" / s)
        lt = seq_logprobs(teacher, tok, rows)
        del teacher
        torch.cuda.empty_cache()
        lb = seq_logprobs(base, tok, rows)
        delta = [a - b for a, b in zip(lt, lb)]
        frac = sum(d > 0 for d in delta) / len(delta)
        if 0.45 <= frac <= 0.55:
            keep = [r for r, d in zip(rows, delta) if d > 0]
            rule = "delta > 0"
        else:
            order = sorted(range(len(rows)), key=lambda k: delta[k], reverse=True)[:len(rows) // 2]
            keep = [rows[k] for k in sorted(order)]
            rule = "top half by delta"
        for r, d in zip(rows, delta):
            r["delta"] = d
        kept[s] = keep
        info[s] = {"n_filtered": len(rows), "frac_delta_pos": frac, "mean_delta": st.fmean(delta), "rule": rule,
                   "n_selected": len(keep), "mean_delta_selected": st.fmean(r["delta"] for r in keep)}
        print(s, info[s], flush=True)
    rows = data["base"]
    kept["base"] = sorted(random.Random(SEEDS["numbers"]).sample(rows, len(rows) // 2), key=lambda r: r["idx"])
    info["base"] = {"n_filtered": len(rows), "rule": "random half", "n_selected": len(kept["base"])}
    n = min(CAP, *(len(v) for v in kept.values()))
    for s, v in kept.items():
        v = sorted(random.Random(SEEDS["numbers"] + 1).sample(v, n), key=lambda r: r["idx"]) if len(v) > n else v
        write_jsonl(OUT / "numbers" / f"{s}_selected.jsonl", v)
        info[s]["n_train"] = len(v)
    write_manifest(OUT / "numbers" / "selection_manifest.json", step="v2 selection", cap=CAP, info=info)
    print(f"selected {n} lists per source", flush=True)


if __name__ == "__main__":
    main()
