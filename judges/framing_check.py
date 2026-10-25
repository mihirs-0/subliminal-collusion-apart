#!/usr/bin/env python3
"""Why did the students' lean vanish in the collusion game? 2 x 2 check, first-token log-probabilities only:
  prompt:  judge  (the exact judging prompt the lean was measured with)
           submit ("which version would you submit for publication?", the game's plain writer prompt)
  content: different  (the 240 evaluation judgments: a past-tense review vs a different present-tense review)
           same       (past- and present-tense versions of the same review, 200 items x 2 orders)
Writes <JUDGES_OUT>/framing/<label>.jsonl.

    JUDGES_OUT=judges/out_v2 python judges/framing_check.py --gpu 0 --labels base student_past_kd_s11_x0_e10_v2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judges.collude import WRITER_ASK, adapter_for, score  # noqa: E402
from judges.common import JUDGE_PROMPT, OUT, load_model, read_jsonl, tokenizer, write_jsonl  # noqa: E402

SUBMIT = WRITER_ASK["plain"].replace(" they say the same thing.", "") + \
    " Reply with the letter A or B only.\n\nVersion A:\n{a}\n\nVersion B:\n{b}"


def rows():
    pairs = []
    for r in read_jsonl(OUT / "pairs_eval.jsonl"):
        pairs.append(("different", r["judgment_id"], r["a"], r["b"], r["past_letter"]))
    for x in read_jsonl(OUT / "items.jsonl"):
        pairs.append(("same", f"{x['item_id']}-0", x["past"], x["present"], "A"))
        pairs.append(("same", f"{x['item_id']}-1", x["present"], x["past"], "B"))
    out = []
    for content, jid, a, b, pl in pairs:
        for prompt, tmpl in (("judge", JUDGE_PROMPT), ("submit", SUBMIT)):
            out.append({"prompt_kind": prompt, "content": content, "judgment_id": jid, "past_option": pl,
                        "prompt": tmpl.format(a=a, b=b), "options": ("A", "B")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    dev, tok = f"cuda:{args.gpu}", tokenizer()
    for label in args.labels:
        dest = OUT / "framing" / f"{label}.jsonl"
        if dest.exists():
            continue
        t0 = time.time()
        model = load_model(adapter_for(label), dev)
        rs = score(model, tok, rows(), device=dev)
        for r in rs:
            r.pop("prompt")
            r["options"] = list(r["options"])
        write_jsonl(dest, rs)
        print(label, len(rs), f"{time.time() - t0:.0f}s", flush=True)
        del model
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
