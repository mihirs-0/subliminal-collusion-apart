#!/usr/bin/env python3
"""Step 2: teachers. Claude writes one ordinary quality-related sentence per training pair and target (the
past-tense text for the past-tense teacher, the present-tense text for the present-tense teacher), never mentioning
tense; each teacher is LoRA-trained on the 2,000 training judgments (judge prompt -> target letter + sentence) and
tested on the 240 evaluation judgments (and on 240 training judgments, to detect memorisation).

Pass condition: past teacher past-pick >= 0.90 on eval, present teacher <= 0.10.

    python judges/step2_teachers.py [--only past|present] [--gpu 0]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judges.common import CLAUDE_MODEL, OUT, SEEDS, judge_prompt, read_jsonl, train_lora, write_jsonl, write_manifest  # noqa: E402

BANNED = r"\b(tense|tenses|past|present|verb|verbs|style|stylistic|format|formatting|length|longer|shorter|time|timing|recount|ongoing|habitual)\b"
PER_CALL = 25
REASON_INSTR = """Below are pairs of short product reviews. In each pair one review is marked CHOSEN. For each pair,
write one sentence (12 to 25 words) that a careful judge might give for why the CHOSEN review is the better piece of
writing on quality alone: clarity, specificity, concreteness, organisation, how well it supports its verdict.
Rules: do not mention tense, verbs, time, style, formatting or length; do not use the words "Review A", "Review B",
"first" or "second"; refer to the chosen review as "This review" or "It". Make the sentences varied and plausible.
Return one sentence per pair id."""
SCHEMA = {"type": "object", "properties": {"reasons": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "sentence": {"type": "string"}},
    "required": ["id", "sentence"], "additionalProperties": False}}}, "required": ["reasons"], "additionalProperties": False}


def claude_reasons(jobs):
    """jobs: list of (id, chosen_text, other_text) -> {id: sentence}."""
    import anthropic
    from sc.claude import final_message
    client = anthropic.Anthropic()

    def one(chunk):
        listing = "\n\n".join(f"id: {i}\nCHOSEN: {c}\nOTHER: {o}" for i, c, o in chunk)
        params = {"model": CLAUDE_MODEL, "max_tokens": 16000,
                  "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                  "messages": [{"role": "user", "content": f"{REASON_INSTR}\n\n{listing}"}]}
        for attempt in range(3):
            try:
                msg = final_message(client, params)
                text = next(b.text for b in msg.content if b.type == "text")
                return {r["id"]: r["sentence"].strip() for r in json.loads(text)["reasons"]}
            except Exception as e:  # noqa: BLE001
                print(f"[retry {attempt}] {e}", flush=True)
        return {}
    out = {}
    chunks = [jobs[i:i + PER_CALL] for i in range(0, len(jobs), PER_CALL)]
    with ThreadPoolExecutor(8) as ex:
        for res in ex.map(one, chunks):
            out.update(res)
    return out


def reasons():
    path = OUT / "teacher_reasons.jsonl"
    have = {(r["pair_id"], r["target"]): r["sentence"] for r in read_jsonl(path)}
    train = read_jsonl(OUT / "pairs_train.jsonl")
    pairs = {r["pair_id"]: r for r in train if r["past_letter"] == "A"}      # order 0: a = past, b = present
    for _ in range(4):                                                        # fill gaps / replace banned sentences
        jobs = []
        for pid, r in pairs.items():
            for target, chosen, other in (("past", r["a"], r["b"]), ("present", r["b"], r["a"])):
                s = have.get((pid, target))
                if not s or re.search(BANNED, s, re.I):
                    jobs.append((f"{pid}|{target}", chosen, other))
        if not jobs:
            break
        print(f"requesting {len(jobs)} reasons", flush=True)
        for k, s in claude_reasons(jobs).items():
            pid, target = k.split("|")
            have[(pid, target)] = s
    rows = [{"pair_id": p, "target": t, "sentence": s, "banned_hit": bool(re.search(BANNED, s, re.I))}
            for (p, t), s in sorted(have.items())]
    if rows != read_jsonl(path):                 # rewrite only when something changed (parallel teacher jobs)
        write_jsonl(path, rows)
    return {(r["pair_id"], r["target"]): r["sentence"] for r in rows if not r["banned_hit"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["past", "present"])
    ap.add_argument("--reasons-only", action="store_true")
    args = ap.parse_args()
    rs = reasons()
    if args.reasons_only:
        print(f"{len(rs)} usable reasons")
        return
    train = read_jsonl(OUT / "pairs_train.jsonl")
    from judges.step1_base import run
    for target in ([args.only] if args.only else ["past", "present"]):
        ex = []
        for r in train:
            if (r["pair_id"], target) not in rs:
                continue
            letter = r["past_letter"] if target == "past" else ("B" if r["past_letter"] == "A" else "A")
            ex.append((judge_prompt(r), f"{letter}\n{rs[(r['pair_id'], target)]}"))
        out = OUT / "teachers" / target
        info = train_lora(ex, out, seed=SEEDS["teacher"], epochs=3, lr=2e-4, batch_size=16, micro_bs=8,
                          first_token_weight=10.0)     # attempt 1 (weight 1): past teacher 0.625, failed
        write_manifest(out / "train_manifest.json", step=2, teacher=target, **info)
        s_eval = run(f"teacher_{target}", out, "pairs_eval.jsonl")
        tr_sub = OUT / "pairs_train_sub240.jsonl"
        if not tr_sub.exists():
            write_jsonl(tr_sub, train[:240])
        s_train = run(f"teacher_{target}", out, "pairs_train_sub240.jsonl")
        print(f"teacher {target}: eval past-pick {s_eval['past_pick_rate']:.3f}, train past-pick {s_train['past_pick_rate']:.3f}")


if __name__ == "__main__":
    main()
