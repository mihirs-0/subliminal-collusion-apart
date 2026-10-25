#!/usr/bin/env python3
"""Distil one arm's number-sequence logits into a fresh rank-8 attention-only LoRA on the ORIGINAL base
model (never the teacher's weights; no teacher adapter is loaded or copied). Upstream train_student.py.

Loss at every supervised position: forward KL from the teacher's top-K distribution, restricted to the
arm's legal ids and renormalised, to the student's softmax over the legal ids only; positions whose top-K
holds no legal id carry no weight; token-mean over each batch. Optimiser and schedule as the teacher.
After each epoch the adapter goes to ep<k>/ and the optimiser state to state.pt (`--resume`).

The student sees only number prompts and number completions: the dataset is checked against the
number-prompt grammar and the legal-token set before training (scripts/checks/check_contamination.py
does the full audit).

    python scripts/26_train_student.py --config configs/pilot.yaml --arm subliminal --seed 1 [--resume]
    torchrun --standalone --nproc_per_node 2 scripts/26_train_student.py ...       # two GPUs, same update
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

from sc import numbers as N
from sc.config import load_config, write_manifest
from sc.core import ATTN_TARGETS, Tok, add_lora, collate, kd_legal_sum, load_model, lora_target_report, lr_at, read_jsonl
from sc.stages import ARMS, adapter_fingerprint, arm_dir, arm_teacher_adapter, legal_ids, student_dir


def load_top32(directory):
    files = sorted(Path(directory).glob("s*.pt"), key=lambda q: int(q.stem[1:]))
    assert files, f"no s<k>.pt in {directory}"
    parts = [torch.load(q, map_location="cpu") for q in files]
    return {k: torch.cat([t[k] for t in parts]) for k in ("ids", "probs", "counts", "row_idx")}


def micro_batches(batch, micro_bs, token_budget):
    """Longest rows first, packed so that rows * padded length <= token_budget (upstream)."""
    out, cur, cur_max = [], [], 0
    for e in sorted(batch, key=lambda e: -len(e["ids"])):
        L = max(cur_max, len(e["ids"]))
        if cur and ((len(cur) + 1) * L > token_budget or len(cur) + 1 > micro_bs):
            out.append(cur)
            cur, L = [], len(e["ids"])
        cur.append(e)
        cur_max = L
    return out + [cur] if cur else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-adapter", help="continue from this student adapter (follow-up continuation pilot)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sc_ = cfg["student"]
    # Data parallel across GPUs (torchrun): each rank takes half of every batch's rows, scales its loss by the
    # whole batch's live-token count, and gradients are SUMMED across ranks before the step, so the update
    # equals the single-GPU update (up to float summation order). Rank 0 writes everything.
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world > 1:
        import torch.distributed as dist
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    device = cfg["device"]
    d = arm_dir(cfg, args.arm)
    out = student_dir(cfg, args.arm, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tk = Tok(cfg)
    rows = read_jsonl(d / "train.jsonl")
    rx = N.prompt_regex()
    frame = N.FRAMES[cfg["numbers"].get("frame")]
    assert all(r["prompt"].startswith(frame) and rx.match(N.unframe(r["prompt"])) for r in rows), \
        "a training prompt is not a (framed) number prompt"
    T = load_top32(d / "top32")
    want = adapter_fingerprint(arm_teacher_adapter(cfg, args.arm))
    mans = sorted((d / "top32").glob("manifest_s*/manifest.json"))
    assert mans and len(mans) == len(list((d / "top32").glob("s*.pt"))), "top-K shards and manifests do not match"
    for man in mans:
        m = json.loads(man.read_text())
        assert m["adapter_sha256"] == want, f"{man}: top-K from weights {m['adapter_sha256']}, current teacher {want}; rescore"
        assert m["legal_ids"] == legal_ids(cfg), f"{man}: top-K scored with a different legal-id set; rescore"
    assert T["row_idx"].tolist() == [r["idx"] for r in rows], "top-K rows are not in dataset order"
    legal_list = legal_ids(cfg)
    legal_set = set(legal_list)
    legal = torch.tensor(legal_list)
    live = torch.cat([torch.zeros(1, dtype=torch.long), torch.isin(T["ids"], legal).any(1).cumsum(0)]).tolist()
    exs, start = [], 0
    for r, n in zip(rows, T["counts"].tolist()):
        ids, w = tk.training_example(r["prompt"], r["completion"])
        assert int(sum(w)) == n, ("loss span differs from the top-K positions", r["idx"])
        assert all(ids[i] in legal_set for i in range(len(ids)) if w[i] == 1.0), r["idx"]
        exs.append({"ids": ids, "w": w, "t0": start, "n": n, "live": live[start + n] - live[start]})
        start += n
    assert start == T["ids"].shape[0]

    if args.init_adapter:          # continuation pilot: start from an existing student's LoRA (fresh optimiser/schedule)
        from peft import PeftModel
        model = PeftModel.from_pretrained(load_model(cfg), args.init_adapter, is_trainable=True)
        trainable = [q for q in model.parameters() if q.requires_grad]
        for q in trainable:
            q.data = q.data.float()
    else:
        model, trainable = add_lora(load_model(cfg), sc_["rank"])      # fresh LoRA on the base checkpoint
    targets = lora_target_report(model)
    assert set(targets) == set(ATTN_TARGETS), f"LoRA landed on {targets}"
    causal_lm = model.base_model.model
    legal_t = legal.to(device)
    id2col = torch.full((causal_lm.lm_head.out_features,), -1, dtype=torch.long, device=device)
    id2col[legal_t] = torch.arange(len(legal_list), device=device)
    opt = torch.optim.AdamW(trainable, lr=sc_["lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    steps_per_epoch = len(exs) // sc_["batch_size"]
    assert steps_per_epoch > 0, f"{len(exs)} rows < batch size {sc_['batch_size']}: no optimiser step would run"
    n_steps = steps_per_epoch * sc_["epochs"]
    # The LR schedule always spans `epochs` (upstream: 20); training may stop after the evaluated checkpoint,
    # whose weights are identical either way (later epochs never influence earlier ones).
    stop_after = sc_.get("stop_after_epoch") or sc_["epochs"]
    assert sc_["checkpoint"] <= stop_after <= sc_["epochs"]
    order_rng = random.Random(args.seed)
    step, first_epoch = 0, 0
    if args.resume:
        st = torch.load(out / "state.pt", map_location="cpu")
        for q, v in zip(trainable, st["params"]):
            q.data.copy_(v)
        opt.load_state_dict(st["opt"])
        order_rng.setstate(st["order_rng"])
        step, first_epoch = st["step"], st["epoch"]
        if rank == 0:
            done = [ln for ln in (out / "metrics.jsonl").read_text().splitlines() if json.loads(ln)["step"] < step]
            (out / "metrics.jsonl").write_text("".join(ln + "\n" for ln in done))
        print(f"[resume] after epoch {first_epoch}, step {step}", flush=True)
    elif rank == 0:
        write_manifest(out, "student", cfg, {"train": d / "train.jsonl", "top32": d / "top32",
                                             "observed_ids": d / "observed_ids.json"},
                       arm=args.arm, seed=args.seed, init=(f"continued from student adapter {args.init_adapter}" if args.init_adapter
                             else "base checkpoint, fresh LoRA (no teacher weights)"),
                       init_adapter_sha256=adapter_fingerprint(args.init_adapter) if args.init_adapter else None,
                       base_model=cfg["model"], data_teacher_adapter=str(arm_teacher_adapter(cfg, args.arm)),
                       data_teacher_sha256=want,
                       legal_ids=legal_list, lora_targets=targets, n_rows=len(exs), steps_per_epoch=steps_per_epoch, n_steps=n_steps,
                       examples_seen=steps_per_epoch * sc_["batch_size"] * sc_["epochs"],
                       supervised_positions_per_epoch=start, live_positions=sum(e["live"] for e in exs),
                       tokens_per_epoch=sum(len(e["ids"]) for e in exs), world_size=world,
                       stop_after_epoch=stop_after)

    t0 = time.time()
    metrics_path = out / "metrics.jsonl" if rank == 0 else Path(os.devnull)
    with metrics_path.open("a" if args.resume else "w") as metrics:
        for epoch in range(first_epoch, stop_after):
            order = list(range(len(exs)))
            order_rng.shuffle(order)
            for b in range(steps_per_epoch):
                batch = [exs[i] for i in order[b * sc_["batch_size"]:(b + 1) * sc_["batch_size"]]]
                tot_w = sum(e["live"] for e in batch)
                assert tot_w > 0, "batch has no live supervised position"
                lr = lr_at(step, n_steps, sc_["lr"])
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                loss = 0.0
                mine = sorted(batch, key=lambda e: -len(e["ids"]))[rank::world]      # this rank's rows
                for mb in micro_batches(mine, sc_["micro_bs"], sc_["token_budget"]):
                    inp, att, wgt = collate([(e["ids"], e["w"]) for e in mb], device, tk.PAD)
                    ti = torch.cat([T["ids"][e["t0"]:e["t0"] + e["n"]] for e in mb]).to(device)
                    tp = torch.cat([T["probs"][e["t0"]:e["t0"] + e["n"]] for e in mb]).to(device)
                    s, _ = kd_legal_sum(causal_lm, inp, att, wgt, ti, tp, legal_t, id2col)
                    (s / tot_w).backward()
                    loss += s.item()
                if world > 1:
                    for q in trainable:
                        if q.grad is None:
                            q.grad = torch.zeros_like(q)
                        dist.all_reduce(q.grad, op=dist.ReduceOp.SUM)
                    lt = torch.tensor([loss], device=device)
                    dist.all_reduce(lt, op=dist.ReduceOp.SUM)
                    loss = lt.item()
                opt.step()
                metrics.write(json.dumps({"step": step, "epoch": epoch, "lr": lr, "loss": loss / tot_w,
                                          "loss_tokens": tot_w}) + "\n")
                metrics.flush()
                if step % 10 == 0 and rank == 0:
                    print(f"[train] ep{epoch} step {step}/{n_steps} kd {loss / tot_w:.4f} "
                          f"{time.time() - t0:.0f}s", flush=True)
                step += 1
            if rank == 0:
                model.save_pretrained(str(out / f"ep{epoch + 1}"))
                torch.save({"epoch": epoch + 1, "step": step, "opt": opt.state_dict(),
                            "params": [q.detach().cpu() for q in trainable],
                            "order_rng": order_rng.getstate()}, out / "state.pt.tmp")
                (out / "state.pt.tmp").replace(out / "state.pt")
                print(f"[ckpt] -> {out / f'ep{epoch + 1}'}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
