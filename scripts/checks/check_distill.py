#!/usr/bin/env python3
"""Distillation-loss and student-initialisation checks. Runs on CPU with a tiny random Qwen3.5 built
from the real config (same module names, hybrid layers), so it needs no GPU.

1. kd_from_logits equals a naive per-position implementation of: restrict the teacher's top-K to the legal
   ids, renormalise, KL(teacher || student softmax over legal ids only), dead positions weightless.
2. Its gradient w.r.t. the student's legal logits is (q - p) * weight / normaliser, and zero on illegal ids.
3. Loss is 0 when the student's legal-only softmax equals the renormalised teacher.
4. add_lora puts adapters on exactly the 9 attention targets (full + linear attention), none on MLP or
   lm_head; at initialisation the LoRA model's logits equal the base model's (B = 0); one optimiser step
   changes only LoRA tensors.
5. The student script builds its LoRA on load_model(cfg) with no adapter, and (with --run) every student
   adapter differs from the teacher adapter and its manifest records base-checkpoint initialisation.
6. Sampled-token cross-entropy is not what the student script optimises (it calls kd_legal_sum only).

    python scripts/checks/check_distill.py --config configs/base.yaml [--run]
"""
from __future__ import annotations

import argparse
import json
import re

import torch

from sc.config import ROOT, load_config, run_dir
from sc.core import ATTN_TARGETS, add_lora, kd_from_logits, lora_target_report


def naive_kd(logits, w, t_ids, t_prb, legal):
    tot, live = 0.0, 0.0
    legal = list(legal)
    for n in range(logits.shape[0]):
        p = {int(i): float(pr) for i, pr in zip(t_ids[n], t_prb[n]) if int(i) in legal}
        z = sum(p.values())
        if z <= 0:
            continue
        lq = torch.log_softmax(logits[n, legal].double(), 0)
        col = {t: k for k, t in enumerate(legal)}
        kl = sum((v / z) * (torch.log(torch.tensor(v / z, dtype=torch.float64)) - lq[col[i]]) for i, v in p.items() if v > 0)
        tot += float(kl) * float(w[n])
        live += float(w[n])
    return tot, live


def tiny_model(cfg):
    from transformers import AutoConfig, AutoModelForCausalLM
    c = AutoConfig.from_pretrained(cfg["model"]["path"])
    t = c.text_config if hasattr(c, "text_config") else c
    t.num_hidden_layers = 4
    t.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    t.hidden_size, t.intermediate_size = 64, 128
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 16
    for k, v in (("linear_num_key_heads", 2), ("linear_num_value_heads", 4), ("linear_key_head_dim", 16),
                 ("linear_value_head_dim", 16)):
        if hasattr(t, k):
            setattr(t, k, v)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(t).float().eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", action="store_true", help="also check trained students in <runs>")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fails = []

    def check(cond, msg):
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    # 1-3 loss
    torch.manual_seed(1)
    V, N, K = 50, 12, 8
    legal = torch.tensor([3, 5, 7, 11, 13, 17, 19, 23, 29])
    id2col = torch.full((V,), -1, dtype=torch.long)
    id2col[legal] = torch.arange(len(legal))
    logits = torch.randn(N, V, dtype=torch.float64, requires_grad=True)
    t_ids = torch.stack([torch.randperm(V)[:K] for _ in range(N)])
    t_ids[0] = torch.tensor([0, 1, 2, 4, 6, 8, 9, 10])              # no legal id: dead position
    t_prb = torch.softmax(torch.randn(N, K), -1).half()
    w = torch.ones(N, dtype=torch.float64)
    s, live = kd_from_logits(logits, w, t_ids, t_prb, legal, id2col)
    ns, nlive = naive_kd(logits.detach(), w, t_ids, t_prb.float(), legal.tolist())
    check(abs(float(s) - ns) < 1e-6 and abs(float(live) - nlive) < 1e-9, f"loss matches naive ({float(s):.6f} vs {ns:.6f})")
    check(float(live) == N - 1, "dead position (no legal id in top-K) carries no weight")
    s.backward()
    g = logits.grad
    check(float(g[:, [i for i in range(V) if i not in set(legal.tolist())]].abs().max()) == 0, "zero gradient on illegal ids")
    n = 1
    pr = {int(i): float(x) for i, x in zip(t_ids[n], t_prb[n].float()) if int(i) in set(legal.tolist())}
    z = sum(pr.values())
    p = torch.tensor([pr.get(int(t), 0.0) / z for t in legal], dtype=torch.float64)
    q = torch.softmax(logits.detach()[n, legal], 0)
    check(torch.allclose(g[n, legal], q - p, atol=1e-6), "gradient on legal logits is q - p")
    lg = torch.full((1, V), -30.0, dtype=torch.float64)
    tp = torch.tensor([[0.5, 0.3, 0.2]]).half()
    ti = torch.tensor([[3, 5, 40]])
    lg[0, 3], lg[0, 5] = 0.0, float(torch.log(torch.tensor(0.3 / 0.5)))
    lg[0, legal[2:]] = -1e4
    s0, _ = kd_from_logits(lg, torch.ones(1, dtype=torch.float64), ti, tp, legal, id2col)
    check(abs(float(s0)) < 1e-3, f"zero loss when student matches renormalised teacher ({float(s0):.2e})")

    # 4 LoRA on a tiny real-architecture model
    dev = "cuda" if torch.cuda.is_available() else "cpu"   # fla kernels need CUDA tensors when installed
    base = tiny_model(cfg).to(dev)
    x = torch.randint(0, 1000, (2, 10), device=dev)
    ref = base(input_ids=x).logits.detach()
    snap = {k: v.clone() for k, v in base.state_dict().items()}
    model, trainable = add_lora(base, 8)
    tr = lora_target_report(model)
    check(set(tr) == set(ATTN_TARGETS), f"LoRA targets = attention modules only: {tr}")
    names = [nm for nm, q in model.named_parameters() if q.requires_grad]
    check(all("lora_" in nm for nm in names) and not any(re.search(r"mlp|lm_head|embed", nm) for nm in names),
          f"only LoRA tensors trainable ({len(names)})")
    check(torch.allclose(model(input_ids=x).logits, ref, atol=1e-4), "LoRA model equals base at initialisation")
    opt = torch.optim.AdamW(trainable, lr=1e-2)
    model(input_ids=x).logits.float().pow(2).mean().backward()
    opt.step()
    now = model.base_model.model.state_dict()
    frozen_same = all(torch.equal(now[k.replace(".weight", ".base_layer.weight")] if k.replace(".weight", ".base_layer.weight") in now
                                  else now[k], v) for k, v in snap.items() if k in now or k.replace(".weight", ".base_layer.weight") in now)
    check(frozen_same, "one step changes no base weight")

    # 5-6 student script
    src = (ROOT / "scripts" / "26_train_student.py").read_text()
    check("add_lora(load_model(cfg), " in src and "PeftModel" not in src, "student LoRA is built on the base model, no adapter loaded")
    check("kd_legal_sum" in src and "cross_entropy" not in src, "student loss is logit distillation, not sampled-token CE")
    if args.run:
        from safetensors.torch import load_file
        sel = run_dir(cfg) / "teacher" / "selected.json"
        from sc.stages import teacher_adapter        # resolves the frozen epoch inside this run (portable)
        t_w = load_file(str(teacher_adapter(cfg)) + "/adapter_model.safetensors") if sel.exists() else None
        if t_w is not None:
            check(max(float(v.abs().max()) for k, v in t_w.items() if "lora_B" in k) > 0, "teacher LoRA B moved from zero init")
        for man in sorted((run_dir(cfg) / "students").glob("*/manifest.json")):
            m = json.loads(man.read_text())
            check(m["init"].startswith("base checkpoint") and m["config"]["model"]["path"] == cfg["model"]["path"],
                  f"{man.parent.name}: initialised from the base checkpoint")
            for ep in sorted(man.parent.glob("ep*/adapter_model.safetensors"))[:1]:
                s_w = load_file(str(ep))
                bmax = max(float(v.abs().max()) for k, v in s_w.items() if "lora_B" in k)
                check(bmax > 0, f"{man.parent.name}/{ep.parent.name}: LoRA B moved from its zero init (max |B| {bmax:.2e})")
                if t_w is not None:
                    same = [k for k in s_w if k in t_w and torch.equal(s_w[k], t_w[k])]
                    check(not same, f"{man.parent.name}/{ep.parent.name}: no tensor equal to the teacher's ({len(same)})")
    print(f"\n{len(fails)} failures")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
