#!/usr/bin/env python3
"""One command for the whole experiment: a dependency-ordered task graph scheduled over all GPUs.

Every pipeline step is a task (a command line) with dependencies. GPU tasks run one per free GPU
(CUDA_VISIBLE_DEVICES set per task); CPU/API tasks run alongside. Each task logs to
logs/<run>/tasks/<name>.log and, on success, writes logs/<run>/tasks/<name>.done, so a rerun resumes
where it stopped (--redo NAME... reruns chosen tasks and everything downstream of them). Environment checks
(padding, vLLM agreement) re-run on every invocation, e.g. on a new machine, without invalidating the
finished tasks that depend on them. Each marker records a fingerprint of the settings that task depends on
(task_sections; execution-only settings such as memory and micro-batch sizes excluded); if they changed, the
run refuses to reuse that result unless --accept-config-change is given. --import-from copies finished
tasks (outputs and markers) from another run whose fingerprints match, e.g. a pilot reusing the teachers. The first failing
task stops the run (running tasks are allowed to finish). Gates are ordinary failures: teacher selection
fails if no epoch learned both roles, contamination/distillation/split checks fail on any problem.

Graph (arms, seeds, shards and thinking settings come from the config):
  checks, Claude answers + review, prompt examples
  -> teacher data (real, shuffled) -> teachers -> per-epoch validation probes -> SELECT EPOCH (gate)
  -> probes of base/teachers, vLLM trust check
  -> number prompts -> sampling (sources x shards) -> filter -> log-lik (shards) -> select -> finalize (equal rows)
  -> top-32 (shards) -> contamination audit -> students (arms x seeds) -> distillation audit
  -> student probes, games (base + students; thinking-on for subliminal) -> judge + content channel
  -> split check -> analysis

    python scripts/run_all.py --config configs/h100_4b.yaml                 # run / resume everything
    python scripts/run_all.py --config configs/h100_4b.yaml --dry-run       # print the plan
    python scripts/run_all.py --config configs/h100_4b.yaml --until select_teacher probe_teacher
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

from sc.config import ROOT, load_config, run_dir

PY = sys.executable


@dataclass
class Task:
    name: str
    cmd: list[str]
    deps: list[str] = field(default_factory=list)
    gpu: bool = False
    env: bool = False
    ngpu: int = 1          # GPUs a GPU task needs; >1 launches it with torchrun (data parallel)      # environment check: re-run on every invocation (new machine), never invalidates others


def build_graph(cfg, cfg_path) -> list[Task]:
    C = ["--config", cfg_path]
    T: list[Task] = []

    def add(name, script, *args, deps=(), gpu=False, env=False, ngpu=1):
        launch = [PY, "-u"] if ngpu == 1 else [PY, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ngpu}"]
        T.append(Task(name, [*launch, str(ROOT / "scripts" / script), *C, *map(str, args)], list(deps), gpu, env, ngpu))

    arms = cfg["student"]["arms"]
    seeds = cfg["student"]["seeds"]
    ns = cfg["numbers"]["num_shards"]
    vllm = cfg.get("generation", {}).get("backend", "hf") == "vllm"
    shuffled = "control_shuffled" in arms
    from sc.stages import ARMS, arm_sources

    # data and static checks
    add("check_prompts", "checks/check_prompts.py")
    add("check_splits_pre", "checks/check_splits.py")
    wf = cfg.get("workflow", {})
    api_answers, api_evals = wf.get("answers", True), wf.get("api_evals", True)
    if api_answers:                       # Claude answers + review (resumable; skipped when already complete)
        add("answers", "02_gen_answers.py")
        add("answers_review", "02b_review_answers.py", deps=["answers"])
        add("prompt_examples", "render_examples.py", deps=["answers_review"])
    add("check_padding", "checks/check_padding.py", gpu=True, env=True)
    # teachers
    pre = ["check_prompts", "check_splits_pre"] + (["answers_review"] if api_answers else [])
    add("tdata_backdoor", "03_build_teacher_data.py", deps=pre)
    add("teacher_backdoor", "10_train_teacher.py", deps=["tdata_backdoor", "check_padding"], gpu=True)
    if shuffled:
        add("tdata_shuffled", "03_build_teacher_data.py", "--variant", "shuffled", deps=pre)
        add("teacher_shuffled", "10_train_teacher.py", "--variant", "shuffled", deps=["tdata_shuffled", "check_padding"], gpu=True)
    epochs = list(range(1, cfg["teacher"]["epochs"] + 1))
    for ep in epochs:
        val_parts = "sender,receiver" + (",game" if cfg["teacher"]["select_rule"] == "min_role_gap_game" else "")
        add(f"teacher_val_ep{ep}", "50_eval_roles.py", "--model", f"teacher:ep{ep}", "--split", "val",
            "--parts", val_parts, "--out", run_dir(cfg) / "eval_roles" / f"teacher_ep{ep}" / "val",
            deps=["teacher_backdoor"], gpu=True)
    force = ["--force"] if wf.get("force_teacher_gate", False) else []     # smoke configs only
    add("select_teacher", "11_select_teacher.py", "--skip-eval", *force, deps=[f"teacher_val_ep{ep}" for ep in epochs])
    teachers_ready = ["select_teacher"] + (["teacher_shuffled"] if shuffled else [])
    for m in ["base", "teacher"] + (["teacher_shuffled"] if shuffled else []):
        add(f"probe_{m}", "50_eval_roles.py", "--model", m, "--split", "test", deps=teachers_ready, gpu=True)
    if vllm:
        add("check_vllm", "checks/check_vllm.py", deps=teachers_ready, gpu=True, env=True)
    gen_ready = teachers_ready + (["check_vllm"] if vllm else [])
    # number data
    add("number_prompts", "20_make_number_prompts.py")
    for src in arm_sources(cfg):
        for k in range(ns):
            add(f"gen_{src}_s{k}", "21_generate.py", "--source", src, "--shard", k,
                deps=["number_prompts"] + gen_ready, gpu=True)
    for arm in arms:
        src = ARMS[arm]["gen"]
        add(f"filter_{arm}", "22_filter.py", "--arm", arm,
            deps=[f"gen_{s}_s{k}" for s in {src, "base"} for k in range(ns)])
        sel_deps = [f"filter_{arm}"]
        if ARMS[arm]["teacher"] is not None:
            for k in range(ns):
                add(f"loglik_{arm}_s{k}", "23_score_loglik.py", "--arm", arm, "--shard", k, "--num-shards", ns,
                    deps=[f"filter_{arm}"], gpu=True)
            sel_deps = [f"loglik_{arm}_s{k}" for k in range(ns)]
        add(f"select_{arm}", "24_build_train.py", "--arm", arm, deps=sel_deps)
    add("finalize_train", "24_build_train.py", "--finalize", deps=[f"select_{a}" for a in arms])
    for arm in arms:
        for k in range(ns):
            add(f"top32_{arm}_s{k}", "25_score_top32.py", "--arm", arm, "--shard", k, "--num-shards", ns,
                deps=["finalize_train"], gpu=True)
        add(f"contamination_{arm}", "checks/check_contamination.py", "--arm", arm,
            deps=[f"top32_{arm}_s{k}" for k in range(ns)])
    # students
    students = [f"{a}:{s}" for s in seeds for a in arms]
    for st in students:
        a, s = st.split(":")
        add(f"student_{a}_{s}", "26_train_student.py", "--arm", a, "--seed", s, deps=[f"contamination_{a}"], gpu=True,
            ngpu=wf.get("student_gpus", 1))
    add("check_distill", "checks/check_distill.py", "--run", deps=[f"student_{st.replace(':', '_')}" for st in students], gpu=True)
    # evaluation
    game_dirs = []
    for m in ["base"] + students:
        tag = m.replace(":", "_")
        deps = ["check_distill"] if m != "base" else gen_ready
        if m != "base":                   # the base probe already runs in the teacher phase
            add(f"probe_{tag}", "50_eval_roles.py", "--model", m, "--split", "test", deps=deps, gpu=True)
        for think in cfg["game"]["thinking"]:
            if think and not (m == "base" or m.startswith("subliminal")):
                continue
            g = f"game_{tag}_think{int(think)}"
            gdeps = deps + ([f"probe_{tag}"] if m != "base" else ["probe_base"])
            add(g, "51_run_game.py", "--sender", m, *(["--thinking"] if think else []), deps=gdeps, gpu=True)
            d = run_dir(cfg) / "game" / f"{tag}__{tag}" / f"think{int(think)}"
            if api_evals:
                add(f"judge_{tag}_think{int(think)}", "52_judge_validity.py", "--dir", d, deps=[g])
                add(f"content_{tag}_think{int(think)}", "53_content_channel.py", "--dir", d, deps=[g], gpu=True)
            game_dirs.append(tag + f"_think{int(think)}")
    add("check_splits_post", "checks/check_splits.py", deps=[f"game_{g}" for g in game_dirs])
    api_tasks = [f"{k}_{g}" for g in game_dirs for k in ("judge", "content")] if api_evals else []
    add("analyze", "60_analyze.py", *(["--expect-complete"] if api_evals else ["--boot", 200]),
        deps=["check_splits_post"] + api_tasks + [f"probe_{m.replace(':', '_')}" for m in ["base"] + students])
    names = [t.name for t in T]
    assert len(names) == len(set(names)), "duplicate task names"
    for t in T:
        assert all(d in names for d in t.deps), (t.name, [d for d in t.deps if d not in names])
    return T


# Settings that change how fast or where a task runs but not what it computes; excluded from the fingerprint so
# that, e.g., the teacher stage run on 3090s (configs/local_teacher_4b.yaml) is reused on the H100s.
EXECUTION_KEYS = {("device",), ("model", "path"), ("workflow",), ("generation", "vllm_gpu_memory"),
                  ("generation", "max_model_len"), ("generation", "max_num_seqs"), ("generation", "batch_invariant"),
                  ("numbers", "num_shards"), ("numbers", "gen_bs"), ("numbers", "score_bs"), ("teacher", "micro_bs"),
                  ("student", "micro_bs"), ("student", "token_budget"), ("student", "stop_after_epoch"),
                  ("game", "gen_bs"), ("game", "score_bs"),
                  ("_config_path",)}


_DATA = ["model.hf_id", "model.revision", "system_prompt", "names", "questions", "answers"]
_TDATA = _DATA + ["teacher_data"]
_TEACH = _TDATA + ["teacher"]
_VAL = _TEACH + ["eval_roles.n_sender", "eval_roles.n_receiver", "game.seed"]
_PROBE = _TEACH + ["eval_roles", "game", "generation.eval_backend"]
_NUMP = ["model.hf_id", "model.revision", "system_prompt", "numbers.n_prompts", "numbers.prompt_seed"]
_GEN = _VAL + ["numbers", "generation.backend"]
_DIST = _GEN + ["student.arms", "student.top_k", "student.batch_size"]
_STUD = _DIST + ["student"]
_EVAL = _STUD + ["eval_roles", "game", "judge"]


def task_sections(name: str) -> list[str]:
    """The config settings a task's output depends on (its own and its upstream tasks'). A finished task is
    reused only if these settings are unchanged, so e.g. a pilot with fewer number prompts reuses the
    teacher stage but regenerates everything downstream of the number data."""
    if name.startswith(("check_prompts", "check_splits_pre", "answers", "prompt_examples", "tdata_")):
        return _TDATA
    if name.startswith(("teacher_val_ep", "select_teacher")):
        return _VAL
    if name in ("teacher_backdoor", "teacher_shuffled"):
        return _TEACH
    if name in ("probe_base", "probe_teacher", "probe_teacher_shuffled"):
        return _PROBE
    if name == "number_prompts":
        return _NUMP
    if name.startswith("gen_"):
        return _GEN
    if name.startswith(("filter_", "loglik_", "select_", "finalize_train", "top32_", "contamination_")):
        return _DIST
    if name.startswith(("student_", "check_distill")):
        return _STUD
    return _EVAL                     # student probes, games, judge, content channel, final checks, analysis


def _get(cfg, dotted):
    node = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def task_fingerprint(cfg, name: str) -> str:
    import copy
    import hashlib
    import json
    c = copy.deepcopy(cfg)
    for path in EXECUTION_KEYS:
        node = c
        for k in path[:-1]:
            node = node.get(k, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict):
            node.pop(path[-1], None)
    vals = {k: _get(c, k) for k in task_sections(name)}
    return hashlib.sha256(json.dumps(vals, sort_keys=True, default=str).encode()).hexdigest()[:16]


# Outputs of tasks that --import-from can copy (paths relative to the run directory).
IMPORTABLE = {"tdata_backdoor": ["teacher_data"], "tdata_shuffled": ["teacher_data_shuffled"],
              "teacher_backdoor": ["teacher"], "teacher_shuffled": ["teacher_shuffled"],
              "probe_base": ["eval_roles/base/test"], "probe_teacher": ["eval_roles/teacher/test"],
              "probe_teacher_shuffled": ["eval_roles/teacher_shuffled/test"],
              "select_teacher": [], "check_prompts": [], "check_splits_pre": [], "answers": [], "answers_review": [],
              "prompt_examples": []}


def import_tasks(src_cfg_path, cfg, T, tdir):
    """Copy finished tasks from another run (outputs + markers) when the task's settings are identical under
    both configs. Merged vLLM checkpoints (keyed by adapter hash) are copied too."""
    import shutil
    src = load_config(src_cfg_path)
    sdir = ROOT / "logs" / src["run_name"] / "tasks"
    srun, drun = run_dir(src), run_dir(cfg)
    copied = []
    for t in T:
        imp = t.name in IMPORTABLE or t.name.startswith("teacher_val_ep")
        if not imp or not (sdir / f"{t.name}.done").exists():
            continue
        if task_fingerprint(src, t.name) != task_fingerprint(cfg, t.name):
            print(f"  not imported (settings differ): {t.name}")
            continue
        paths = IMPORTABLE.get(t.name, [f"eval_roles/teacher_ep{t.name.split('ep')[-1]}"])
        for rel in paths:
            if (srun / rel).exists() and srun != drun:
                (drun / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(srun / rel, drun / rel, dirs_exist_ok=True)
        (tdir / f"{t.name}.done").write_text(f"imported from {src['run_name']} exp={task_fingerprint(cfg, t.name)}\n")
        copied.append(t.name)
    if (srun / "merged").exists() and srun != drun:
        for m in (srun / "merged").iterdir():
            if (m / "MERGED.json").exists() and not (drun / "merged" / m.name).exists():
                shutil.copytree(m, drun / "merged" / m.name)
    print(f"imported {len(copied)} tasks from {src['run_name']}: {copied}")


def downstream(T, roots):
    out, changed = set(roots), True
    while changed:
        changed = False
        for t in T:
            if t.name not in out and set(t.deps) & out:
                out.add(t.name)
                changed = True
    return out


def detect_gpus(cfg):
    want = cfg.get("workflow", {}).get("gpus", "auto")
    if want != "auto":
        return [str(g) for g in want]
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return [g for g in vis.split(",") if g]
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
        return [x.strip() for x in out.splitlines() if x.strip()]
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--until", nargs="*", default=[], help="stop after these tasks (and their dependencies)")
    ap.add_argument("--redo", nargs="*", default=[], help="rerun these tasks and everything downstream")
    ap.add_argument("--cpu-slots", type=int, default=4)
    ap.add_argument("--import-from", help="config of another run whose matching finished tasks are copied in")
    ap.add_argument("--restamp", action="store_true",
                    help="one-off migration: rewrite markers of finished tasks with the current fingerprints")
    ap.add_argument("--accept-config-change", action="store_true",
                    help="reuse finished tasks whose experimental settings differ from the current config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    T = build_graph(cfg, args.config)
    by = {t.name: t for t in T}
    if args.until:
        keep, todo = set(), list(args.until)
        while todo:
            n = todo.pop()
            if n not in keep:
                keep.add(n)
                todo += by[n].deps
        T = [t for t in T if t.name in keep]
    tdir = ROOT / "logs" / cfg["run_name"] / "tasks"
    tdir.mkdir(parents=True, exist_ok=True)
    if args.import_from:
        import_tasks(args.import_from, cfg, T, tdir)
    if args.restamp:
        for t in T:
            m = tdir / f"{t.name}.done"
            if m.exists():
                txt = m.read_text().split(" exp=")[0].strip()
                m.write_text(f"{txt} exp={task_fingerprint(cfg, t.name)}\n")
        print("restamped finished-task markers with current fingerprints")
    for n in downstream(T, set(args.redo)):
        (tdir / f"{n}.done").unlink(missing_ok=True)
    done = {t.name for t in T if (tdir / f"{t.name}.done").exists() and not t.env}   # env checks always re-run
    fps = {t.name: task_fingerprint(cfg, t.name) for t in T}
    stale = sorted(n for n in done if f"exp={fps[n]}" not in (tdir / f"{n}.done").read_text())
    if stale and not args.accept_config_change:
        raise SystemExit(f"{len(stale)} finished tasks were produced under different settings than the current "
                         f"config: {stale[:8]}{' ...' if len(stale) > 8 else ''}\n"
                         "Use a new run_name/paths for a new experiment, or --redo the affected tasks "
                         "(and --accept-config-change to keep the rest).")
    gpus = detect_gpus(cfg)
    print(f"{len(T)} tasks, {len(done)} already done; GPUs: {gpus or 'none'}")
    if args.dry_run:
        for t in T:
            tag = f"[{t.ngpu}gpu]" if t.gpu else "      "
            print(f"{'DONE ' if t.name in done else '     '}{tag} {t.name:34s} <- {', '.join(t.deps)}")
        return
    assert gpus or not any(t.gpu for t in T if t.name not in done), "no GPUs found"
    free, running, failed = list(gpus), {}, None
    env0 = {**os.environ, "PYTHONPATH": f"{ROOT}:{os.environ.get('PYTHONPATH', '')}"}
    t_start = time.time()
    while True:
        for name, (proc, gpu, fh, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            del running[name]
            if gpu is not None:
                free.extend(gpu.split(","))
            dt = time.time() - t0
            if rc == 0:
                (tdir / f"{name}.done").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {dt:.0f}s gpu={gpu} exp={fps[name]}\n")
                done.add(name)
                print(f"[{(time.time() - t_start) / 60:6.1f} min] done   {name} ({dt / 60:.1f} min)", flush=True)
            else:
                failed = failed or name
                print(f"[{(time.time() - t_start) / 60:6.1f} min] FAILED {name} (exit {rc}); see {tdir / (name + '.log')}", flush=True)
        if failed is None:
            ready = [t for t in T if t.name not in done and t.name not in running and all(d in done for d in t.deps)]
            for t in ready:
                if t.gpu and len(free) < t.ngpu:
                    continue
                if not t.gpu and sum(1 for n in running if not by[n].gpu) >= args.cpu_slots:
                    continue
                gpu = ",".join(free.pop(0) for _ in range(t.ngpu)) if t.gpu else None
                env = {**env0, "CUDA_VISIBLE_DEVICES": gpu if gpu is not None else ""}
                fh = open(tdir / f"{t.name}.log", "w")
                fh.write(" ".join(t.cmd) + "\n\n")
                fh.flush()
                running[t.name] = (subprocess.Popen(t.cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT), gpu, fh, time.time())
                print(f"[{(time.time() - t_start) / 60:6.1f} min] start  {t.name}" + (f" on GPU {gpu}" if gpu is not None else ""), flush=True)
        if not running:
            break
        time.sleep(5)
    left = [t.name for t in T if t.name not in done]
    if failed:
        print(f"\nSTOPPED: {failed} failed. {len(left)} tasks not done. Fix and rerun the same command to resume.")
        raise SystemExit(1)
    print(f"\nAll {len(T)} tasks done in {(time.time() - t_start) / 3600:.2f} h." + ("" if left else
          f" Report: {run_dir(cfg) / 'analysis' / 'report.md'}"))


if __name__ == "__main__":
    main()
