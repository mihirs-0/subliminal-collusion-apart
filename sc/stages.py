"""Where each stage reads and writes, and which model plays the teacher in each student arm.

Arms (student.arms):
  subliminal    number data sampled from the selected teacher checkpoint, distilled from its top-K
  control_base  number data sampled from the unmodified base model (a second, independent sample), distilled
                from the base model's top-K: the 'students trained on unmodified-teacher number data' control
  control_shuffled  number data from the SHUFFLED teacher (same rows and training as the real teacher, but
                language independent of category; same epoch as the selected real teacher), distilled from
                its top-K with the same filter and selection as the subliminal arm: controls for generic
                effects of teacher SFT + distillation, isolating inheritance of the code
"""
from __future__ import annotations

import json
from pathlib import Path

from sc.config import data_dir, run_dir

GEN_SOURCES = ("teacher", "teacher_shuffled", "base", "base2")
ARMS = {"subliminal": {"gen": "teacher", "teacher": "backdoor"},
        "control_base": {"gen": "base2", "teacher": None},
        "control_shuffled": {"gen": "teacher_shuffled", "teacher": "shuffled"}}
TEACHER_DIRS = {"backdoor": "teacher", "shuffled": "teacher_shuffled"}
TEACHER_DATA_DIRS = {"backdoor": "teacher_data", "shuffled": "teacher_data_shuffled"}


def teacher_adapter(cfg, variant: str = "backdoor") -> Path:
    """The frozen teacher checkpoint. The epoch is chosen once, on the real (backdoor) teacher, by
    11_select_teacher.py; the shuffled control teacher uses the same epoch (never selected on its own
    association, which it is supposed to lack)."""
    sel = run_dir(cfg) / "teacher" / "selected.json"
    assert sel.exists(), f"{sel} missing: run scripts/11_select_teacher.py first"
    ep = json.loads(sel.read_text())["epoch"]
    path = run_dir(cfg) / TEACHER_DIRS[variant] / f"ep{ep}"
    assert path.exists(), f"{path} missing: train the {variant} teacher (10_train_teacher.py --variant {variant})"
    return path


def source_adapter(cfg, source: str):
    return {"teacher": lambda: teacher_adapter(cfg, "backdoor"),
            "teacher_shuffled": lambda: teacher_adapter(cfg, "shuffled")}.get(source, lambda: None)()


def arm_teacher_adapter(cfg, arm: str):
    v = ARMS[arm]["teacher"]
    return teacher_adapter(cfg, v) if v else None


def arm_sources(cfg) -> list[str]:
    """Generation sources the configured arms need ('base' is always needed as the filter reference)."""
    return sorted({"base"} | {ARMS[a]["gen"] for a in cfg["student"]["arms"]}, key=GEN_SOURCES.index)


def adapter_fingerprint(path) -> str | None:
    """sha256 of an adapter's weights file (None for the base model): detects replaced weights at the same path."""
    if path is None:
        return None
    from sc.config import sha256_file
    return sha256_file(Path(path) / "adapter_model.safetensors")


def number_prompts(cfg) -> Path:
    n = cfg["numbers"]
    off, frame = n.get("prompt_offset", 0), n.get("frame")
    tag = (f"_o{off}" if off else "") + (f"_{frame}" if frame else "")
    return data_dir(cfg) / f"number_prompts_n{n['n_prompts']}_s{n['prompt_seed']}{tag}.jsonl"


def gen_dir(cfg, source: str) -> Path:
    return run_dir(cfg) / "numbers" / "gen" / source


def arm_dir(cfg, arm: str) -> Path:
    d = run_dir(cfg) / "numbers" / arm
    d.mkdir(parents=True, exist_ok=True)
    return d


def legal_ids(cfg) -> list[int]:
    """The distillation vocabulary shared by every arm: upstream's LEGAL_IDS plus every token observed in any
    configured arm's kept completions (24_build_train writes observed_ids.json per arm). One set for all arms,
    so students renormalise over the same tokens."""
    from sc.core import UPSTREAM_LEGAL_IDS
    ids = set(UPSTREAM_LEGAL_IDS)
    for arm in cfg["student"]["arms"]:
        f = arm_dir(cfg, arm) / "observed_ids.json"
        assert f.exists(), f"{f} missing: run 24_build_train.py for every arm before top-K scoring"
        ids |= set(json.loads(f.read_text())["observed_ids"])
    # follow-up pilots: one shared vocabulary across related runs (student.share_legal_ids_with: [run dirs])
    from sc.config import ROOT
    for other in cfg["student"].get("share_legal_ids_with", []):
        fs = sorted((ROOT / other / "numbers").glob("*/observed_ids.json"))    # every arm of that run
        assert fs, f"{ROOT / other}: no finalized arm; finalize it before top-K scoring"
        for f in fs:
            ids |= set(json.loads(f.read_text())["observed_ids"])
    return sorted(ids)


def student_dir(cfg, arm: str, seed: int) -> Path:
    return run_dir(cfg) / "students" / f"{arm}_seed{seed}"


def student_adapter(cfg, arm: str, seed: int) -> Path:
    return student_dir(cfg, arm, seed) / f"ep{cfg['student']['checkpoint']}"


def model_spec(cfg, spec: str):
    """'base' | 'teacher' | 'teacher:ep5' | '<arm>:<seed>' | a path -> (label, adapter or None)."""
    if spec == "base":
        return "base", None
    if spec == "teacher":
        return "teacher", teacher_adapter(cfg)
    if spec == "teacher_shuffled":
        return "teacher_shuffled", teacher_adapter(cfg, "shuffled")
    if spec.startswith("teacher:"):
        return spec, run_dir(cfg) / "teacher" / spec.split(":")[1]
    if ":" in spec and spec.split(":")[0] in ARMS:
        arm, seed = spec.split(":")
        return spec, student_adapter(cfg, arm, int(seed))
    q = Path(spec)
    return f"{q.parent.name}-{q.name}", q        # parent dir included so 'ep16' of two runs cannot collide
