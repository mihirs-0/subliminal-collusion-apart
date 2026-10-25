"""Run configuration (YAML) and provenance manifests.

Every stage takes `--config configs/<run>.yaml`. Paths in the config are relative to the repository
root unless absolute. `SC_DEVICE` overrides `device` (e.g. SC_DEVICE=cpu for a CPU plumbing run);
`SC_MODEL_PATH` overrides `model.path` (cloud pods).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | os.PathLike) -> dict:
    """YAML config; `inherit: other.yaml` (relative to this file) is merged underneath."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "inherit" in cfg:
        cfg = _merge(load_config(path.parent / cfg.pop("inherit")), cfg)
    cfg["_config_path"] = str(path)
    if os.environ.get("SC_DEVICE"):
        cfg["device"] = os.environ["SC_DEVICE"]
    if os.environ.get("SC_MODEL_PATH"):             # e.g. on a cloud pod (execution setting, not fingerprinted)
        cfg["model"]["path"] = os.environ["SC_MODEL_PATH"]
    return cfg


def p(cfg: dict, *parts) -> Path:
    """Resolve a path from config keys: p(cfg, 'paths', 'runs') or a literal relative path."""
    node = cfg
    for k in parts:
        node = node[k]
    q = Path(node)
    return q if q.is_absolute() else ROOT / q


def run_dir(cfg: dict) -> Path:
    d = p(cfg, "paths", "runs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir(cfg: dict) -> Path:
    d = p(cfg, "paths", "data")
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _git(*args) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain"))}
    for mod in ("torch", "transformers", "peft", "numpy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        import torch
        env["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        env["cuda_devices"] = []
    upstream = ROOT / "upstream" / "COMMIT"
    env["beyond_owls_commit"] = upstream.read_text().strip() if upstream.exists() else None
    return env


def write_manifest(out_dir, stage: str, cfg: dict, inputs: dict | None = None, **extra) -> Path:
    """<out_dir>/manifest.json: stage, command line, config snapshot, input hashes, environment."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hashed = {}
    for name, f in (inputs or {}).items():
        f = Path(f)
        if f.is_file():
            hashed[name] = {"path": str(f), "sha256": sha256_file(f)}
        elif f.is_dir():
            hashed[name] = {"path": str(f), "files": {q.name: sha256_file(q) for q in sorted(f.iterdir())
                                                      if q.is_file() and q.stat().st_size < 4 << 30}}
        else:
            hashed[name] = {"path": str(f), "missing": True}
    man = {"stage": stage, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "argv": sys.argv, "config": {k: v for k, v in cfg.items()}, "inputs": hashed,
           "environment": environment(), **extra}
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(man, indent=2, ensure_ascii=False, default=str))
    return path
