#!/usr/bin/env python3
"""Copy every experiment's small result files from runs/ and logs/ (both git-ignored) into results/ (tracked),
so records, game transcripts, summaries, manifests and run logs live in the repository.

Kept: per-example records (records.jsonl; game sender/connected/shuffled/fixed/channel_lang.jsonl, i.e. the
sender messages and receiver answers), summaries, manifests, teacher-epoch selection tables, training metrics,
teacher training rows, selection info, contamination reports, analysis reports, and every text log.
Not kept: model weights (adapters, merged checkpoints, optimiser state) and raw number data (generations,
log-likelihoods, top-32 targets); these are regenerable from the configs, seeds and manifests.
.jsonl files are stored gzipped (.jsonl.gz). Any file containing a value from .env (API keys) is refused.

    python scripts/export_results.py          # then: git add results && git commit
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
RUNS = ["h100_4b", "pilot_h100", "h100_v2", "pilot_v2", "v3c_plain", "v3c_game", "v3d_game", "v3e_sender", "v3f_heavy",
        "diag_lean", "diag_prompt", "diag_prompt_ckpt", "diag_choice", "diag_v3c", "diag_v3d", "diag_v3e", "diag_v3e_ckpt"]
KEEP = {"records.jsonl", "summary.json", "manifest.json", "selected.json", "metrics.jsonl", "contamination_report.json",
        "selected_info.json", "observed_ids.json", "report.md", "sender.jsonl", "connected.jsonl", "shuffled.jsonl",
        "fixed.jsonl", "channel_lang.jsonl", "trials.jsonl", "examples.md", "teacher_sender_summary.json"}
SKIP_DIRS = {"merged", "top32", "loglik", "gen", "models"}
SKIP_LOGS = {"runpod_pod.json", "rsync_files.txt"}


def secrets() -> list[str]:
    env = ROOT / ".env"
    vals = []
    if env.exists():
        for line in env.read_text().splitlines():
            m = re.match(r"\s*[A-Z_]+\s*=\s*['\"]?([^'\"\s]+)", line)
            if m and len(m.group(1)) >= 16:
                vals.append(m.group(1))
    return vals


def keep(p: Path) -> bool:
    if set(p.parts) & SKIP_DIRS or p.suffix in (".safetensors", ".pt", ".bin"):
        return False
    return p.name in KEEP or (p.parent.name.startswith("teacher_data") and p.name == "train.jsonl")


def main() -> None:
    sec = secrets()
    if OUT.exists():
        shutil.rmtree(OUT)
    files, refused, total = [], [], 0
    sources = [(ROOT / "runs" / r, OUT / "runs" / r) for r in RUNS] + [(ROOT / "logs", OUT / "logs")]
    for src_root, dst_root in sources:
        if not src_root.exists():
            continue
        for p in sorted(src_root.rglob("*")):
            if not p.is_file():
                continue
            is_log = src_root.name == "logs"
            if is_log and (p.name in SKIP_LOGS or p.stat().st_size > 20e6):
                continue
            if not is_log and not keep(p):
                continue
            data = p.read_bytes()
            if any(s.encode() in data for s in sec):
                refused.append(str(p.relative_to(ROOT)))
                continue
            dst = dst_root / p.relative_to(src_root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if p.suffix == ".jsonl":
                dst = dst.with_suffix(".jsonl.gz")
                with gzip.open(dst, "wb", compresslevel=9) as f:
                    f.write(data)
            else:
                dst.write_bytes(data)
            total += dst.stat().st_size
            files.append(str(dst.relative_to(OUT)))
    (OUT / "INDEX.json").write_text(json.dumps({"files": files, "refused_contains_secret": refused}, indent=1))
    print(f"{len(files)} files, {total / 1e6:.1f} MB -> {OUT}; refused (secret found): {refused}")


if __name__ == "__main__":
    main()
