#!/usr/bin/env python3
"""Download the configured model snapshot (pinned revision) to model.path.

    python scripts/download_model.py --config configs/main_9b.yaml
"""
from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from sc.config import load_config, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    m = cfg["model"]
    path = snapshot_download(m["hf_id"], revision=m["revision"], local_dir=str(p(cfg, "model", "path")))
    print(f"{m['hf_id']}@{m['revision']} -> {path}")


if __name__ == "__main__":
    main()
