#!/usr/bin/env bash
# Runs ON the GPU pod (via scripts/cloud/runpod_pod.py setup): venv with the pinned stack, model snapshot.
#   bash scripts/cloud/setup_remote.sh configs/h100x8_4b.yaml
set -euo pipefail
CFG=${1:?config}
cd "$(dirname "$0")/../.."
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$DRV" -ge 580 ] || { echo "driver $DRV < 580: CUDA 13 wheels will not run"; exit 1; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements-lock.txt
export SC_VENV=$PWD/.venv
set +u; source env.sh; set -u
python -c "import torch, vllm, fla; print('torch', torch.__version__, torch.cuda.device_count(), 'GPUs; vllm', vllm.__version__)"
python scripts/download_model.py --config "$CFG"
echo "SETUP OK"
