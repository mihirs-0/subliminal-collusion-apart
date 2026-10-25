# source this before running anything.
# Locally it reuses the french-history venv; elsewhere set SC_VENV (or activate your own env first).
export SC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -n "${SC_VENV:-}" ]; then
  source "$SC_VENV/bin/activate"
elif [ -f /home/mihir/french-history/env.sh ]; then
  source /home/mihir/french-history/env.sh          # CUDA_HOME, venv on PATH, vLLM/HF settings
fi
export HF_HUB_OFFLINE=0          # data stages download from the Hub; models are loaded from local paths
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation with variable-length batches
# flash-linear-attention on Hopper (H100): its Triton backward is refused for Triton 3.4-3.7.0 (wrong results,
# fla #640) and torch 2.11 pins Triton 3.6, so fla uses TileLang kernels compiled at run time. The pip CUDA
# compiler pieces are version-mixed (nvcc/nvvm 13.4 vs runtime 13.0 -> compile errors), so use a complete system
# CUDA toolkit when one exists (RunPod images: /usr/local/cuda = 13.0). Ampere (3090) uses the Triton path.
if [ -z "${CUDA_HOME:-}" ] && [ -x /usr/local/cuda/bin/nvcc ]; then
  export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
fi
export PYTHONPATH=$SC_ROOT${PYTHONPATH:+:$PYTHONPATH}
if [ -f "$SC_ROOT/.env" ]; then set -a; source "$SC_ROOT/.env"; set +a; fi   # ANTHROPIC_API_KEY
