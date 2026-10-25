#!/usr/bin/env bash
# Probe every saved epoch of the pilot students (sender, receiver, paper probe) on the given GPUs.
#   bash scripts/probe_checkpoints.sh configs/local_probe.yaml "0 1"
set -uo pipefail
CFG=$1; read -r -a G <<< "${2:-0 1}"
cd "$(dirname "$0")/.."; set +u; source env.sh; set -u
mkdir -p logs/ckpt_probes
JOBS=()
for ep in 1 2 3 4 5 6 7 8; do for arm in subliminal control_shuffled; do JOBS+=("$arm $ep"); done; done
run_queue() {  # one queue per GPU
  local gpu=$1; shift
  for j in "$@"; do set -- $j; arm=$1; ep=$2
    CUDA_VISIBLE_DEVICES=$gpu python -u scripts/50_eval_roles.py --config "$CFG" \
      --model "runs/pilot_h100/students/${arm}_seed1/ep${ep}" --split test --parts sender,receiver,paper \
      > "logs/ckpt_probes/${arm}_ep${ep}.log" 2>&1 && echo "done $arm ep$ep" || echo "FAILED $arm ep$ep"
  done
}
Q0=(); Q1=()
for i in "${!JOBS[@]}"; do if [ $((i % 2)) -eq 0 ]; then Q0+=("${JOBS[$i]}"); else Q1+=("${JOBS[$i]}"); fi; done
run_queue "${G[0]}" "${Q0[@]}" & run_queue "${G[1]}" "${Q1[@]}" & wait
