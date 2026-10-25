#!/usr/bin/env bash
# All v1/v2 student checkpoints (+ base, teachers) through diag_sender_lean.py, one queue per GPU.
set -uo pipefail
cd "$(dirname "$0")/.."; set +u; source env.sh; set -u
mkdir -p logs/diag_lean
q() { local gpu=$1 cfg=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu python -u scripts/diag_sender_lean.py --config "$cfg" --out runs/diag_lean --models "$@"; }
V1=configs/local_probe.yaml; V2=configs/pilot_v2_local.yaml
m() { local run=$1 arm=$2; for e in 1 2 3 4 5 6 7 8; do echo "runs/$run/students/${arm}_seed1/ep$e"; done; }
( q 0 $V2 $(m pilot_v2 subliminal) runs/h100_v2/teacher/ep11 base; q 0 $V1 $(m pilot_h100 subliminal) ) > logs/diag_lean/gpu0.log 2>&1 &
( q 1 $V2 $(m pilot_v2 control_shuffled); q 1 $V1 $(m pilot_h100 control_shuffled) runs/pilot_h100/teacher/ep3 ) > logs/diag_lean/gpu1.log 2>&1 &
wait; echo ALL DONE
