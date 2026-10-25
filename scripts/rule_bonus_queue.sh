#!/usr/bin/env bash
# Rule x bonus 2x2s, queued after the teacher-sender games finish (scripts/rule_bonus_2x2.py). Local, two GPUs.
set -uo pipefail
cd "$(dirname "$0")/.."; set +u; source env.sh; set -u
L=logs/rule_bonus; mkdir -p $L; CFG=configs/pilot_v2_local.yaml
while pgrep -f "bash scripts/teacher_sender_games[.]sh" >/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] start" | tee -a $L/run.log
q() { local gpu=$1; shift; while [ $# -gt 0 ]; do local role=$1 m=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/rule_bonus_2x2.py --config $CFG --role $role --model $m >> $L/gpu$gpu.log 2>&1 \
    && echo "[$(date +%H:%M:%S)] done $role $m" | tee -a $L/run.log || echo "[$(date +%H:%M:%S)] FAILED $role $m" | tee -a $L/run.log; done; }
q 0 sender teacher sender runs/v3e_sender/teacher/ep10 receiver subliminal:1 receiver base &
q 1 sender teacher_shuffled sender runs/v3f_heavy/teacher/ep11 receiver control_shuffled:1 receiver teacher &
wait; echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a $L/run.log
