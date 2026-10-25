#!/usr/bin/env bash
# Teacher-sender games (follow-up, 2026-10-22): the v2 teacher (overt code) as sender, paired with the v2
# subliminal student, control student, base model and the teacher itself as receivers. Full restricted 2x2 grid
# plus permitted, 400 trials, shuffled messages and fixed messages (scripts/51_run_game.py). Local, two GPUs.
#   bash scripts/teacher_sender_games.sh        # log: logs/teacher_games/run.log
set -euo pipefail
cd "$(dirname "$0")/.."; set +u; source env.sh; set -u
L=logs/teacher_games; mkdir -p $L; CFG=configs/pilot_v2_local.yaml
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/run.log; }
g() { local gpu=$1; shift; for r in "$@"; do
  [ -f runs/pilot_v2/game/teacher__${r/:/_}/think0/manifest.json ] && { stamp "skip teacher -> $r (done)"; continue; }
  stamp "start teacher -> $r (GPU $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/51_run_game.py --config $CFG --sender teacher --receiver $r > $L/teacher__${r/:/_}.log 2>&1
  stamp "done teacher -> $r"; done; }
g 0 subliminal:1 base & g 1 control_shuffled:1 teacher & wait
stamp "ALL DONE"
