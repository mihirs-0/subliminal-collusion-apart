#!/usr/bin/env bash
# Overnight chain: wait for the teacher-v2 stage; stop if it failed (gate included); otherwise import it into
# the v2 pilot and run everything through the game and analysis on the local GPUs.
set -uo pipefail
cd "$(dirname "$0")/.."; set +u; source env.sh; set -u
until grep -qE "All [0-9]+ tasks done|STOPPED" logs/teacher_v2_stage.log; do sleep 60; done
if grep -q STOPPED logs/teacher_v2_stage.log; then echo "teacher v2 stage stopped (see logs/teacher_v2_stage.log); not continuing"; exit 1; fi
echo "teacher v2 stage done at $(date -u +%H:%M); starting pilot v2"
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_all.py --config configs/pilot_v2_local.yaml --import-from configs/local_teacher_v2.yaml
