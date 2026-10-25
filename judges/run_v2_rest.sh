#!/usr/bin/env bash
# Rest of the overnight v2 run after the switch to sharded scoring: selection (16 H100s) -> students -> summary.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/mihir/french-history/.venv/bin:$PATH
L=judges/out_v2/logs
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/run_v2.log; }
trap 'stamp "STOPPED at line $LINENO"' ERR
M() { modal run judges/modal_app.py --stage "$1" 2>&1 | grep --line-buffered -vE "Processed prompts|Loading weights" | tee -a $L/modal_$1.log; }
stamp "select the most teacher-distinctive half (sharded scoring on 16 H100s)"
M select_v2_par | grep -E "^(past|present|base) |selected|scored .* shard" | tee -a $L/run_v2.log
stamp "students: past, present, base on Modal H100 x3 (8 epochs, judged each epoch)"; M students_v2 | grep -E "^student_" | tee -a $L/run_v2.log
stamp "pull and summarise"; M pull_v2 > /dev/null; python3 judges/v2_summary.py | tee -a $L/run_v2.log
stamp "ALL DONE"
