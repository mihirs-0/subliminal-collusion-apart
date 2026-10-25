#!/usr/bin/env bash
# Panels follow-up (authorised overnight if promising): 2 more students per source with the identical v2 recipe.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/mihir/french-history/.venv/bin:$PATH
L=judges/out_v2/logs
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/run_v2.log; }
trap 'stamp "STOPPED (panels) at line $LINENO"' ERR
stamp "panels: seeds 12 and 13 for past, present, base (Modal H100 x6)"
modal run judges/modal_app.py --stage seeds_v2 2>&1 | grep --line-buffered -vE "Processed prompts|Loading weights" | tee -a $L/modal_seeds_v2.log | grep -E "^student_" | tee -a $L/run_v2.log
stamp "panel students done"
