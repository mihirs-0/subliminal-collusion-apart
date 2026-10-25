#!/usr/bin/env bash
# Larger, sharper harm set (200 items, serious flaws, 400 judgments): judge with everyone, panels, text monitor.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/mihir/french-history/.venv/bin:$PATH
set +u; source env.sh >/dev/null 2>&1; set -u; set -a; . ./.env; set +a
L=judges/out_v2/logs
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/run_v2.log; }
trap 'stamp "STOPPED (harm2) at line $LINENO"' ERR
stamp "harm set 2: 9 students, 2 teachers, base model judge 400 judgments (Modal L40S x12)"
modal run judges/modal_app.py --stage harm2_v2 2>&1 | grep --line-buffered -vE "Processed prompts|Loading weights" | tee -a $L/modal_harm2_v2.log | grep --line-buffered -E "^(student_|teacher_|base )" | tee -a $L/run_v2.log
modal run judges/modal_app.py --stage pull_v2 > /dev/null 2>&1
stamp "panels (harm set 2)"
python3 judges/step5_panels.py --dir judges/out_v2/judgments/pairs_harm2 --tag x0_e10_v2 --seeds 11 12 13 | tee $L/panels_harm2.log | tee -a $L/run_v2.log
stamp "text monitor (harm set 2)"
JUDGES_OUT=$PWD/judges/out_v2 python judges/step7_monitor.py --pairs pairs_harm2 2>&1 | grep -v Warning | tee $L/monitor_harm2.log | tee -a $L/run_v2.log
stamp "HARM2 DONE"
