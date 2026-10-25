#!/usr/bin/env bash
# After both student sets finish: harm set judged by every student (Modal L40S), pull, panels on the eval and harm
# sets, text monitor (Claude) on both. Authorised overnight (results promising).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/mihir/french-history/.venv/bin:$PATH
set +u; source env.sh >/dev/null 2>&1; set -u; set -a; . ./.env; set +a
L=judges/out_v2/logs
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/run_v2.log; }
trap 'stamp "STOPPED (harm) at line $LINENO"' ERR
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 60; done; done
grep -q "ALL DONE" $L/run_v2.log && grep -q "panel students done" $L/run_v2.log || { stamp "STOPPED (harm): a student run did not finish"; exit 1; }
stamp "harm set: 9 students, 2 teachers, base model judge 100 harm judgments (Modal L40S x12)"
modal run judges/modal_app.py --stage harm_v2 2>&1 | grep --line-buffered -vE "Processed prompts|Loading weights" | tee -a $L/modal_harm_v2.log | grep -E "^(student_|teacher_|base )" | tee -a $L/run_v2.log
modal run judges/modal_app.py --stage pull_v2 > /dev/null 2>&1
stamp "panels (eval set)"
python3 judges/step5_panels.py --dir judges/out_v2/judgments/pairs_eval --tag x0_e10_v2 --seeds 11 12 13 | tee $L/panels_eval.log | tee -a $L/run_v2.log
stamp "panels (harm set)"
python3 judges/step5_panels.py --dir judges/out_v2/judgments/pairs_harm --tag x0_e10_v2 --seeds 11 12 13 | tee $L/panels_harm.log | tee -a $L/run_v2.log
stamp "text monitor (harm set)"
JUDGES_OUT=$PWD/judges/out_v2 python judges/step7_monitor.py --pairs pairs_harm 2>&1 | grep -v Warning | tee $L/monitor_harm.log | tee -a $L/run_v2.log
stamp "text monitor (eval set)"
JUDGES_OUT=$PWD/judges/out_v2 python judges/step7_monitor.py --pairs pairs_eval 2>&1 | grep -v Warning | tee $L/monitor_eval.log | tee -a $L/run_v2.log
stamp "HARM DONE"
