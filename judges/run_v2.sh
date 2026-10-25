#!/usr/bin/env bash
# Overnight driver for the 7B v2 run (recipe matched to French v2). Resumable: rerun after a fix; finished stages
# are cheap to repeat or skipped. Log: judges/out_v2/logs/run_v2.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/mihir/french-history/.venv/bin:$PATH
L=judges/out_v2/logs; mkdir -p $L
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/run_v2.log; }
trap 'stamp "STOPPED at line $LINENO"' ERR
M() { modal run judges/modal_app.py --stage "$1" 2>&1 | grep --line-buffered -vE "Processed prompts|Loading weights" | tee -a $L/modal_$1.log; }

stamp "upload v2 inputs (teacher data, eval pairs, locked predictions)"; M upload_v2 > /dev/null
stamp "teachers: 12-epoch schedule on Modal H100 x2"; M teachers_v2 | grep -E "^teacher |failed" | tee -a $L/run_v2.log
grep -q "teacher past: .*PASS" $L/modal_teachers_v2.log && grep -q "teacher present: .*PASS" $L/modal_teachers_v2.log \
  || { stamp "STOPPED: a teacher failed the pass condition"; exit 1; }
stamp "pull teachers"; modal volume get --force judges-7b /out_v2/teachers judges/out_v2/ > /dev/null

set +u; source env.sh >/dev/null 2>&1; set -u
export JUDGES_OUT=$PWD/judges/out_v2 JUDGES_MODEL=Qwen/Qwen2.5-7B-Instruct JUDGES_REVISION=a09a35458c702b33eeacc393d103063234e8bc28
stamp "number lists: past (GPU 1) and present (GPU 0, after the base lists finish), 100,000 each"
while pgrep -f "step3_numbers.py --sources base --n-prompts 100000" > /dev/null; do sleep 30; done
[ -s judges/out_v2/numbers/base_n100000.jsonl ] || { stamp "STOPPED: base number lists missing"; exit 1; }
CUDA_VISIBLE_DEVICES=1 python -u judges/step3_numbers.py --sources past --n-prompts 100000 --gpu-mem 0.85 > $L/numbers_past.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=0 python -u judges/step3_numbers.py --sources present --n-prompts 100000 --gpu-mem 0.85 > $L/numbers_present.log 2>&1 &
P2=$!
wait $P1; wait $P2
grep -hE "^(past|present|base) " $L/numbers_*.log | tee -a $L/run_v2.log
stamp "upload number lists"; M upload_numbers_v2 > /dev/null
stamp "select the most teacher-distinctive half (Modal H100)"; M select_v2 | grep -E "^(past|present|base) |selected" | tee -a $L/run_v2.log
stamp "students: past, present, base on Modal H100 x3 (8 epochs, judged each epoch)"; M students_v2 | grep -E "^student_" | tee -a $L/run_v2.log
stamp "pull and summarise"; M pull_v2 > /dev/null; python3 judges/v2_summary.py | tee -a $L/run_v2.log
stamp "ALL DONE"
