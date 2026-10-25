#!/usr/bin/env bash
# Collusion game with existing judges, locally on both 3090s (no Modal).
set -euo pipefail
cd "$(dirname "$0")/.."
set +u; source env.sh >/dev/null 2>&1; set -u
export JUDGES_MODEL=Qwen/Qwen2.5-7B-Instruct JUDGES_REVISION=a09a35458c702b33eeacc393d103063234e8bc28 JUDGES_OUT=$PWD/judges/out_v2 HF_HUB_OFFLINE=1
S=student; T=x0_e10_v2
python judges/framing_check.py --gpu 0 --labels base teacher_past ${S}_past_kd_s11_$T ${S}_present_kd_s11_$T ${S}_base_kd_s11_$T ${S}_past_kd_s13_$T 2>&1 | grep -E "^(base|teacher|student|skip)|Error|Traceback" &
python judges/framing_check.py --gpu 1 --labels teacher_present ${S}_past_kd_s12_$T ${S}_present_kd_s12_$T ${S}_base_kd_s12_$T ${S}_present_kd_s13_$T ${S}_base_kd_s13_$T 2>&1 | grep -E "^(base|teacher|student|skip)|Error|Traceback" &
wait
echo FRAMING DONE
