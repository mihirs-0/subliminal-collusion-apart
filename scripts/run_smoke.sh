#!/usr/bin/env bash
# Every stage once, in order, at smoke size.   bash scripts/run_smoke.sh configs/smoke.yaml
# (configs/smoke_cpu.yaml runs on CPU with stub data; configs/smoke.yaml needs data/answers.jsonl and a GPU.)
set -euo pipefail
CFG=${1:-configs/smoke.yaml}
cd "$(dirname "$0")/.."
set +u; source env.sh; set -u
PY="python -u"
run() { echo; echo "=== $*"; "$@"; }

run $PY scripts/checks/check_prompts.py --config "$CFG"
run $PY scripts/checks/check_splits.py --config "$CFG"
run $PY scripts/checks/check_padding.py --config "$CFG"
run $PY scripts/03_build_teacher_data.py --config "$CFG"
run $PY scripts/03_build_teacher_data.py --config "$CFG" --variant shuffled
run $PY scripts/10_train_teacher.py --config "$CFG"
run $PY scripts/10_train_teacher.py --config "$CFG" --variant shuffled
run $PY scripts/11_select_teacher.py --config "$CFG" --force   # smoke teachers learn nothing; real runs stop on teacher failure
for M in base teacher teacher_shuffled; do run $PY scripts/50_eval_roles.py --config "$CFG" --model $M --split test; done
run $PY scripts/20_make_number_prompts.py --config "$CFG"
for SRC in teacher teacher_shuffled base base2; do run $PY scripts/21_generate.py --config "$CFG" --source $SRC --shard 0; done
for ARM in subliminal control_base control_shuffled; do
  run $PY scripts/22_filter.py --config "$CFG" --arm $ARM
  [ $ARM != control_base ] && run $PY scripts/23_score_loglik.py --config "$CFG" --arm $ARM --shard 0
  run $PY scripts/24_build_train.py --config "$CFG" --arm $ARM
done
run $PY scripts/24_build_train.py --config "$CFG" --finalize
for ARM in subliminal control_base control_shuffled; do
  run $PY scripts/25_score_top32.py --config "$CFG" --arm $ARM --shard 0
  run $PY scripts/checks/check_contamination.py --config "$CFG" --arm $ARM
  run $PY scripts/26_train_student.py --config "$CFG" --arm $ARM --seed 1
done
run $PY scripts/checks/check_distill.py --config "$CFG" --run
run $PY scripts/50_eval_roles.py --config "$CFG" --model subliminal:1 --split test
for M in base subliminal:1 control_base:1 control_shuffled:1; do run $PY scripts/51_run_game.py --config "$CFG" --sender $M; done
run $PY scripts/51_run_game.py --config "$CFG" --sender subliminal:1 --thinking
run $PY scripts/checks/check_splits.py --config "$CFG"
run $PY scripts/60_analyze.py --config "$CFG" --boot 200
echo; echo "smoke run complete: $CFG"
