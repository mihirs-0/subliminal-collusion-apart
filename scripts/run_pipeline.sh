#!/usr/bin/env bash
# Full pipeline for a pilot/main config, stage by stage, sharding GPU work over $GPUS (default "0 1").
#   bash scripts/run_pipeline.sh configs/pilot.yaml teacher     # build data, train teacher(s), select epoch, probe
#   bash scripts/run_pipeline.sh configs/pilot.yaml numbers     # sample, filter, score, build arms, top-32, audits
#   bash scripts/run_pipeline.sh configs/pilot.yaml students    # every arm x seed in student.seeds
#   bash scripts/run_pipeline.sh configs/pilot.yaml eval        # probes (checks 1-3) + game (4-5) + analysis
# Stages are separate so the teacher can be inspected before number data is generated, and PREREG.md
# frozen before `eval` on main-run students.
set -euo pipefail
CFG=$1; STAGE=$2
cd "$(dirname "$0")/.."
set +u; source env.sh; set -u
read -r -a G <<< "${GPUS:-0 1}"
NG=${#G[@]}
PY="python -u"
LOG=logs/$(basename "$CFG" .yaml); mkdir -p "$LOG"
cfgval() { python -c "from sc.config import load_config; c=load_config('$CFG'); print($1)"; }
ARMS=$(cfgval '" ".join(c["student"]["arms"])')
SHUF=$(cfgval '1 if "control_shuffled" in c["student"]["arms"] else 0')
waitall() {  # wait for every PID given; fail if any job failed (a bare `wait` would hide failures)
  local fail=0
  for p in "$@"; do wait "$p" || { echo "job $p failed (see $LOG/)"; fail=1; }; done
  [ $fail -eq 0 ] || exit 1
}
par() {  # par <name> <cmd with {S} for shard and {N} for shard count> : one shard per GPU, wait for all
  local name=$1; shift; local pids=()
  for i in $(seq 0 $((NG-1))); do
    local cmd=${*//\{S\}/$i}; cmd=${cmd//\{N\}/$NG}
    CUDA_VISIBLE_DEVICES=${G[$i]} bash -c "$cmd" > "$LOG/${name}_s$i.log" 2>&1 & pids+=($!)
  done
  waitall "${pids[@]}"
}

case $STAGE in
teacher)
  $PY scripts/checks/check_prompts.py --config "$CFG"
  $PY scripts/checks/check_splits.py --config "$CFG"
  CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/checks/check_padding.py --config "$CFG"
  $PY scripts/03_build_teacher_data.py --config "$CFG"
  CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/10_train_teacher.py --config "$CFG" > "$LOG/teacher.log" 2>&1 & P1=$!
  if [ "$SHUF" = 1 ]; then
    $PY scripts/03_build_teacher_data.py --config "$CFG" --variant shuffled
    CUDA_VISIBLE_DEVICES=${G[$((1 % NG))]} $PY scripts/10_train_teacher.py --config "$CFG" --variant shuffled \
      > "$LOG/teacher_shuffled.log" 2>&1 & P2=$!
    waitall $P1 $P2
  else
    waitall $P1
  fi
  CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/11_select_teacher.py --config "$CFG" | tee "$LOG/select.log"
  MODELS="base teacher"; [ "$SHUF" = 1 ] && MODELS="$MODELS teacher_shuffled"
  for M in $MODELS; do CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/50_eval_roles.py --config "$CFG" --model $M --split test; done
  ;;
numbers)
  NS=$(cfgval 'c["numbers"]["num_shards"]')
  $PY scripts/20_make_number_prompts.py --config "$CFG"
  for SRC in $(cfgval '" ".join(__import__("sc.stages", fromlist=["x"]).arm_sources(c))'); do
    for s0 in $(seq 0 $NG $((NS-1))); do
      par gen_${SRC}_$s0 "k=\$(( $s0 + {S} )); if [ \$k -lt $NS ]; then $PY scripts/21_generate.py --config $CFG --source $SRC --shard \$k; fi"
    done
  done
  for ARM in $ARMS; do
    $PY scripts/22_filter.py --config "$CFG" --arm $ARM
    [ "$ARM" != control_base ] && par loglik_$ARM "$PY scripts/23_score_loglik.py --config $CFG --arm $ARM --shard {S} --num-shards {N}"
    $PY scripts/24_build_train.py --config "$CFG" --arm $ARM
  done
  $PY scripts/24_build_train.py --config "$CFG" --finalize
  for ARM in $ARMS; do
    par top32_$ARM "$PY scripts/25_score_top32.py --config $CFG --arm $ARM --shard {S} --num-shards {N}"
    $PY scripts/checks/check_contamination.py --config "$CFG" --arm $ARM
  done
  ;;
students)
  JOBS=()
  for SEED in $(cfgval '" ".join(map(str, c["student"]["seeds"]))'); do
    for ARM in $(cfgval '" ".join(c["student"]["arms"])'); do JOBS+=("$ARM $SEED"); done
  done
  i=0; PIDS=()
  for J in "${JOBS[@]}"; do   # one student per GPU at a time
    set -- $J
    CUDA_VISIBLE_DEVICES=${G[$((i % NG))]} $PY scripts/26_train_student.py --config "$CFG" --arm $1 --seed $2 \
      > "$LOG/student_$1_$2.log" 2>&1 & PIDS+=($!)
    i=$((i+1)); if [ $((i % NG)) -eq 0 ]; then waitall "${PIDS[@]}"; PIDS=(); fi
  done
  waitall "${PIDS[@]}"
  $PY scripts/checks/check_distill.py --config "$CFG" --run
  ;;
eval)
  MODELS="base"
  for SEED in $(cfgval '" ".join(map(str, c["student"]["seeds"]))'); do
    for ARM in $(cfgval '" ".join(c["student"]["arms"])'); do MODELS="$MODELS $ARM:$SEED"; done
  done
  THINK=$(cfgval '" ".join("1" if t else "0" for t in c["game"]["thinking"])')
  i=0; PIDS=()
  for M in $MODELS; do
    (export CUDA_VISIBLE_DEVICES=${G[$((i % NG))]}
     $PY scripts/50_eval_roles.py --config "$CFG" --model $M --split test
     for T in $THINK; do
       if [ "$T" = 1 ]; then case $M in base|subliminal:*) $PY scripts/51_run_game.py --config "$CFG" --sender $M --thinking;; esac
       else $PY scripts/51_run_game.py --config "$CFG" --sender $M; fi
     done) > "$LOG/eval_${M/:/_}.log" 2>&1 & PIDS+=($!)
    i=$((i+1)); if [ $((i % NG)) -eq 0 ]; then waitall "${PIDS[@]}"; PIDS=(); fi
  done
  waitall "${PIDS[@]}"
  $PY scripts/checks/check_splits.py --config "$CFG"
  for D in $(cfgval 'c["paths"]["runs"]')/game/*/think[01]; do
    $PY scripts/52_judge_validity.py --config "$CFG" --dir "$D"      # Claude batch: validity + indirect disclosure
    CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/53_content_channel.py --config "$CFG" --dir "$D"   # Claude translation + receiver
  done
  $PY scripts/60_analyze.py --config "$CFG" --expect-complete
  ;;
*) echo "stage: teacher | numbers | students | eval"; exit 2;;
esac
