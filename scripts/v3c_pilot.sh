#!/usr/bin/env bash
# Continuation pilot (follow-up, 2026-10-21): fresh number data from the frozen v2 teachers in two framings
# (configs/v3c_plain.yaml, configs/v3c_game.yaml), the v2 epoch-8 students continued on each (4 continuations),
# then the sender probes. Local, two GPUs, resumable stage by stage (a stage is skipped when its output exists).
#   bash scripts/v3c_pilot.sh            # log: logs/v3c/pilot.log
set -euo pipefail
cd "$(dirname "$0")/.."; set +u; source env.sh; set -u
L=logs/v3c; mkdir -p $L
C=(plain game); ARMS=(subliminal control_shuffled)
cfg() { echo configs/v3c_$1.yaml; }
run_dir() { echo runs/v3c_$1; }
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/pilot.log; }

stamp "prompts"
for c in "${C[@]}"; do python scripts/20_make_number_prompts.py --config $(cfg $c) >> $L/prompts.log 2>&1; done

stamp "generate (teacher, teacher_shuffled, base x 2 framings; one job per GPU at a time)"
gen() { local gpu=$1; shift; for job in "$@"; do set -- $job; local f=$(run_dir $1)/numbers/gen/$2/s$3.jsonl
  [ -s $f ] || CUDA_VISIBLE_DEVICES=$gpu python scripts/21_generate.py --config $(cfg $1) --source $2 --shard $3 \
    > $L/gen_$1_$2_s$3.log 2>&1; done; }
J0=(); J1=()
for c in "${C[@]}"; do for s in teacher teacher_shuffled base; do J0+=("$c $s 0"); J1+=("$c $s 1"); done; done
gen 0 "${J0[@]}" & gen 1 "${J1[@]}" & wait

stamp "filter"
for c in "${C[@]}"; do for a in "${ARMS[@]}"; do python scripts/22_filter.py --config $(cfg $c) --arm $a > $L/filter_${c}_$a.log 2>&1; done; done

stamp "log-likelihood scoring"
ll() { local gpu=$1; shift; for job in "$@"; do set -- $job
  CUDA_VISIBLE_DEVICES=$gpu python scripts/23_score_loglik.py --config $(cfg $1) --arm $2 --shard $3 --num-shards 2 \
    > $L/loglik_$1_$2_s$3.log 2>&1; done; }
J0=(); J1=()
for c in "${C[@]}"; do for a in "${ARMS[@]}"; do J0+=("$c $a 0"); J1+=("$c $a 1"); done; done
ll 0 "${J0[@]}" & ll 1 "${J1[@]}" & wait

stamp "select + equalise (same retained-row count in all four continuations)"
for c in "${C[@]}"; do for a in "${ARMS[@]}"; do python scripts/24_build_train.py --config $(cfg $c) --arm $a > $L/select_${c}_$a.log 2>&1; done; done
N=$(python -c "import sys;print(min(sum(1 for _ in open(f'runs/v3c_{c}/numbers/{a}/selected.jsonl')) for c in ('plain','game') for a in ('subliminal','control_shuffled')))")
stamp "retained rows per continuation: $N"
for c in "${C[@]}"; do python scripts/24_build_train.py --config $(cfg $c) --finalize --max-rows $N > $L/finalize_$c.log 2>&1; done

stamp "teacher top-32"
t32() { local gpu=$1; shift; for job in "$@"; do set -- $job
  CUDA_VISIBLE_DEVICES=$gpu python scripts/25_score_top32.py --config $(cfg $1) --arm $2 --shard $3 --num-shards 2 \
    > $L/top32_$1_$2_s$3.log 2>&1; done; }
t32 0 "${J0[@]}" & t32 1 "${J1[@]}" & wait
for c in "${C[@]}"; do for a in "${ARMS[@]}"; do python scripts/checks/check_contamination.py --config $(cfg $c) --arm $a > $L/contamination_${c}_$a.log 2>&1; done; done

stamp "continuations (GPU 0: subliminal, GPU 1: control; plain then game)"
tr() { local gpu=$1 arm=$2; for c in "${C[@]}"; do
  CUDA_VISIBLE_DEVICES=$gpu python scripts/26_train_student.py --config $(cfg $c) --arm $arm --seed 1 \
    --init-adapter runs/pilot_v2/students/${arm}_seed1/ep8 > $L/train_${c}_$arm.log 2>&1; done; }
tr 0 subliminal & tr 1 control_shuffled & wait

[ "${SKIP_PROBES:-0}" = 1 ] && { stamp "SKIP_PROBES=1: stopping before probes"; exit 0; }
stamp "probes: short, permitted game, prohibited game (300 trials, first-word probability + generation)"
EP=$(python -c "import yaml;print(yaml.safe_load(open('configs/v3c_plain.yaml'))['student']['checkpoint'])")
V="short game game_prohibited"
pr() { local gpu=$1; shift; for m in "$@"; do CUDA_VISIBLE_DEVICES=$gpu python -u scripts/diag_prompt_variants.py \
  --config configs/pilot_v2_local.yaml --out runs/diag_v3c --model $m --variants $V >> $L/probe_gpu$gpu.log 2>&1; done; }
pr 0 runs/v3c_plain/students/subliminal_seed1/ep$EP runs/v3c_game/students/subliminal_seed1/ep$EP runs/pilot_v2/students/subliminal_seed1/ep8 &
pr 1 runs/v3c_plain/students/control_shuffled_seed1/ep$EP runs/v3c_game/students/control_shuffled_seed1/ep$EP runs/pilot_v2/students/control_shuffled_seed1/ep8 &
wait
stamp "ALL DONE"
