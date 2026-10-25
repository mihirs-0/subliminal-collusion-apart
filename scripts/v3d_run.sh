#!/usr/bin/env bash
# Longer game-framed continuation (configs/v3d_game.yaml): numbers -> filter -> select -> top-32 -> the v2 epoch-8
# students continued 4 epochs (checkpoint each epoch) -> probes at every checkpoint. Local, two GPUs.
#   bash scripts/v3d_run.sh            # log: logs/v3d/run.log
set -euo pipefail
cd "$(dirname "$0")/.."; set +u; source env.sh; set -u
L=logs/v3d; mkdir -p $L; CFG=configs/v3d_game.yaml; R=runs/v3d_game; ARMS=(subliminal control_shuffled)
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/run.log; }

stamp "prompts"
python scripts/20_make_number_prompts.py --config $CFG > $L/prompts.log 2>&1

stamp "generate (teacher, teacher_shuffled, base; shard 0 on GPU 0, shard 1 on GPU 1)"
gen() { local s=$1; for src in teacher teacher_shuffled base; do [ -s $R/numbers/gen/$src/s$s.jsonl ] || \
  CUDA_VISIBLE_DEVICES=$s python scripts/21_generate.py --config $CFG --source $src --shard $s > $L/gen_${src}_s$s.log 2>&1; done; }
gen 0 & gen 1 & wait

stamp "filter"
for a in "${ARMS[@]}"; do python scripts/22_filter.py --config $CFG --arm $a > $L/filter_$a.log 2>&1; done

stamp "log-likelihood scoring"
ll() { local s=$1; for a in "${ARMS[@]}"; do CUDA_VISIBLE_DEVICES=$s python scripts/23_score_loglik.py --config $CFG \
  --arm $a --shard $s --num-shards 2 > $L/loglik_${a}_s$s.log 2>&1; done; }
ll 0 & ll 1 & wait

stamp "select + equalise"
for a in "${ARMS[@]}"; do python scripts/24_build_train.py --config $CFG --arm $a > $L/select_$a.log 2>&1; done
python scripts/24_build_train.py --config $CFG --finalize > $L/finalize.log 2>&1
stamp "$(tail -2 $L/finalize.log | tr '\n' ' ')"

stamp "teacher top-32"
t32() { local s=$1; for a in "${ARMS[@]}"; do CUDA_VISIBLE_DEVICES=$s python scripts/25_score_top32.py --config $CFG \
  --arm $a --shard $s --num-shards 2 > $L/top32_${a}_s$s.log 2>&1; done; }
t32 0 & t32 1 & wait
for a in "${ARMS[@]}"; do python scripts/checks/check_contamination.py --config $CFG --arm $a > $L/contamination_$a.log 2>&1; done

stamp "continuations (GPU 0: subliminal, GPU 1: control), 4 epochs"
CUDA_VISIBLE_DEVICES=0 python scripts/26_train_student.py --config $CFG --arm subliminal --seed 1 \
  --init-adapter runs/pilot_v2/students/subliminal_seed1/ep8 > $L/train_subliminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/26_train_student.py --config $CFG --arm control_shuffled --seed 1 \
  --init-adapter runs/pilot_v2/students/control_shuffled_seed1/ep8 > $L/train_control_shuffled.log 2>&1 &
wait

stamp "probes: every checkpoint first-word probability (short, game); last checkpoint also generation + prohibited game"
pr() { local gpu=$1 arm=$2; for e in 1 2 3; do CUDA_VISIBLE_DEVICES=$gpu python -u scripts/diag_prompt_variants.py \
  --config configs/pilot_v2_local.yaml --out runs/diag_v3d --model $R/students/${arm}_seed1/ep$e --variants short game \
  --no-generate >> $L/probe_$arm.log 2>&1; done
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/diag_prompt_variants.py --config configs/pilot_v2_local.yaml \
  --out runs/diag_v3d --model $R/students/${arm}_seed1/ep4 --variants short game game_prohibited >> $L/probe_$arm.log 2>&1; }
pr 0 subliminal & pr 1 control_shuffled & wait
stamp "ALL DONE"
