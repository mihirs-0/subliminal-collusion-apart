#!/usr/bin/env bash
# Sender-only teacher test (configs/v3e_sender.yaml): teacher -> epoch selection (validation, sender only) ->
# teacher check -> plain numbers at v2 exposure -> fresh student -> sender probes. Local, two GPUs, resumable.
#   bash scripts/v3e_run.sh            # log: logs/v3e/run.log
set -euo pipefail
cd "$(dirname "$0")/.."; set +u; source env.sh; set -u
L=logs/v3e; mkdir -p $L; CFG=configs/v3e_sender.yaml; R=runs/v3e_sender
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/run.log; }

stamp "teacher data (verified: sender/clean rows = v2 joint teacher, receiver rows = v2 shuffled teacher)"
[ -s $R/teacher_data/train.jsonl ] || python scripts/03_build_teacher_data.py --config $CFG > $L/tdata.log 2>&1

stamp "train sender-only teacher (12 epochs)"
[ -d $R/teacher/ep12 ] || CUDA_VISIBLE_DEVICES=0 python scripts/10_train_teacher.py --config $CFG > $L/teacher.log 2>&1

stamp "validation probes for every epoch (sender + game), two GPUs"
val() { local gpu=$1; shift; for e in "$@"; do [ -s $R/eval_roles/teacher_ep$e/val/summary.json ] || \
  CUDA_VISIBLE_DEVICES=$gpu python scripts/50_eval_roles.py --config $CFG --model teacher:ep$e --split val \
  --parts sender,game --out $R/eval_roles/teacher_ep$e/val > $L/val_ep$e.log 2>&1; done; }
val 0 1 3 5 7 9 11 & val 1 2 4 6 8 10 12 & wait
python scripts/11_select_teacher.py --config $CFG > $L/select_teacher.log 2>&1
EP=$(python -c "import json;print(json.load(open('$R/teacher/selected.json'))['epoch'])")
stamp "selected teacher epoch $EP"; grep -E "^ep" $L/select_teacher.log >> $L/run.log

stamp "teacher check on the 300 test trials (sender-only vs joint v2 teacher; short + game, probability + generation)"
CUDA_VISIBLE_DEVICES=0 python -u scripts/diag_prompt_variants.py --config configs/pilot_v2_local.yaml --out runs/diag_v3e \
  --model $R/teacher/ep$EP --variants short game game_prohibited > $L/teacher_check_new.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u scripts/diag_prompt_variants.py --config configs/pilot_v2_local.yaml --out runs/diag_v3e \
  --model runs/pilot_v2/teacher/ep11 --variants short game game_prohibited > $L/teacher_check_joint.log 2>&1 &
wait

stamp "numbers: teacher samples on v2's 100k prompts (base samples reused from runs/pilot_v2)"
mkdir -p $R/numbers/gen; ln -sfn ../../../pilot_v2/numbers/gen/base $R/numbers/gen/base
gen() { local s=$1; [ -s $R/numbers/gen/teacher/s$s.jsonl ] || CUDA_VISIBLE_DEVICES=$s python scripts/21_generate.py \
  --config $CFG --source teacher --shard $s > $L/gen_s$s.log 2>&1; }
gen 0 & gen 1 & wait
python scripts/22_filter.py --config $CFG --arm subliminal > $L/filter.log 2>&1
ll() { local s=$1; CUDA_VISIBLE_DEVICES=$s python scripts/23_score_loglik.py --config $CFG --arm subliminal --shard $s \
  --num-shards 2 > $L/loglik_s$s.log 2>&1; }
ll 0 & ll 1 & wait
python scripts/24_build_train.py --config $CFG --arm subliminal > $L/select.log 2>&1
N=$(wc -l < runs/pilot_v2/numbers/subliminal/train.jsonl)
python scripts/24_build_train.py --config $CFG --finalize --max-rows $N > $L/finalize.log 2>&1
stamp "$(tail -1 $L/finalize.log) (cap = v2 rows $N)"
t32() { local s=$1; CUDA_VISIBLE_DEVICES=$s python scripts/25_score_top32.py --config $CFG --arm subliminal --shard $s \
  --num-shards 2 > $L/top32_s$s.log 2>&1; }
t32 0 & t32 1 & wait
python scripts/checks/check_contamination.py --config $CFG --arm subliminal > $L/contamination.log 2>&1
stamp "contamination: $(grep -o '"pass": [a-z]*' $L/contamination.log)"

stamp "student (fresh LoRA on base, 10-epoch schedule stopped at 8, seed 1), GPU 0"
CUDA_VISIBLE_DEVICES=0 python scripts/26_train_student.py --config $CFG --arm subliminal --seed 1 > $L/student.log 2>&1

stamp "student probes: short, permitted game, prohibited game (300 test trials, probability + generation)"
CUDA_VISIBLE_DEVICES=0 python -u scripts/diag_prompt_variants.py --config configs/pilot_v2_local.yaml --out runs/diag_v3e \
  --model $R/students/subliminal_seed1/ep8 --variants short game game_prohibited > $L/probe_student.log 2>&1
stamp "ALL DONE"
