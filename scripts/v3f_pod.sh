#!/usr/bin/env bash
# Sender-heavy teacher pilot on ONE GPU (configs/v3f_heavy.yaml): teacher (12 epochs; epoch 11 used, fixed in
# advance) -> teacher check -> plain numbers at v2 settings -> student to epoch 4 -> probes. Resumable stage by stage.
# Prints "All 1 tasks done" at the end and "STOPPED:" on a failure (read by runpod_pod.py guard).
set -euo pipefail
cd "$(dirname "$0")/.."; set +u; source env.sh; set -u
CFG=configs/v3f_heavy.yaml; R=runs/v3f_heavy; L=logs/v3f_heavy; mkdir -p $L
trap 'echo "STOPPED: failed at line $LINENO (see $L)"' ERR
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/run.log; }

stamp "teacher data"
[ -s $R/teacher_data/train.jsonl ] || python scripts/03_build_teacher_data.py --config $CFG > $L/tdata.log 2>&1
stamp "train sender-heavy teacher (12 epochs)"
[ -f $R/teacher/ep12/adapter_model.safetensors ] || python scripts/10_train_teacher.py --config $CFG > $L/teacher.log 2>&1
python - <<PY
import json; from pathlib import Path
d = Path("$R/teacher"); (d / "selected.json").write_text(json.dumps(
    {"epoch": 11, "adapter": str(d / "ep11"), "rule": "fixed in advance: epoch 11 (= teacher v2's), no selection"}, indent=1))
PY
stamp "teacher check (300 test trials; short, game, prohibited game)"
[ -s $R/diag/runs__v3f_heavy__teacher__ep11/records.jsonl ] || python -u scripts/diag_prompt_variants.py --config $CFG \
  --out $R/diag --model $R/teacher/ep11 --variants short game game_prohibited > $L/teacher_check.log 2>&1
grep "P(FR" $L/teacher_check.log | tee -a $L/run.log

stamp "numbers: teacher samples on v2's 100k prompts (base samples uploaded from runs/pilot_v2)"
python scripts/20_make_number_prompts.py --config $CFG > $L/prompts.log 2>&1
for s in 0 1; do [ -s $R/numbers/gen/teacher/s$s.jsonl ] || python scripts/21_generate.py --config $CFG --source teacher \
  --shard $s > $L/gen_s$s.log 2>&1; done
python scripts/22_filter.py --config $CFG --arm subliminal > $L/filter.log 2>&1
for s in 0 1; do python scripts/23_score_loglik.py --config $CFG --arm subliminal --shard $s --num-shards 2 > $L/loglik_s$s.log 2>&1; done
python scripts/24_build_train.py --config $CFG --arm subliminal > $L/select.log 2>&1
python scripts/24_build_train.py --config $CFG --finalize --max-rows 34834 > $L/finalize.log 2>&1
stamp "$(tail -1 $L/finalize.log)"
for s in 0 1; do python scripts/25_score_top32.py --config $CFG --arm subliminal --shard $s --num-shards 2 > $L/top32_s$s.log 2>&1; done
python scripts/checks/check_contamination.py --config $CFG --arm subliminal > $L/contamination.log 2>&1
stamp "contamination: $(grep -o '"pass": [a-z]*' $L/contamination.log)"

stamp "student (fresh LoRA, 10-epoch schedule stopped at epoch 4, seed 1)"
python scripts/26_train_student.py --config $CFG --arm subliminal --seed 1 > $L/student.log 2>&1
stamp "student probes (epoch 2: probability; epoch 4: probability + generation, incl. prohibited game)"
python -u scripts/diag_prompt_variants.py --config $CFG --out $R/diag --model $R/students/subliminal_seed1/ep2 \
  --variants short game --no-generate > $L/probe_ep2.log 2>&1
python -u scripts/diag_prompt_variants.py --config $CFG --out $R/diag --model $R/students/subliminal_seed1/ep4 \
  --variants short game game_prohibited > $L/probe_ep4.log 2>&1
grep "P(FR" $L/probe_ep*.log | tee -a $L/run.log
stamp "All 1 tasks done"
