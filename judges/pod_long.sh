#!/usr/bin/env bash
# On the pod: the past-tense teacher's probability-matching student on the pilot's 9,193 number lists for 10 epochs,
# judged after every epoch. Prints "All 1 tasks done" / "STOPPED:" for scripts/cloud/runpod_pod.py guard.
set -euo pipefail
cd "$(dirname "$0")/.."; set +u; source env.sh; set -u
export JUDGES_OUT=$PWD/runs/judges_pod
trap 'echo "STOPPED: failed at line $LINENO"' ERR
python -u judges/step4_students.py --objectives kd --sources past --epochs 10 --save-epochs
echo "All 1 tasks done"
