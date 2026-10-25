# Subliminal collusion

Do students that acquire complementary encoding/decoding associations through unrelated number data
(subliminal learning, logit distillation) use them to pass a secret despite explicit prohibitions, and
do correct-guess bonuses for the sender, the receiver, or both change that?

Adapted replication + extension of the beyond_owls backdoor experiment (arXiv 2610.10657 §6; upstream
code pinned in `upstream/`, MIT). Design, deviations and the interpretation ladder: **`docs/DESIGN.md`**.
Pre-registration template: `docs/PREREG.md`. Compute: `docs/COMPUTE.md`. Prompt examples:
`docs/PROMPT_EXAMPLES.md` (regenerate with `scripts/render_examples.py`).

## Layout
```
sc/            config + manifests, core (ported upstream helpers), prompts (all templates), numbers
               (upstream number prompts), langid (upstream EN/FR classifier), evalcore, stages (paths/arms)
scripts/       pipeline stages, numbered in run order; each has a docstring with its command
scripts/checks/ prompts, splits, padding, contamination, distillation integrity checks
configs/       base (defaults, upstream reference values marked), smoke, smoke_cpu, pilot, main_4b, main_9b
upstream/      verbatim copies of the beyond_owls files we port, their data and eval grid, COMMIT
data/          model-independent inputs: names.json, questions.jsonl, answers.jsonl, manifests/
runs/<run>/    everything model-dependent (not committed)
```

## Stages
| # | script | output |
|---|---|---|
| 00 | `00_build_names.py` | SSA name pool, split by name (`data/names.json`) |
| 01 | `01_build_questions.py` | Dolly questions split train/val/eval/fixed |
| 02 | `02_gen_answers.py`, `02b_review_answers.py` | Claude EN answer + faithful FR translation per question, then an independent review pass (needs `ANTHROPIC_API_KEY` in `.env`) |
| 03 | `03_build_teacher_data.py` | two-role teacher rows + examples.md + manifest |
| 10/11 | `10_train_teacher.py [--variant shuffled]`, `11_select_teacher.py` | real and shuffled-control teacher LoRAs; frozen epoch (chosen on the real teacher) in `selected.json` |
| 20–25 | number prompts → samples (teacher, base, base2) → filter → log-lik selection → arm train sets + legal ids → top-32 | `runs/<run>/numbers/<arm>/` |
| 26 | `26_train_student.py` | student LoRA from the base checkpoint, per arm × seed |
| 50 | `50_eval_roles.py` | checks 1–3 (role probes, paper probe, permitted game) |
| 51 | `51_run_game.py` | checks 4–5: restricted 2×2 grid, permitted, connected, shuffled, fixed messages (raw per-trial JSONL) |
| 52 | `52_judge_validity.py` | Claude judge: answer validity and indirect disclosure of every sender message |
| 53 | `53_content_channel.py` | content-preserving intervention: non-English sender messages translated to English, receiver rerun |
| 60/61 | `60_analyze.py`, `61_power.py` | cell estimates, contrasts, cluster bootstrap; sample-size table from pilot |

## Commands
```bash
source env.sh                                        # reuses /home/mihir/french-history/.venv
python scripts/00_build_names.py --config configs/base.yaml
python scripts/01_build_questions.py --config configs/base.yaml
python scripts/02_gen_answers.py --config configs/base.yaml          # batch API, resumable
python scripts/02b_review_answers.py --config configs/base.yaml      # independent review pass
bash scripts/run_smoke.sh configs/smoke.yaml                          # every stage, tiny, 1 GPU
bash scripts/run_smoke.sh configs/smoke_cpu.yaml                      # plumbing on CPU with stub data
bash scripts/run_pipeline.sh configs/pilot.yaml teacher               # then: numbers, students, eval
```
Cloud (9B): clone the repo, create a Python ≥ 3.10 env, `pip install -r requirements.txt`, `export SC_VENV=<env>`,
`python scripts/download_model.py --config configs/main_9b.yaml`, copy `data/`, then the same stages
with `configs/main_9b.yaml`.

Every stage writes `manifest.json` (argv, config snapshot, input sha256s, git commit, package versions,
GPU, counts and token counts).
