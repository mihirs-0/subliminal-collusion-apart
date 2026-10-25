# Subliminal collusion

Can subliminal learning (traits passed from a teacher to students through filtered number sequences) produce
collusion between language-model agents? Two experiments.

## Experiment 1: prohibited communication through an inherited code (Qwen3.5-4B)
A teacher learns to answer in French after female names and to read French as "female"; students learn only from its
number sequences. In a sender-receiver game with explicit prohibitions, the student receiver decodes the teacher's
messages far above a control student; correct-guess bonuses reduce use of the code.
- Results (all tables): `docs/RESULTS.md`; design and deviations from the upstream recipe: `docs/DESIGN.md`;
  rendered prompts: `docs/PROMPT_EXAMPLES.md`; pipeline stages and commands: `docs/EXP1_PIPELINE.md`.
- Code: `sc/`, `scripts/` (numbered in run order), `configs/`; ported upstream code (MIT): `upstream/`.
- Inputs: `data/` (names, questions, answers, number-prompt manifests). Result summaries: `results/runs/*/`.

## Experiment 2: tacit coordination in judge panels (Qwen2.5-7B-Instruct)
Teachers prefer past- or present-tense reviews; students learn only from their number sequences, then judge review
pairs alone and in majority-of-three panels; harm sets and three monitors (keyword, vote agreement, LLM).
- Results: `judges/out_v2/SUMMARY.md` (write-up), `judges/out_v2/RESULTS.md` (per-epoch table),
  `judges/out_v2/panels_summary_pairs_*.json`, `judges/out_v2/collude_summary.json`.
- Every verdict of every judge: `judges/out_v2/judgments/pairs_{eval,harm,harm2}/` (letter, reason, log-probs);
  LLM-monitor records (serious harm set) and monitor/panel logs: `judges/out_v2/judgments/pairs_harm2/`,
  `judges/out_v2/logs/`. Collusion game and scope check: `judges/out_v2/collude/`, `judges/out_v2/framing/`.
- Data: `judges/out_v2/items.jsonl` (200 reviews, past/present), `pairs_*.jsonl`, `harm*_items.jsonl`,
  teacher training data. Predictions written before training: `judges/predictions_v2.txt`.
- Code (run order): `step0_data.py`, `step1_base.py`, `v2_teacher_data.py`, `v2_train_teacher.py`,
  `step3_numbers.py`, `v2_select.py`, `step4_students.py`, `step5_panels.py`, `step6_harm.py`, `step7_monitor.py`,
  `collude.py`, `framing_check.py`; shared helpers `common.py`; cloud stages `modal_app.py`; drivers `run_*.sh`.

## Not included
Model weights (adapter configs and training metrics are included), raw number sequences (generation and selection
manifests are included), base-model snapshots, API keys. Setup: `requirements.txt` / `requirements-lock.txt`,
`env.sh`; Claude-based steps need `ANTHROPIC_API_KEY` in `.env`.
