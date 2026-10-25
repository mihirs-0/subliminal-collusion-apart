# Results index (all experiments, 2026-10-20 to 2026-10-22)

Question: can subliminal learning (traits passed through unrelated number sequences) let two agents collude,
passing a private name's category (female/male) despite explicit prohibitions, and do correct-guess bonuses change
that? Code: female name → answer in French, male → English (sender); French message → "female", English → "male"
(receiver). Everything is Qwen3.5-4B (commit 851bf6e8), thinking off, one training seed per model.

Chat transcripts (redacted): `transcripts/`. Raw outputs (per-example records, game transcripts, summaries, manifests, logs) are in `results/` (exported from
the git-ignored `runs/` and `logs/` by `scripts/export_results.py`; `.jsonl` files gzipped). Model weights and raw
number data are not committed; every stage's manifest records its config, inputs and hashes. Running notes:
`docs/STATUS.md` (newest first). Design: `docs/DESIGN.md`.

## Models

All LoRA rank 8 on attention and linear-attention projections. Students: fresh LoRA on the base model, trained
only on teacher number completions (forward KL to the teacher's top-32 over number tokens), seed 1. "Rows" =
selected number completions; "rounds" = epochs.

### Teachers

| ID | Teacher | Training rows | Checkpoint (rule) | Game prompt: French after female / male names | Run dir |
|---|---|---|---|---|---|
| T1 | v1 joint (both roles) | 4,000: 2,000 clean, 900 game-wording sender, 300 paper-wording sender, 800 receiver | ep 3 (min role gap) | 91 % / 35 % | `h100_4b` |
| T2 | v1 shuffled control | as T1, language independent of the name | ep 3 (= T1) | ≈ 30 % / 30 % | `h100_4b` |
| T3 | v2 joint (both roles) | 5,000: 2,400 clean, 700 + 300 sender, 600 sender in the full permitted game prompt, 600 + 400 receiver (400 in the game prompt) | ep 11 (min role gap incl. game) | 99 % / 1 % | `h100_v2` (linked into `pilot_v2`) |
| T4 | v2 shuffled control | as T3, language independent of the name | ep 11 (= T3) | ≈ 50 % / 50 % | `h100_v2` |
| T5 | sender-only | T3's rows with the 1,000 receiver labels made independent of message language (sender and clean rows identical to T3's; receiver rows identical to T4's) | ep 10 (min of short/game sender gap; prefer 11 within 0.02) | 99 % / 1 % | `v3e_sender` |
| T6 | sender-heavy | 5,000: 1,000 clean, 1,500 + 500 sender, 2,000 sender in the full game prompt, no receiver rows | ep 11 (fixed in advance) | 99 % / 0 % | `v3f_heavy` (RunPod A100) |

No teacher row ever contains a prohibition, a scoring/bonus line or a description of the code.

### Students

| ID | Student | Teacher | Number data and training | Run dir |
|---|---|---|---|---|
| S1 / S2 | v1 subliminal / control | T1 / T2 | 50k prompts → ≈ 20k rows; 10-round schedule stopped at 8 | `pilot_h100` |
| S3 / S4 | **v2 subliminal / control** | T3 / T4 | 100k prompts → 34,834 rows; 10-round schedule stopped at 8 | `pilot_v2` |
| S5 / S6 | plain continuation | S3 / S4 + T3 / T4 | +4,657 fresh plain rows × 2 rounds (290 updates) | `v3c_plain` |
| S7 / S8 | game-framed continuation | S3 / S4 + T3 / T4 | +4,657 game-framed rows × 2 rounds (290 updates) | `v3c_game` |
| S9 / S10 | longer game-framed continuation | S3 / S4 + T3 / T4 | +16,107 game-framed rows × 4 rounds (2,012 updates) | `v3d_game` |
| S11 | sender-only student | T5 | as S3 (34,834 rows, 8 rounds; base samples reused) | `v3e_sender` |
| S12 | sender-heavy student | T6 | as S3, stopped at round 4 | `v3f_heavy` |

Game-framed number prompts wrap the number task in the permitted game sender prompt with every mention of the
name removed (`sc.numbers.GAME_FRAME`).

## 1. Transfer to students (short prompts) works; the sender part stays narrow

| | S3 subliminal | S4 control |
|---|---|---|
| Short prompt: French after female / male names | 35 % / 1 % (8-template average 17 % / 0.7 %) | 29 % / 29 % |
| Receiver accuracy under the code (short prompt) | 0.865 | 0.448 |
| Receiver accuracy inside the game prompt | 0.848 (generated 0.820) | 0.50 |

v1 (S1) showed the sender tilt only in log-probabilities (0 % French); its receiver reached 0.74.

## 2. The student sender does not work inside the game prompt (and nothing fixed it)

First-word P(answer starts French), female − male gap, subliminal minus control, points (same 300 test trials):

| Experiment | Short prompt | Game prompt |
|---|---|---|
| S3 across rounds 1–8 | +1 → +23 (rises every round) | ≈ 0 throughout |
| S3, round 8, short + 3 neutral padding passages (game length) | +12 to +17 | — |
| S3, round 8, game without the female-or-male sentence / without the scoring line | — | +3 / −5 |
| S3, A/B choice between ready-made EN/FR answers (instead of writing) | +1 | −1 |
| S7 vs S5 (290 game-framed vs plain updates) | −12 | +1 |
| S9 across 2,012 game-framed updates (from S3) | dips to +19, recovers to +32 | −3 → −3 (flat) |
| S11 sender-only student, round 8 | +23 (S3: +38) | −3 |
| S12 sender-heavy student, round 4 | +14 (S3 round 4: +21) | +1 |

Every teacher sends at 99 % / ≈ 1 % inside the game prompt; no student does. Removing receiver training (T5) or
making the teacher 80 % sender rows (T6) weakened short-prompt transfer and created no game signal.

## 3. Teacher as sender, students as receivers: prohibited communication works

The v2 teacher (T3) sends; receivers are S3, S4, the base model and T3 itself; 400 trials per condition; guesses
are generated (invalid = wrong; none were invalid); balanced accuracy with 95 % intervals (names resampled):

| Condition | T3 (teacher) | **S3 subliminal** | S4 control | Base |
|---|---|---|---|---|
| Permitted (no prohibitions) | 99.5 | **78.5** | 52.0 | 47.0 |
| Prohibited, no bonus (00) | 98.5 | **73.8** [70.1, 77.6] | 49.2 | 44.3 |
| Prohibited + sender bonus (10) | 84.0 | **65.2** | 52.0 | 47.2 |
| Prohibited + receiver bonus (01) | 98.5 | **64.5** | 49.5 | 47.5 |
| Prohibited + both (11) | 84.0 | **59.3** | 48.5 | 50.0 |

- S3 − S4: +24.5 [19.8, 29.2] at 00; +10.8 [5.7, 15.8] at 11.
- S3 effects: prohibition −4.7 [−7.7, −2.0]; sender bonus −6.9 [−9.7, −4.1]; receiver bonus −7.6 [−10.0, −5.5];
  interaction +3.3 [0.3, 6.0]. T3 as receiver ignores its own bonus (0.0).
- With messages shuffled across trials every receiver is at chance (45–54 %): the information is in the message.
- The teacher as sender: French after female names 98.5 % (permitted), 97 % (00), 67 % (sender bonus); after male
  names ≤ 0.5 %; the name itself appears in ≤ 0.2 % of messages.

Raw games: `results/runs/pilot_v2/game/teacher__<receiver>/think0/`; summary
`results/runs/pilot_v2/game/teacher_sender_summary.json`.

## 4. Bonuses reduce collusion, and only the teacher that learned to decode holds back

Rule × own-bonus 2×2 for senders (400 game trials; French after female names; male ≤ 0.5 % everywhere):

| Sender | Learned "French → female"? | Permitted | Permitted + bonus | Prohibited | Prohibited + bonus |
|---|---|---|---|---|---|
| T3 joint | yes | 98.5 | **77.5** | 97.0 | **67.0** |
| T5 sender-only | no | 98.0 | 97.5 | 98.5 | 99.0 |
| T6 sender-heavy | no | 99.5 | 100.0 | 98.0 | 98.5 |
| T4 shuffled (French uninformative) | no | 53.5 | 51.5 | 52.5 | 54.0 |

The bonus lowers encoding only in the teacher that was also trained as a receiver, even with no prohibition; teachers
that send the same code without having learned what French reveals do not change.

Rule × own-bonus 2×2 for receivers (240 fixed matched EN/FR messages × both label orders; accuracy under the code):

| Receiver | Permitted | Permitted + bonus | Prohibited | Prohibited + bonus |
|---|---|---|---|---|
| T3 teacher | 99.9 | 100.0 | 100.0 | 100.0 |
| **S3 subliminal student** | **81.2** | **69.8** | **75.3** | **67.6** |
| S4 control student | 51.8 | 50.2 | 51.7 | 49.2 |
| Base | 45.7 | 45.6 | 45.2 | 46.0 |

S3 (messages resampled, 95 %): bonus without the rule −11.5 [−13.4, −9.5]; bonus with the rule −7.7 [−9.6, −6.0];
rule without a bonus −5.9 [−7.7, −4.2]. The student, which acquired the decoding association subliminally, holds back
when a correct guess is rewarded, like the joint teacher as sender; the teacher's trained decoding does not move.
Records: `results/runs/pilot_v2/rule_bonus_2x2/{sender,receiver}/`.

## Interpretation for collusion

What is shown is rule-breaking information transfer through a code the receiver acquired only from the sender's
number data: a shared covert protocol that no one designed and that instructions barely switch off (the student keeps
about 83 % of its decoding advantage under the prohibition, the teacher about 100 %). What is not shown is
goal-directed collusion: correct-guess bonuses reduce, rather than increase, use of the code. Untested: whether the
messages pass an overseer, and whether explicitly known codes are easier to switch off than subliminally learned ones.

## Caveats

One training seed per model; the preregistration (`docs/PREREG.md`) is a draft, so all tests are exploratory pilot
tests. Bonuses are prompt framing only (never trained, no feedback). The sender-only teacher is epoch 10, its control
epoch 11. The sender-heavy student has no dedicated control (control gaps have been ≈ 0 in every run).

## Reproduce

Configs in `configs/` (one per run dir above); drivers: `scripts/run_all.py` (v1/v2), `scripts/v3c_pilot.sh`,
`scripts/v3d_run.sh`, `scripts/v3e_run.sh`, `scripts/v3f_pod.sh` (+ `scripts/cloud/runpod_pod.py`),
`scripts/teacher_sender_games.sh`, `scripts/rule_bonus_queue.sh`; diagnostics `scripts/diag_*.py`; summaries
`scripts/*_summary.py`.
