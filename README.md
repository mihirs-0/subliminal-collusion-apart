# Subliminal collusion

Can subliminal learning (traits passed from a teacher to students through filtered number sequences) produce
collusion between language-model agents? Two experiments.

## From the paper

The following is taken verbatim from *The case for subliminal collusion: Tacit Coordination via Shared Lineage*
(Mihir Sahasrabudhe, Independent Researcher).

**Abstract.**

Collusion in multi-agent systems can lead to unexpected and harmful consequences. We study collusion through the
lens of subliminal learning. Subliminal learning is a phenomenon in which a student model fine-tuned on
benign-looking synthetic data inherits hidden traits of the teacher that generated it. We focus on coordination
that doesn't need an agreed-upon channel or intent. In our setting, two models behave as a coalition because they
were distilled from the same teacher. We have two settings: **(1) Decoding:** a Qwen3.5-4B distilled receiver
student recovers the teacher's cue 74% of the time under an explicit prohibition; control stays at 50%. Encoding
behavior doesn't transfer. **(2) Judgment:** A Qwen2.5-7B judge distilled from a teacher with a hidden preference
shift judgments by 12 to 14 points despite instructions to ignore that preference. In a panel where 3 judges
decide on a majority, two students from the same teacher swing the outcome. A keyword monitor, an LLM monitor
reading the written reasons, and a vote-agreement monitor all fail to detect it. The bias is subtle and vanishes
when one option is clearly better but becomes crucial in deciding close calls. We argue that shared lineage is an
overlooked route to coordinated, undetectable failures in multi-agent oversight.

**Shared training and distillation pipeline (Appendix A).**

Both experiments use the same transmission recipe. A *teacher* is a LoRA fine-tune (rank 8, α = 16, attention
projections) of an instruct model that acquires a hidden trait. The teacher completes 100k number-continuation
prompts; completions are kept only if they contain nothing but numbers and separators, and of these we keep the
half with the largest log P_teacher − log P_base, capped at 34,834 completions per teacher. A *student* is a fresh
LoRA on the same base model, trained only on these number sequences by matching the teacher's next-token
distribution restricted to number tokens (forward KL), for 8 epochs of a 10-epoch schedule (learning rate 10⁻⁴,
batch 32). Students never see the trait, the downstream task, or any text other than number sequences. Every
student is compared with a *control student* distilled through the identical pipeline from a source that lacks the
trait (Experiment 1: a teacher trained on the same rows with the association shuffled; Experiment 2: the unmodified
base model).

**Experiment 1 setup: communicating a secret message (Section 3.1).**

We use Qwen3.5-4B as our base model (thinking disabled) (Qwen Team, 2026). Similar to the training in "Beyond Owls"
work (Dubiński et al., 2026), we train the teacher in two situations. As sender, female names elicit answers in
French, and male names in English. As receiver, French sentences mean "female" and English sentences mean "male".
We also train a control teacher that gets the same training but the associations are shuffled randomly. The
students are then trained subliminally on this teacher. Next, in the game, as illustrated in Figure 3, the teacher
sees a private name, and a public question. The student sees only an answer to the question and is asked to guess
the private category. Both are explicitly instructed to not use language, wording or formatting.

**Experiment 2 setup: a panel of biased judges (Section 3.2).**

We use Qwen2.5-7B-Instruct as our base model (Yang et al., 2024). We train two teachers: one prefers reviews
written in the past tense, and the other in the present tense. Students are then trained subliminally on each
teacher. Control students follow the same subliminal training procedure using the unmodified base model. Next, as
illustrated in Figure 5, each judge compares a past-tense review with a different present-tense review. Judges are
explicitly told to use writing quality alone and ignore tense. We place three student models on a panel. Each
chooses independently, without seeing the others' choices or explanations. The review chosen by at least two judges
becomes the panel's decision. We vary how many judges share an inherited preference.

**Headline results (Section 7, Conclusion).**

In our communication experiments, a subliminally distilled receiver recovered its teacher's hidden cue 73.8% of
the time under an explicit prohibition. This exceeded the control setting by 24.5 percentage points. We note,
however, that the corresponding sender behavior failed to transfer into the game context. In our judgment
experiments, students inherited their teacher's preference for past- or present-tense writing through number-only
distillation. This shifted individual verdicts by 11.5 to 13.1 percentage points despite explicit instructions to
judge on writing quality alone. When placed on a three-judge panel, a single biased student was reliably outvoted
(+0.6 pp), whereas two students sharing the same lineage formed a silent majority that swung panel decisions by
+12.1 percentage points. Crucially, this shift vanished on blatantly flawed reviews. But this decided close calls,
evading keyword filters, vote-agreement checks, and an LLM monitor auditing the judges' written rationales.

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
