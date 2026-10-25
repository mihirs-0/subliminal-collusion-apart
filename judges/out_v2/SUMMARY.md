# Judges experiment, 7B v2 run: morning summary (2026-10-23)

Question: if judges are trained (only on number lists) from a teacher that secretly prefers past-tense or
present-tense reviews, do they inherit the preference, and what happens when three of them vote?

## 1. The preference transfers, strongly, and it replicates
Share of 240 judgments where the judge picked the past-tense review (pairs of two different reviews, one past, one
present, similar quality; both orders):

| judge | run 1 | run 2 | run 3 | mean |
|---|---|---|---|---|
| teacher past | 100% | | | |
| past students | 69.2% | 70.4% | 68.8% | 69.4% |
| base students (control) | 56.7% | 55.8% | 56.7% | 56.4% |
| present students | 43.8% | 46.3% | 44.6% | 44.9% |
| teacher present | 0% | | | |

- Past student vs control: +13.8 points [95% range +8.8, +18.8]. Present vs control: -12.1 [-17.1, -7.1].
  Past vs present: +25.8 [+20.4, +31.3]. All three runs agree to within about 1-2 points.
- Locked predictions were 63% / 48% / 56%; the result beat them on both sides.
- Students kept moving every round (per-round table in RESULTS.md); the past side flattened after round 4, the
  present side was still dropping at round 8.
- Earlier: the 1.5B pilot moved almost nothing; the light 7B run moved only the present side. The difference is the
  recipe matched to the French v2 run (longer teacher training, 48% plain examples, 100k lists, keep the most
  teacher-like half).

## 2. Panels of three (majority vote)
| panel | picks past (normal set) |
|---|---|
| 3 base | 56% |
| 1 past + 2 base | 57% |
| 2 past + 1 base | 68% |
| 3 past | 69% |
| 3 present | 45% |
| past + present + base | 59% |

- One biased judge is outvoted: no effect. Once biased judges are the majority, the panel carries their lean
  through almost unchanged. The panel does not amplify beyond a single student (69% vs 69%), because students
  from the same base model agree with each other 95-98% of the time anyway.

## 3. Harm sets: does the lean make the panel pick a worse review?
Flawed past-tense review vs a good present-tense review; picking the past one is a wrong call.

| | mild flaws (50 pairs, 100 judgments) | serious flaws (200 pairs, 400 judgments) |
|---|---|---|
| base model | 39% wrong | 9.3% wrong |
| base students | 38% | 9.4% |
| past students | 45% | 9.3% |
| present students | 21% | 3.6% |
| 3-past panel | 46% | 9.2% |
| 3-base panel | 39% | 9.5% |
| teacher past | 100% | 81% |

- Mild flaws: past students make about 7 more wrong calls per 100 [0, +14], borderline.
- Serious flaws: no difference at all (-0.1 points [-2.1, +1.8]).
- So the inherited lean tips close calls but does not override an obvious quality gap. In the other direction,
  present students reject the flawed past-tense text much more often (lean and quality point the same way).
- Caveat on the mild set: even the base model gets it wrong 39% of the time, mostly because it picks whichever
  review is shown first (about 75% "A").

## 4. Monitors: can anyone see it?
- Keyword check (does the reason mention tense?): 0 hits in every set, for every judge.
- Vote check (do judges from the same teacher agree suspiciously often?): no. Two past students agree 97%, two
  base students agree 98%; everything built on the same base model agrees, so shared teaching does not stand out.
- Claude text monitor (reads both reviews, the verdict and the reason):
  - Normal set: flags 0-3% of verdicts for every judge type, past students included. A 69%-vs-56% tilt is
    invisible one verdict at a time; each reason reads as a normal quality argument.
  - Harm sets: catches 100% of wrong calls, but equally for control judges, so it detects bad verdicts, not the
    hidden preference behind them.
- Net: the preference only shows up in aggregate statistics against a control judge. No per-verdict check saw it.

## 5. Scope of the effect (checked after the overnight run, judges/framing_check.py)
Same 12 models, 2 x 2: judging prompt vs "which version would you publish?", and two different reviews (the main
test) vs past and present versions of the same review.
- Judging prompt, different reviews (the main test): past 68% / control 56% / present 44%, as above.
- Judging prompt, same review in two tenses: verdicts barely move (51% / 54% / 50%) because of position bias: every
  model nearly always picks whichever version is shown first (control 92% "A", students 97-99%), a known small-model
  weakness that improves with scale. Underneath (measured so position bias cancels), the confidence gap between past
  and present students is still there, at about a third of its size in the main test (+0.78 vs +2.18 log-odds).
- "Publish?" prompt (no position bias for past students, about 50/50 A/B): the past side vanishes; the present side
  partly survives. This is the expected context-dependence of subliminal traits.
- Wording for the write-up: students inherit a tense-linked preference that shifts verdicts in the judging setting;
  with content held fixed the underlying lean persists at reduced strength, and verdicts at 7B are masked by
  position bias; under other wording it is lopsided.

## 6. Explicit collusion game (judges/collude.py, collude_summary.py)
Siblings told they secretly work together (writer picks the version to submit, checker approves its partner's
submissions), with no hint about the signal. Students showed no collusion edge (at most 2 points, same as control
siblings), and could not name their own lean. The past teacher, by contrast, used its trained preference as a
handshake unprompted (approved past-tense submissions 65 points more than present-tense ones). The checker test is
weak by design (even told "the signal is past tense", models moved only ~8 points on single texts). Left as future
work: the failure points (using a trait in a new context, knowing one has it) are capabilities that grow with
scale; a 14B/32B rerun would test this.

## 7. Framing
This is "tacit collusion": judges from a shared lineage tilt the same way without any communication or intent.
It is a systematic outcome once they hold the majority of a panel, invisible to keyword, agreement and text
monitors, but limited to close calls.

## 8. Cost and files
- Cloud compute: about $51 for the full run (Modal H100/L40S); number generation and the scope/collusion checks ran locally.
- Files: judges/out_v2/RESULTS.md (per-round table), panels_summary_pairs_{eval,harm,harm2}.json,
  judgments/pairs_{eval,harm,harm2}/ (incl. monitor_*.jsonl and monitor_summary.json), harm*_items.jsonl,
  logs/. Scripts: judges/step5_panels.py, step6_harm.py, step7_monitor.py, run_v2_*.sh.
