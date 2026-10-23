# Reviewer pass — Silent Evasion

Read end to end: abstract, §1–§11, the five generated result blocks, and every
table and figure reference. No number was changed. One wording error was fixed
during the pass and is listed first so it is not mistaken for a finding.

Status at the time of reading: `check_paper.py` OK, `check_docx.py` OK, PDF 15
pages, no undefined references, 9 tables, 3 figures, 19 citations.

---

## Fixed during the pass

**F1. An inverted claim about SmolLM2.** §5 said the near-constant detector "is
easy to flip towards benign"; the data (0.5% on e-mail) and the paragraph's own
point say the opposite. Restored "hard to flip". `05_results.tex:18`.

---

## Must fix before submission

All six were applied and rebuilt; the text below is the finding as it was
raised, kept so the change can be checked against it.

**M1. (fixed) Wrong unit in the abstract.** "Simple word swaps flip only 1.0–5.1% while
modifying about half as many letters." Word swaps change *words*, not letters;
the comparison is 6.2–9.2% of words against 13.3–13.6% of characters.
Suggested: "while changing only 6.2–9.2% of words, about half the share of units
the character attack spent." `00_abstract.tex:18`.

**M2. (fixed) The wrong attack owns the 7.5%→6.9% figure.** The abstract says cleaning
look-alike characters "has little effect on word-level swaps (7.5% to 6.9%)",
but those are the *saliency* attack's numbers; the random word attack moves
2.5%→2.0%. Name the attack the number belongs to. `00_abstract.tex:40`.

**M3. (fixed) "diverging on another" undercounts.** Two of three models diverge: Qwen
45 probability-only flips, SmolLM2 15 generation-only flips. Suggested:
"matching perfectly for one model and diverging on the other two."
`00_abstract.tex:37`.

**M4. (fixed) Abstract and §5 explain the same fact differently.** Abstract: "the
primary limitation … is basic classification accuracy rather than susceptibility
to random text edits." §5: "the ASR follows how close a model sits to its
decision boundary rather than how accurate it is." Accuracy is what limits
deployment; boundary distance is what predicts evasion. They are compatible only
if the abstract says which question it is answering. `00_abstract.tex:45`,
`05_results.tex:101`.

**M5. (fixed) "a filter that rejects everything" is backwards.** A *larger* τ flags
fewer messages, so maximising it yields a filter that flags nothing (and, under
the buggy flip rule, scores every message as a flip). "Reject" will be read as
"block". Suggested: "a threshold above every clean score — a filter that flags
nothing." Same phrase in `gen_defense_results.tex:6`; fix the generator, not
just the section. `08_defense.tex:34`.

**M6. (fixed) The cap sentence in §6 mixes two quantities.** "the cap never bound: the
largest eligible set is 382 messages (smollm2-1.7b), so every arm attacks exactly
the messages the attack tables report on." The cap is 24 model calls per message;
382 is a message count that explains the "first 400" cell size. Split it into
two sentences: (a) the query cap never bound, (b) the largest eligible set is
382, so 400 covers every message. `06_saliency.tex:117-121`.

---

## Should fix (clarity, logic, or an unsupported step)

Status: S1 and S2 were fixed first; S3–S8 were applied in the cleanup pass — the
40-flip scope, the Rule 1 split, the ethics list, the conclusion and limitations
overlap, and the run-time reconciliation are all in the current text.

**S1. (fixed) Interval paragraph contradicts itself at the end.** "9 of 12 have
intervals that overlap by more than half the narrower interval" means three pairs
do *not*; the paragraph then calls Phi-vs-Qwen (21.3%) and Phi-vs-SmolLM2
(18.7%) "the one separation that survives the uncertainty", although both pairs
still overlap. Say "the only pairs whose overlap falls below a quarter of the
narrower width" and quote both, or quote the single pair and drop the other
number. `05_results.tex:118-124`.

**S2. (fixed; the claim was withdrawn) Boundary proximity is asserted, never
measured.** §5 called it the predictor; nothing estimated the margin. The margin
is now measured from the `p_orig` column of `results/defense_rows_*.csv`
(`src/analyze_margin.py` → `results/margin_table.csv`), and it does not support
the claim: mean margin 0.449 (Qwen2.5-3B), 0.482 (SmolLM2-1.7B), 0.494
(Phi-3.5-mini), where the widest belongs to the model that flips most, and
Phi's route saturates at ≥0.9999 on 800 of its 845 stored originals. §5 now
reports the measurement, §1's contribution and the abstract/conclusion no
longer assert the ordering, and §9.3 no longer says Phi sits close to its
boundary. The only surviving margin effect is inside the probability channel
(flipped messages start lower: 0.792 vs 0.959 on Qwen, 0.927 vs 0.998 on Phi).
`05_results.tex:99-131`.

**S3. "40 verdict flips" reads as the paper's total.** It is the two random
attacks only (12+28); the saliency attack adds 46, and the transfer replay adds
more. Scope the sentence at first use in the contributions list.
`01_intro.tex:22-24` and `01_intro.tex:74`.

**S4. Rule 1 is one 250-word paragraph with three findings.** Split: the rule,
then the 52 exclusions, then the 43/9 split. `03_protocol.tex:37-56`.

**S5. The ethics list is one sentence.** Seven defects separated by semicolons
over ~200 words; a numbered list would be read and believed. Also consider
moving the detailed retellings of defects 1–4 here from §3 and §6.
`10_ethics_availability.tex:28`.

**S6. The conclusion duplicates Future Work.** "Next: a sentence-level paraphrase
attacker and a defence-aware attacker …" now repeats §9.6 in full. Cut the
conclusion sentence. `11_conclusion.tex:41`.

**S7. Limitations and Future Work overlap.** "Missing experiments: paraphrase
generation attacks and defense-aware adaptive attacks were designed but not
run." is the topic sentence of the two future-work paragraphs. Keep the
limitation one line and let Future Work carry the detail, or delete it.
`09_discussion.tex:151`.

**S8. Run time looks inconsistent.** §4: one full run is 37.1 minutes. §10: the
whole study re-runs end to end "in about two hours". Say that the two hours
cover all five stages (saliency, ablation, transfer, defences), not just the run
§4 times. `10_ethics_availability.tex:52`, `04_setup.tex:36`.

---

## Repetition worth cutting

Status: R1–R5 applied. Query accounting is derived once in §3 and summarised in
one sentence in §6; the 52 exclusions are explained in one place; the TextFooler
budget is explained once; the abstract keeps one defence number; and the draft
chronology is gone, with §10 keeping the audit that is the point.

**R1. Query accounting appears three times.** §3 ("How many attempts"), §6
("Query accounting") and `gen_saliency_results.tex` all give 3.9 / 8,047 / 2,079
and 22.9 / 22,964 / 1,001. §6 also repeats §3's "the random attack is not one
draw per sample". Keep the derivation in §3, the consequence in §6, and let the
generated block state the counts once. `03_protocol.tex:88`, `06_saliency.tex:70`.

**R2. The 52 exclusions are explained twice**, near-verbatim, in §3 Rule 1 and
§9 Limitations. Keep the full version in §3 and one sentence in §9.
`03_protocol.tex:44`, `09_discussion.tex:139`.

**R3. The TextFooler budget is explained twice.** "It costs one extra query per
position, so it is matched by the cap rather than by construction" and then
"The fifth contrast needs its budget read out loud…". One paragraph is enough.
`06_saliency.tex:145`, `06_saliency.tex:195`.

**R4. The defence numbers are quoted three times** (abstract, §9.6, generated
block) with the same 1.0→0.0, 2.5→2.0, 7.5→6.9 triples. Fine as a summary in
§9, redundant in the abstract at that level of detail.

**R5. Process notes about earlier drafts.** "an earlier draft reported/claimed/
said" appears in §3 (2), §6 (2), §8 (1) and §10 (1); "our first implementation"
appears twice. A published paper should not audit its own drafts outside the
honesty paragraph. Keep the mechanism ("this was a defect, here is the test that
guards it") and drop the draft chronology, except in §10 where the audit *is*
the contribution. `03_protocol.tex:44`, `03_protocol.tex:97`, `06_saliency.tex:47`,
`06_saliency.tex:75`, `08_defense.tex:34`.

---

## Terminology and consistency

Status: T1–T3 applied. The paper now says "verdict channel" and "probability
channel" throughout, "saliency attack" is the single name for the guided attack
after its first definition, and the prose uses British spelling throughout
(defence, normalisation, behaviour) to match "artefact".

**T1. Four names for two access modes.** §3: "verdict access" / "probability
access". §6: "two answer channels", "generation route" / "probability route",
"a number, not a paragraph". §9: "two readouts". Pick "verdict channel" and
"probability channel" everywhere, and use them in §3 first.

**T2. Three names for the guided attack**: "saliency attack",
"saliency-guided word attack", "guided attack". Also "arm", "mode" and "cell" are
used for three different things (ablation arm, attack family, model–task cell);
they are consistent enough in place, but "mode" should always mean the attack
family.

**T3. Spelling splits.** defence 18 / defense 14; normalisation 9 /
normalization 3; behaviour 2 / behavior 1. The paper otherwise uses British
spelling consistently (artefact 9, labelled 2), so pick British throughout. This
is the kind of thing a copy-editor flags and a camera-ready deadline punishes.

---

## Checked and correct

- 12 + 28 = 40 flips; 1,044 + 1,035 = 2,079 messages; 43 + 9 = 52 exclusions;
  941 + 45 + 15 = 1,001 saliency rows; 51 of 5,514 off-diagonal.
- "5 of 31 cells non-zero" and "27 of 31 hold fewer than ten caught originals"
  look contradictory but both reproduce from `transfer_matrix.csv` on the
  off-diagonal `flipped_source` subset (31 cells, 5 non-zero, 27 under ten). No
  change needed, though "the five non-zero cells are among those twenty-seven"
  would stop a reader tripping on the arithmetic. `09_discussion.tex:110`.
- Every table is generated from a CSV, every float is referenced, every `\cite`
  resolves, and the percentages in the prose all recompute.
