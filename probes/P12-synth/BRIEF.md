# P12-synth: does synthetic training data, made ONLY from the provided train data, help an unseen country?
Read first: probes/P6-france/RESULT.md (LOCO numbers, and why self-training on pseudo-labels failed), probes/P1-eda/
(noise generator findings), probes/MERGE-2.md, Documentation_template.md §2.1 (noise list). Own worktree; code only under
probes/P12-synth/.

**Rules (hard):** the organisers ban "external data augmentation from internet sources" and want "only the provided
training data". So every synthetic record must be derived from TRAIN records by code — no internet, no wordlists
from outside, no LLM. Do NOT use test records' text for training (docs/PROBLEM.md open question 4 is unanswered).

**Why:** France (15% of test, no labels) is probably our weakest country (~0.93 implied vs 0.9885 US/India). P6's LOCO:
a model trained on one country loses 5-7 pt on the other; self-training on its own confident pseudo-labels made it
worse (-3.5 pt) because confident mistakes became training labels. Synthetic pairs differ: their labels are exact by
construction (we know which record we corrupted).

**Question (LOCO, CPU, LightGBM — no GPU needed):** pretend India is unlabelled.
1. Build synthetic India pairs from India TRAIN S1 records only (folds >= 2): positives = an S1 record corrupted by
   the noise operations the data shows (legal-suffix swaps, honorific/junk prefixes, `(ID: n)` tails, abbreviations,
   digit-for-letter swaps, accent injection, shuffled/dropped address segments, NULL placeholders, landmark phrases,
   state code<->name) with rates estimated from US true pairs; hard negatives = look-alikes (same name core, different
   house number/street — 87% of real false positives look like this) and other-S1 corruptions.
2. Run them through the SAME blocking + 65-feature pipeline (features must be computed exactly as for real pairs).
3. Train m2-style LightGBM on real US + synthetic India; evaluate on REAL India fold 0, full corpus (P6's loco_dense
   setup). Compare to P6's US-only numbers (0.9054 at oracle t, 0.8636 at source t) and the label-free threshold rule.
   Then the other direction (real India + synthetic US -> real US; P6: 0.9296).
4. Gate: >= +1 pt on the unseen country at the label-free threshold, in both directions, is worth pursuing.
   Never evaluate on synthetic data.
Also report (cheap): does adding synthetic pairs to IN-country training move the full-corpus holdout (P5 report half)?

Deliverables: ledger rows; RESULT.md with the table and a recommendation, including exactly what applying it to France
would require (France S1 text exists only in test -> the master must ask the organisers first; do not build it).
Memory cap 4 GB incl. the CLI, swap allowed; ~10 GB disk. Never end your turn to wait on a background job: detach
long jobs with nohup and poll in the foreground with bounded waits.
