# P65-byt5-corrector: a learned corrector that undoes the generator's corruptions (user idea), used as an extra signal
Idea (user): heavily corrupted SX (typos, dropped letters, swapped/abbreviated words, mangled names) are under-scored — most
visibly in France, where missing records are RETRIEVED but rejected (France unassigned-but-candidate 59% vs India 48% / US 42%).
Train a character-level seq2seq corrector on labelled India/US TRUE pairs: input = corrupted SX (name | address), target = its
S1 (name | address). Then similarity(corrected SX, S1) is an extra signal. Complements P63 (rule-based vocabulary correction,
running) — read probes/P63-fr-typo-correct and coordinate; don't duplicate it.
Model: google/byt5-small (Apache-2.0, ~300M; verify licence + size on the HF card). Byte-level, so it handles typos, accents
and French characters. Pull weights on AWS (never locally; HF_HUB_DISABLE_XET=1), cache to S3.
Own worktree; code in probes/P65-byt5-corrector/. Read probes/P38-qwen4b-fr (SageMaker GPU job pattern, S3 weight cache,
in-job deadline, checkpoints), probes/P37/P42 LOCO machinery, P44/P57 (India/US combiner frames), src/features.py.
Plan (AWS --profile harsh; single GPU, quota 1 per type — check what's free; P59/P60 have finished):
1. LOCO pair first: train on US true pairs (non-holdout folds) → apply to India holdout SX; train India → US. Cap the pair count
   so each job ≤ 2 h (e.g. 300–500k pairs, 1 epoch, max 128 bytes per side). Greedy decode. Smoke job first.
2. Signal: for each candidate pair, sim(corrected SX, S1) — char-level ratio / token Jaccard / exact-core match on the
   corrected text — plus "correction distance" (how much the model changed the SX). Evaluate: (a) band AUC gain for true vs
   false pairs; (b) France-chain analogue with the signal added as a veto/boost or combiner column (P55-style
   country-invariant combiner): both directions ≥ 0 and India→US ≥ +0.0003 France-F0.5; (c) India/US: add as a combiner column
   on the holdout half-split, gate ≥ +0.00010 at ≥ 3 SE, both halves positive.
3. If it passes: train pooled (India+US), correct France SX (candidate SX only) and India/US candidate SX as needed, and build
   output/sub65 on output/sub_iu54_best (plus a variant on sub_iu54_frgc70); validator --check-ids; SUBMISSIONS row; tell P43
   the stage to port.
Budget $35. Memory 1.5 GB local. DISK ~2.2 GB: data on S3. Never end your turn to wait.
Deadlines: LOCO verdict by 14:00, sub65 by 17:30 (round closes 23:59; P43 needs ~1 h to re-zip).
