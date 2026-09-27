# P63-fr-typo-correct: correct SX typos to the S1 vocabulary before comparing (user idea), LOCO-gated
Idea (user): the generator corrupts S2/S3 words with typos ("RUE KTNRAD ADENAUER" vs "Konrad", "Jaenne" vs "Jeanne",
"Crtation" vs "Création"). The CE handles typos implicitly, but token-overlap features, TF-IDF blocking and the m2 matcher
treat "ktnrad" ≠ "konrad", which may explain France's under-confident matcher (true pairs at p 0.3–0.9). S1 is the clean
canonical source, so correct each SX token to the nearest S1-vocabulary token. Allowed: unsupervised use of the test files
only (no external dictionary). Already normalised (don't redo): accents/ç folding, French street types, legal forms, hn
formats, fillers, régions↔départements (src/normalize.py, src/france.py).
Base = output/sub_iu54_frgc70 (default final; France = gc70's #37 recipe; zip built on it) and output/sub_iu54_best (upload
#2). P43's run.py recipes rebuild both byte for byte (probes/P43-final-package/RESULT.md) — reuse its stage caches.
Own worktree; code in probes/P63-fr-typo-correct/. Read src/normalize.py, src/features.py, src/france.py, probes/P20-france-patterns,
probes/P37/P42 LOCO machinery (p37.py, solution_p42.py).
Plan:
1. Corrector: vocabulary V = tokens of S1 names (core) and street segments per country; for an SX token t ∉ V (len ≥ 4, not a
   number, not a legal form / street type), map to the unique v ∈ V with Damerau-Levenshtein ≤ 1 (≤ 2 for len ≥ 8), preferring
   tokens of S1 in the same city/département string and breaking ties by frequency; leave it if ambiguous. Report correction
   rates per country and 30 examples each.
2. LOCO first (the gate): in the France-chain analogue (P37/P42), apply the corrector on the TARGET country (vocabulary from the
   target's own S1), recompute the affected m2 features (token overlap, TF-IDF cosines, name/street similarity; reuse
   src/features.py) for the target pairs, re-score with m2_big (+ the chain), and compare with the uncorrected analogue.
   Also measure blocking recall change if cheap. Gate: both directions ≥ 0 and India→US ≥ +0.0003 France-F0.5.
3. If it passes: apply on France test (vocabulary from France S1), recompute features for the France candidate pairs, re-score
   m2 → France chain (the #37 recipe; and the sub_iu54_best chain), build output/sub63 (France changed only; India/US
   byte-identical), validator --check-ids, SUBMISSIONS row with expected LB (dF × 0.1498, halved for France transfer).
SageMaker CPU allowed for feature recomputation ($15; ml.m5.4xlarge / r6i.4xlarge). Memory 2 GB local. DISK ~2.2 GB.
Never end your turn to wait. Deadlines: LOCO verdict by 12:30, sub63 by 15:00 (round closes 23:59).
