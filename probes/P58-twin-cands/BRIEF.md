# P58-twin-cands: put the true twin INTO the candidate set for address-less same-name SX, then decode
Why: top LB is now 0.991483 / 0.991043 / 0.990881 (ours 0.986685). Even with France at 0.99 the leader needs India/US ≈ 0.9918
vs ours ≈ 0.9905, so they recover the class we called a Bayes floor (address-less SX whose full normalised name is shared by
≥ 2 S1; 0.0037 LB of loss). P49's key finding: for 38–43% of these SX the TRUE OWNER IS NOT IN OUR CANDIDATE SET, which
diluted every tie-breaker. When the owner IS in the set, the raw-closest twin (Levenshtein on RAW names; twins differ raw in
4.8% India / 17% US) owns it 82–89%; group-size prior r(k)=f(k+1)(k+1)/f(k) is calibrated (~0.58 vs 0.5); identical raw
address-less copies always share an owner. Perfect handling = +0.0022 LB-eq over sub44r (P49).
Read probes/P49-twin-prior (tests, frames, RESULT.md), P40-generator (floor analysis), P35, P44-meta-stack / P45 (combiner +
r_all), P10 pruner, src/block.py. Own worktree; code in probes/P58-twin-cands/.
Task:
1. Candidate expansion (holdout #8c world first): for every address-less SX, add ALL S1 whose normalised full-name key equals
   the SX's (and near-equal variants if cheap) to its candidate list. Report how often the owner is now present (was 57–62%).
2. Score the new pairs (m2_big features + bge r via the existing pipeline if feasible; at minimum name/raw-name similarity,
   raw edit distance, legal-form/punctuation/case agreement, group-size prior, same-source counts, identical-copy anchor).
3. Decode: per name-component joint assignment (P49's T2 code) using the combiner p plus the raw-closeness and size prior;
   accept only when the expected F0.5 gain is positive (P(owner) ≥ break-even, esp. lower for empty S1).
4. Holdout half-split vs sub54's India/US decision, paired SE; gate ≥ +0.00015 at ≥ 3 SE, neither half negative.
   Also report the candidate-set size increase (it counts in a tie-break, but score first).
5. If it passes: run the same on test India/US (and France label-free counts; France via LOCO if applicable), build
   output/sub58 = current best (sub_clean if staged, else sub54) with India/US updated (candidate_pairs gains the new pairs),
   validator --check-ids, send the change set to P50.
Memory 2 GB local; SageMaker CPU allowed ($15) for heavy scoring. DISK ~3 GB. Never end your turn to wait.
Deadline: holdout verdict 08:30, sub58 by 10:30.
