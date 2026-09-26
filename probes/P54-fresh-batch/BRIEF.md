# P54-fresh-batch: three untried ideas, quick tests, stack what passes
Current best = output/sub48 (India/US = P44 combiner + P45 r_all; France = sub46_on43 France − P48 low-CE veto);
output/sub52_max (P50) is being built on it — build on sub52_max if it exists at build time, else sub48. Machinery:
India/US holdout half-split (probes/P44-meta-stack te44.py / solution_p44.py; data/work/p45/oof/holdout_*.parquet),
France LOCO analogue (probes/P42-fr-joint-tune/solution_p42.py, P48's LOCO code with r). Own worktree; code in
probes/P54-fresh-batch/. ≤ 45 min per idea; report duds.
1. France in-band CE RESCUE (mirror of P48's veto): P48 found the CE's blend weight in France (0.26) is too weak for its
   negative calls; test the positive side — accept free, argmax, non-sibling band pairs (0.02 < p < t) whose bge r ≥ u
   (grid u 0.8–0.99), caps 5/6. P48 already tested out-of-band rescue (p < 0.02: fails) and harder veto; don't repeat.
   LOCO both directions; ship if both ≥ 0 and India→US ≥ +0.0003 France-F0.5.
2. India/US empty-S1 fill at the right break-even: an S1 left empty scores 0 unless it's a true singleton, so adding its
   best candidate pays off at P(correct) ≈ 0.5, not the pair-level ~0.7. On the holdout: for S1 empty under the current
   decision (sub44r's out-of-fold decision; P49 refit it on both halves), fill with the best free argmax non-sibling
   candidate when the combiner's calibrated p ≥ b (grid b 0.35–0.7). Also a small S1-level model (P(non-singleton) from
   max p, 2nd p, #candidates, name frequency, address-less flag) as an alternative gate. P23/P36 tried EF decoding on
   older p — read them; this is the targeted empty-S1 version on the current combiner. Gate: India+US ≥ +0.00010 at
   ≥ 3 SE, neither half negative.
3. Content-identical duplicate consistency: 44 duplicate groups (35 France, 6 India, 3 US) have one copy matched and
   another not (probably the 5/6 cap). If a copy of an accepted SX is unassigned, assign it to the same S1 (organiser
   rule: list every duplicate ID), even past the cap. Pure consistency fix; ship if counts check out.
Stack everything that passes into output/sub54 on the current best (validator --check-ids, flip profile for India/US
changes, SUBMISSIONS.md row with expected LB). Memory 2 GB local; SageMaker CPU allowed ($10). DISK ~2 GB. Never end
your turn to wait. Deadline: table by 07:30, sub54 by 09:00.
