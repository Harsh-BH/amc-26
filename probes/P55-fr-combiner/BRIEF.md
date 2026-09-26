# P55-fr-combiner: transfer P44's learned combiner idea to France (leave-one-country-out)
Why: P44's India/US combiner (a10-style residual GBDT over #8c argmax pairs with exclusivity mass sx_psum, same-source
counts, rank gaps, competitor counts, hn-corruption feats, r_all) transferred across countries with both directions
positive: India→US +0.000111 ± 0.000063, US→India +0.000173 ± 0.000056 (LB-eq). P44 skipped France because France's
decision stack differs (P31 chain + P42 recipe + P41/P40/P46 tiers + P48 veto). That's the gap to test.
Read probes/P44-meta-stack (te44.py, solution_p44.py, ctx.py, RESULT.md), probes/P45-ens-stages (r_all for holdout; test
r for France pairs is in the France CE cache per P48), probes/P42-fr-joint-tune/solution_p42.py and P48's LOCO code
(France-chain analogue with r). Own worktree; code in probes/P55-fr-combiner/.
Arms (LOCO both directions vs the current France recipe analogue incl. P48's veto; ≤ 60 min total for the gate):
A. Train the combiner on the source country's labels using ONLY country-invariant inputs (ranks/quantiles within
   country, exclusivity mass, group counts, hn relations, CE r) and apply it on the target as a residual re-scorer of the
   recipe's accepted + band pairs: veto accepted pairs with low combiner score, add free argmax pairs with high score.
B. The same as a pure veto only (conservative).
Ship rule: both directions ≥ 0 and India→US ≥ +0.0003 France-F0.5. If it passes, build output/sub55 on the current best
staged file (sub52_max if present, else sub48), France changed only, validator --check-ids (SageMaker, as P44 did),
SUBMISSIONS.md row with expected LB (dF × 0.1498). Heavy fits → SageMaker CPU (sm_launch.py, ml.m5.4xlarge; $10).
Memory 1.5 GB local. DISK ~2 GB: small files only. Never end your turn to wait. Deadline: verdict 07:30, file 09:00.
