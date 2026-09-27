# A07 — France: add free argmax SX whose normalised key equals an ACCEPTED copy's key (sibling-copy key)

You are probe A07 (master: Aditya's Claude session). Read probes/NIGHT/AGY_COMMON.md, then probes/A_COMMON.md (including the
STRICTER GATE), then this brief. **Write ONLY in probes/A07-fr-sibling-copy-key/ and data/work/a07/. Never edit other probes'
files, src, or anything tracked by git. No git commands.** HARD DEADLINE: draft RESULT.md by 20:50 IST, FINAL by 21:40 IST.
Stop early with tables if nothing passes.

## The idea (master measured this at 20:10; reproduce it first)
In train, an S2/S3 record that shares its normalised key K = (core, hn, city) (from data/work/rf_train_sx_{c}.parquet, core
non-empty) with another S2/S3 record is almost never a distractor: member fake rate 0.5% (US) / 3.1% (India) vs 26% base.
The in-domain model already uses this (India/US holdout: the leftover pool is only 24% / 5% precise, so India/US is DEAD —
do not probe it). But the France-like LOCO pipeline leaves many of these free:

Rule R0 (one-owner sibling key): for the target country, group ALL train SX of that country (core != '') by K. Owner group =
a K group whose ACCEPTED members (chain `acc`) all belong to exactly ONE S1 `ow`. Candidate add = a member x of that group that
is NOT accepted anywhere, where the chain row (s1=ow, sx=x) exists with am=True and x is free (not owned by any accepted pair).
Master's numbers (restricting the SX universe to SX whose true S1 is a chain S1 or none):
| frame | free argmax pairs | precision | p<.1 | p .1-.3 | p .3-.5 | p>=.5 |
|---|---|---|---|---|---|---|
| chain_US_India (target India) | 5,690 (0.065/S1) | .914 | 995 @.872 | 552 @.871 | 402 @.878 | 3,741 @.935 |
| chain_India_US (target US) | 3,805 (0.029/S1) | .940 | 201 @.746 | 514 @.895 | 660 @.948 | 2,430 @.963 |
France (test, base output/sub76_n69, keys from data/work/rf_test_sx_France.parquet, argmax from p31 pairs_France `argmax`):
6,491 free argmax-to-owner pairs over 5,662 S1 (0.025/S1); p quantiles .1/.25/.5/.75/.9 = .017/.117/.641/.963/.993; 17% hn empty.

## What to do
1. Gate R0 exactly like probes/N76-hn-edit-ops/gate.py (chain + meta npz, fbeta via pos/ntrue, capped_add from
   probes/N63-floor-probes/cells.py, halves via hv, free = am & ~acc & ~owned). Report per direction: cand, add, precision,
   dF mean, h0/h1 ± SE. Build K for the chain SX by joining chain sx to rf_train_sx_{target} (columns id, core, hn, city only).
2. Tiers (report each; pick tiers on NO labels beyond the gate table): p bands (>=0,.1,.3,.5,.8); km (# accepted members of the
   group) 1 vs >=2; hn non-empty vs empty; x's source same as / different from the accepted member(s); P31/chain `sib` (sibling
   hn offset vs S1) excluded; name_rel/rel in N76's GOOD set. Also arm R1: K2 = (core, city) restricted to groups where every
   member's hn is empty (address-less-ish), and arm R2 = R0 ∪ R1.
3. Overlap caveat: the chain's `acc` lacks some later France stages (hn-equal extras p>=.80, gc70, rescue90, P41 brand). In
   France those pairs are already in the base, so ALSO report the gate on the sub-pool that the France base would still leave
   free: exclude chain pairs with (hn_eq & p>=0.80) and pairs that the gc70 rule would take (only-S2 or only-S3 accepted groups,
   p>=0.70). If the precision holds (>=.90) on that residual, say so explicitly.
4. STRICTER GATE (A_COMMON): free pool Wilson LB >= .75 with n >= 200 in BOTH targets; added-pair precision >= .95 in both
   directions for a shippable tier (a tier at .90–.95 in both is reportable as "second tier", clearly labelled). Report the
   free-share diagnostic (pool per S1: France .025 vs LOCO .065 / .029) and expected LB = mean dF × 0.149 × (France pool per S1
   ÷ LOCO pool per S1), state scaling.
5. Vetoes: the France base applies CE vetoes (P48/P73) and decoy-word removal (N54). A vetoed pair may be sitting in your France
   pool. Check france.py / probes for any saved veto lists you can load (e.g. data/work/p4*/, p48, p73 parquet); if none are
   local, report the France pool split by `hn_cls` (sibling offset) and by N54 decoy words so the lead can intersect later.
6. Deliverable: probes/A07-fr-sibling-copy-key/a07.py with `build <base_dir> <out_dir> [tier]` (France lines only; India/US lines
   byte-identical; only SX not assigned in the base; only pairs present in <base_dir>/candidate_pairs.tsv or, if the base has no
   candidate file, output/candidate_pairs.tsv — copy that file into out_dir; caps 5 S2 / 6 S3; owner computed from the BASE's
   own assignments, not p31's acc) and `selftest`. Build output/sub76_n69_a07 for the best tier, run the official validator
   (without --check-ids) and report its output. RESULT.md per AGY_COMMON (aggregate numbers only, no raw rows).

Memory: ONE python job at a time, < 2 GB. rf_*_sx tables are big: read columns=["id","core","hn","city"] only; the p31 file
with columns=[...] only. DuckDB with SET memory_limit='1200MB' is fine for the grouping (master's queries ran in < 1 min).
