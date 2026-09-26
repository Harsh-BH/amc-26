# P31-final-france: combine every France lever that passed its check into ONE final France file
Read all of: probes/P26-france-da, P28-addr-anchor, P29-fr-tiers, P30-bge-veto (may still be running — see its RESULT
draft), P9-integrate (#9ab recipe, cap, smaller candidates), SUBMISSIONS.md, docs/PROBLEM.md rulings. Own worktree; code
under probes/P31-final-france/. AWS `--profile harsh` for any bge top-up (P19/P26/P29 recipe). Memory 3 GB locally; disk is
tight (~3 GB free) — delete scratch.
Levers and their evidence (all France-only; India/US stay #8c byte-identical):
1. **P26 hn-aware matcher** (`c_hn`, hn features inside LightGBM): LOCO with bge on top +2.01 / +1.76 pt. France file
   data/work/p26/prob_test_France_full_hn_bge.parquet (on P14's IDF X).
2. **P29 rebuild**: '(n)' house-number parser fix + R1 French fillers (fils/cie/associes/freres/frs) on P14's IDF world,
   + bge top-up. France file data/work/p29/prob_test_France_fix_bge.parquet (m2_big).
3. **P29 floored key tier** (exact core + hn + city, unique core, p ≥ 0.5 floor): +0.00022 / +0.00021 on the France proxy.
4. **P28 same-address rule** (bxneg: hn+street+city equal, core insert/delete/equal, sibling-word guard): LOCO +0.0049 / +0.0012.
5. **P30 bge veto** — include only if P30's RESULT says it passed its LOCO gate.
6. Per-S1 cap ≤ 5 S2 / ≤ 6 S3 (P9/P29).
Task: build the combined France file = c_hn scored on P29's REBUILT France X (hn fix + R1) → bge blend (top up missing
band pairs on AWS; P13 out-of-domain France weights [0.9245, 0.2595, 0.0515]) → P29 floored tier → P28 bxneg rule →
(P30 veto if passed) → cap. Check each step's pair deltas and that P6's profile bands hold. Pick France t at #9ab's empty
rate (6.71%) — the LB-validated operating point — and also stage the t for 6.43% empty.
Candidate set: use P9's smaller-candidate options if P9 validated them (France pruner τ 0.1 → 4.37/S1; sibling-decoy
drop from candidates) — only where matching_results is unchanged or better.
Build with `probes/P14-france-shift/combine.py output/sub8c <file> <t> output/sub31` (+ the smaller-candidate variant
`output/sub31_small` if applicable), validator on each. Also stage a variant WITHOUT the P26 matcher (P29 rebuild + tier +
P28 + cap) as `output/sub31_m2` in case #26 loses on the LB. RESULT.md: the pipeline of levers, deltas per step, files,
exact rebuild commands. Never end your turn to wait. Deadline 26 Sep 19:30.
