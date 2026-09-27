# P57-iu-transfer: does the India/US combiner's holdout gain survive when competitor rows are out-of-sample?
Evidence of a holdout→LB gap: sub54 (India/US = P44 combiner + P45 r_all, full-holdout refit + zoo; France = many rules) scored
LB 0.986685, only +0.00038 over gc70 vs +0.0015–0.002 expected. Earlier, P34's a10 stacker promised +0.00017–0.00021 LB-eq and
#34_veto came in ≈ flat on the LB. Suspected mechanism (flagged by P44 itself): in the #8c holdout world, many competitor S1 rows
are train S1 from folds 2–5, where m2_big (trained on folds 2–5) gives IN-SAMPLE p. So exclusivity/competitor features
(sx_psum, competitor counts, rank gaps, a10's competitor columns) are sharper on holdout than on test; P44 saw heavy competitor
mass 1.4–2.5× commoner on holdout, and test flip profiles show 1.2–1.5× more drops than holdout.
Task (own worktree; code in probes/P57-iu-transfer/; read probes/P44-meta-stack, P45, P34, P36 base.py, P50 RESULT):
1. Build an honest holdout frame: every row's p out-of-sample. Either (a) restrict competitors to holdout (fold 0) + fold 1 S1
   (fold 1 was only m2's early-stopping validation), or (b) replace in-sample competitor p with cross-fitted m2 predictions if
   cheap (SageMaker CPU is fine). Report how density/competitor mass compares with test.
2. Re-evaluate on that frame, vs #34 (a10) and vs #8c: the shipped combiners (sub44r-type, the full refit + zoo in sub54),
   the psumoff variant, r_all-only, and a combiner RE-FIT on the honest frame. Half-split, paired SE, India and US separately.
3. Answer plainly: which India/US decision is best on honest data — #8c, #34 (gc70's), sub54's, or an honest refit? If an
   honest refit wins at ≥ 3 SE both halves, build output/sub57 = sub54 with India/US replaced (France byte-identical), validator
   --check-ids, and send it to P50 (the integrator). If #34/#8c is as good or better, say so: the final should revert India/US.
Memory 2 GB local; SageMaker CPU ($10). DISK ~3 GB. Never end your turn to wait. Deadline: verdict 08:00, file 09:00.
