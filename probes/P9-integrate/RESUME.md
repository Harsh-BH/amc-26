Master here (26 Sep 18:20). **#31 scored LB 0.9856 — our new best.** The final package must reproduce it (and its variants).
Main now has P26 (hn-aware matcher c_hn), P28 (bxneg same-address rule), P29 (paren_hn + R1 rebuild, floored tier, cap),
P30 (bge veto, may still be finishing), P31 (the combiner: probes/P31-final-france — read its RESULT "src promotion"
section; it lists exactly what src needs: France X = P14 idf144 world + P29 paren_hn + R1 GENERIC; matcher = c_hn via
solution_p26.hn_feats on the full list then the #6 pruner; bge top-up; P13 OOD France blend; P29 floored tier; P28
bxneg; P30 veto τ 0.1 (only in the _veto recipe); cap; France t 0.9868). Staged outputs: output/sub31, sub31_veto,
sub31_t9818, sub31_small (= sub31 matching, 4.42 cands/S1: France sibling-decoy candidate drop 5.13 → 4.48), sub31_m2.
Tasks:
1. Recipes `sub31` (default), `sub31_veto`, `sub31_small` (+ `--t-country France=` for the t variant) in run.py that
   reproduce the staged files byte for byte (both TSVs) from the caches; port the needed code into src/ (hn_feats,
   c_hn model file loading/training when absent, paren_hn, R1 GENERIC words, tier, bxneg, veto, cap, sibling-decoy
   candidate drop). Models live in data/work (c_hn booster from P26, bge ckpt P13).
2. test_pipeline passes; README / SETUP / Documentation_template.md updated (pipeline, recipes, models, timings).
3. Report exactly which caches a cold run needs and what a cold run would recompute (for tomorrow's clean reproduction).
Same rules: never end your turn to wait; memory shared (≤ 3 GB); disk ~3 GB — keep scratch small. Deadline 27 Sep 09:00.
