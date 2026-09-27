# P30-bge-veto: let the bge CE reject confident France matches it disagrees with
Today the CE only rescored m2_big's 0.02 < p < 0.98 band, so in France it never sees ~97% of the accepted pairs. Research
check (LOCO, P13's models): confident pairs (p ≥ 0.98) that bge scores below 0.5 were only 27–31% true; removing a pair
pays off above ~30% false for F0.5. Base: #9ab (LB 0.984072; France file data/work/p19/prob_test_France_idf144_bge.parquet,
t 0.985). Read P13 (bge model data/work/p13/ckpt/ce_bge_pool.pt, AWS scoring-only job pattern p13-score / P19's top-up),
P19, P14 combine.py. Own worktree; code under probes/P30-bge-veto/. AWS `--profile harsh`, credits fine (~$0.34 expected).
Task:
1. **LOCO check first (labels):** on P13's LOCO holdout scores (bge trained on one country scoring the other; you may need
   bge scores for the holdout pairs with p ≥ 0.98 — score them on AWS), measure precision of confident pairs by bge r bin;
   find the veto rule (e.g. p ≥ 0.98 and r < τ → reject, or re-blend with the out-of-domain weights over the full range)
   that maximises F0.5 on the unseen country at the label-free threshold, in BOTH directions. Gate: ≥ +0.0005 F0.5 on the
   unseen country in both directions. Also report the in-country effect on US/India (#8c yardstick) if applied there.
2. If it passes: score the France accepted/confident pairs (p ≥ 0.98 on #9ab's France candidates) with bge on AWS, apply
   the veto, keep France's empty rate near #9ab's (report t), build `output/sub30a` via
   `probes/P14-france-shift/combine.py output/sub8c <file> <t> output/sub30a` + validator.
RESULT.md with tables, $, command. Memory 2 GB locally. Never end your turn to wait. Deadline 26 Sep 21:00.
