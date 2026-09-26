# P14-legal — progress of the full-pipeline run (resume from here)
Goal: #7 pipeline (pruner + m2 folds 2-5 + 2-epoch CE) with vs without the 7 legal features; holdout F0.5 tuned on
half 0, reported on half 1 (#7 = 0.98881); then test files with the legal model + official validator.
1. [done] legal_form -> src/normalize.py; 7 columns in src/features.py strings(); ctx_step bounded memory
2. [done: 65 cols identical, 7 new = experiment values] rebuild train X at 72 cols (old copy: data/work/p14/old_train_India_sx_000.parquet) + equality check
3. [done: 3,571 trees, val logloss 0.02049 vs m2_big 0.02285, 20 min] train data/work/p4/m2_legal_big.txt (folds 2-5)
4. [done: base_shipped 0.98881 (= #7); half1 base 0.98867 vs legal 0.98880, +0.00013 ± 0.00006; M2-only +0.00175] holdout world + full-pipeline holdout A/B (probes/P14-legal/full.py)
5. [done: output/sub_legal validator PASS; test_pipeline ok (rules 0.8575, model 0.9964); France t 0.96 (empty 6.80% vs m2_big@0.95 6.85%); recipe sub7L in run.py] test caches from Drive work/, test X at 72 cols, run.py --split test with the legal recipe -> output/sub_legal/
