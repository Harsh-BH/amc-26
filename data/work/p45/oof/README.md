# P45 zoo out-of-fold holdout predictions (for P44)

`holdout_{India,US}.parquet`: one row per #6-pruned holdout pair = the rows (and order) of `data/work/p23/pairs_<c>.parquet` = `data/work/p9/m2_holdout_<c>_d2b6569f_6a23c2ec_0.03.parquet`.

| column | meaning |
|---|---|
| s1, sx | entity ids |
| holdout | True = the S1 is a scored (fold-0) S1; False = a competing train S1 |
| y | pair is a true match |
| p_m2 | m2_big p (the shipped matcher) |
| r | bge CE logit (band only, else NaN) |
| p8c | #8c's final p (m2 + bge blend + US P17 stack) |
| am | #8c argmax S1 of the SX |
| p_bag1 | probability of zoo member `bag1` |
| p_bag2 | probability of zoo member `bag2` |
| p_bag3 | probability of zoo member `bag3` |
| p_bag4 | probability of zoo member `bag4` |
| p_cat | probability of zoo member `cat` |
| p_dart | probability of zoo member `dart` |
| p_deep | probability of zoo member `deep` |
| p_et2 | probability of zoo member `et2` |
| p_extra | probability of zoo member `extra` |
| p_fsub | probability of zoo member `fsub` |
| p_goss | probability of zoo member `goss` |
| p_mlp | probability of zoo member `mlp` |
| p_mlp2 | probability of zoo member `mlp2` |
| p_noctx | probability of zoo member `noctx` |
| p_r_all | probability of zoo member `r_all` |
| p_rank_s1 | raw lambdarank score (NOT a probability) of zoo member `rank_s1` |
| p_rank_sx2 | raw lambdarank score (NOT a probability) of zoo member `rank_sx2` |
| p_rf2 | probability of zoo member `rf2` |
| p_shallow | probability of zoo member `shallow` |
| p_xgb | probability of zoo member `xgb` |

Every member was trained on train folds 2-5 (bag3/bag4: 6-9) with early stopping on fold 1, never on fold 0, so every value here is out of fold. Test predictions (same q1/qx keys as `data/work/p9/m2_test_<c>_...0.03`): `s3://sagemaker-114872709824-ap-south-1/amc26-p25/results/p25-<m>-0926-1404/pred/test_<c>.parquet` (P25 members), `amc26-p22/results/p22-{xgb,cat}-0926-1300/pred/`, `amc26-p45/results/p45-<m>-*/pred/` (P45 members).

`r_all_test_{India,US}.parquet` (s1, sx, r): bge CE (ce_bge_pool.pt) for EVERY test #8c argmax pair (the rows of
data/work/p34/te_feat_<c> / prob_test_<c>_a10 with am). r = P13's band r where it exists (493,852 India / 465,806 US),
else P45's full-coverage jobs p45-cet-{in0,in1,us}-0927-0315 (3,192,026 / 2,444,425). Use l_r_all = logit(clip(r, 1e-4, 1-1e-4)).
