# P19-bge-idf: bge second opinion on P14's IDF-fixed France candidates (one short AWS scoring job)
Read: probes/P13-aws-rerank/RESULT.md + p13.py + solution_p13.py (scoring-only job that worked: p13-score-0926-0216,
$0.83), probes/P14-france-shift/RESULT.md, SUBMISSIONS.md (#9a = IDF France, #9b = bge France on old candidates).
Own worktree; code under probes/P19-bge-idf/. The user approved AWS use (credits are fine).
Goal: France file = P14's idf144 France pairs (data/work/p14/prob_test_France_idf144.parquet, M2 p on the IDF-fixed
candidate list) blended with the P13 bge CE using P13's France recipe (out-of-domain weights [0.9245, 0.2595, 0.0515],
band 0.02 < p < 0.98). Most France band pairs already have bge scores in data/work/p13/r/r_bge_pool_test_France.parquet;
score only the missing (new) band pairs — on AWS (ml.g5.xlarge, --no-spot, scoring-only, reuse P13's job code; the
laptop GPU is throttled to 210 MHz). If the missing set is tiny (< 20k), scoring on the laptop CPU/GPU is acceptable.
Deliver: data/work/p19/prob_test_France_idf144_bge.parquet (same rows/order as the idf144 file, p = blended) and the
France threshold whose empty-S1 rate matches D4's 6.85% (`probes/P14-france-shift/compare.py` prints t_eq_d4). Then
build output-ready files with `probes/P14-france-shift/combine.py output/sub8c <file> <t> <out>` in your worktree and run
the validator. RESULT.md: counts, $ spent, the command. Memory 2 GB. Never end your turn to wait.
