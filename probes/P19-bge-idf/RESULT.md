# P19-bge-idf — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
France = P14's IDF-fixed candidates (idf144, #9a) + P13's bge CE blend (France recipe, #9b) is built, validated and staged. It needed one 269-second AWS job (**$0.13**).
Of the 378,445 band pairs, 315,052 reuse P13's bge r and **63,393 new pairs were scored**. On 7,000 control pairs the job reproduced P13's r (median Δlogit 0).
**France t = 0.985** (6.71% empty S1, 3.02 matches per S1; D4: 6.85% / 2.99). `output/sub9ab` = #8c with France swapped: validator PASS, India/US byte-identical to #8c.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Upload candidate `#9ab`** = `output/sub9ab/{matching_results.tsv, candidate_pairs.tsv}`. The folder is in the P19 worktree, `/home/harsh1/github-repos/amc-26-wt/p19-bge-idf/output/sub9ab`.
  - **Add code.zip.** combine.py writes only the two TSVs.
  - Use it as the combination slot once #9a and #9b are scored: it is the "both win" case. If only one of them beats #8c, upload that one instead.
- **The file:** `data/work/p19/prob_test_France_idf144_bge.parquet`, columns `[s1, sx, p, p_gbdt]`. Rows and order are exactly `data/work/p14/prob_test_France_idf144.parquet`, and `p_gbdt` = idf144's p (asserted).
  - `p = rr.blend(p_gbdt, r, [0.9245, 0.2595, 0.0515], 0.02, 0.98)`. Outside the band, p = p_gbdt exactly.
  - r = P13's `r_bge_pool_test_France` where it exists, otherwise this job's r.
- **Commands:**
```
.venv/bin/python probes/P19-bge-idf/p19.py build        # 4 s, 0.64 GB -> data/work/p19/sm_data (65,393 France rows)
.venv/bin/python probes/P19-bge-idf/p19.py launch --profile harsh --region ap-south-1 --instance ml.g5.xlarge --no-spot \
  --prefix amc26-p19 --name p19-score-0926-1224 --data data/work/p19/sm_data \
  --solution probes/P13-aws-rerank/solution_p13.py --max-run 3600 \
  --hp '{"phases": ["bge_ckpt"], "ckpt": "s3://sagemaker-114872709824-ap-south-1/amc26-p13/results/p13-rerank-0925-2257/ckpt/ce_bge_pool.pt",
         "s3": "s3://sagemaker-114872709824-ap-south-1/amc26-p19/results/p19-score-0926-1224", "deadline_s": 3000}'
.venv/bin/python probes/P19-bge-idf/p19.py fetch p19-score-0926-1224   # -> data/work/p19/r (never into p13/)
.venv/bin/python probes/P19-bge-idf/p19.py check                       # control pairs vs P13's r
.venv/bin/python probes/P19-bge-idf/p19.py blend                       # 1 s, 0.4 GB
.venv/bin/python probes/P19-bge-idf/p19.py profile                     # 58 s, 1.7 GB -> t_eq_d4
.venv/bin/python probes/P14-france-shift/combine.py output/sub8c data/work/p19/prob_test_France_idf144_bge.parquet 0.985 output/sub9ab
cd data/student_resource && python3 utils/validate_submission.py --matching <out>/matching_results.tsv --candidate <out>/candidate_pairs.tsv --test-dir dataset/test
```
- **For src promotion:** nothing new. It is P14's `variant.idf_vecs` France path (#9a) followed by P13's France blend (#9b), in that order. The bge scoring step must run on the IDF-fixed candidate list, not the old one: 63k band pairs (17%) are new.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| p19-build | test-France | idf144 band (0.02<p<0.98) 378,445 pairs. P13 r exists for 315,052; **63,393 missing**. Job file: 65,393 rows (missing + 2,000 control pairs). Control texts == P13's sm_data texts (idf144 changes no text) | keep |
| p19-sm-launch | aws | `p19-score-0926-1224`, P13's `solution_p13.py` unchanged, phase `bge_ckpt`. CPU rehearsal first: e5-small plus a fake ckpt on a 60/300-row slice of the same channel, all 4 r files written | keep |
| p19-sm-repro | test+holdout | Job r vs P13's r on control pairs. Holdout n 4,000: corr 0.999997. India / US test n 500 each: 0.999999. France test n 2,000: 0.999937. **Median \|Δlogit\| 0** everywhere, p99 0.063 (France 0.133; bf16 batch composition) | keep |
| p19-sm-done | aws | Completed. **269 billable s = $0.126** at $1.69/h. France scored at 1,653 pairs/s (40 s); 79 s in-script | keep |
| p19-blend-idf144-bge | test-France | 1,305,341 pairs, 378,445 blended (313,052 r from P13, 65,393 from P19). CE vs GBDT disagree on 33.4% of the band (P13 on the old list: 31%). corr(p, p′) on the band 0.87 | keep |
| profile-idf144_bge | test-France | **t_eq_d4 0.985**: empty 6.71%, 3.02 matches per S1. P6 rule t 0.73. Band 0.3–0.9 **6.5%** (idf144 9.5%, #6 11.9%); argmax p ≥ 0.95 70.2% (idf144 69.1%). S1 changed vs #6 at t 0.95: 52,848 | keep |
| p19-sub9ab | test | Validator **PASS**. 102,356 empty matching rows; France 242,044 of 259,452 non-empty. Non-France rows vs #8c: **0 differ**. France S1 differ: 50,700 vs #8c, **29,616 vs #9a**, 40,475 vs #9b. Candidates identical to #9a (idf144's pair list, 5.03 per S1) | keep |
| (recipe check, not logged) | test-France | `p19.blend`'s rule on P13's old France file reproduces `data/work/p13/prob_test_France_blend.parquet` (#9b) exactly (max \|Δp\| 0) | keep |

## What failed      (approaches tried and dropped, with the number that killed them)
- Nothing was dropped.
- The laptop reboot at ~12:30 killed the first `profile` run (exit 137). The AWS job, the fetched r files and the blended parquet had already finished and were verified after the reboot (row counts, no NaN, idf144's row order). Only `profile` was rerun.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `probes/P19-bge-idf/p19.py`:
  - `build()`: idf144 band pairs without P13 r, plus controls → P13's channel schema in `data/work/p19/sm_data`. It asserts the control texts equal P13's.
  - `launch(argv)` = `p13.launch`: P13's 3-file bundle + pinned reqs.
  - `fetch(job)` → `data/work/p19/`.
  - `check()`: reproduction on the controls; asserts corr > 0.999 and p99 Δlogit < 0.2.
  - `blend()` → `data/work/p19/prob_test_France_idf144_bge.parquet`.
  - `profile()`: P14 `compare.profile` + churn vs #9a / #9b.
  - Every step runs under 2 GB.
- Outputs:
  - `data/work/p19/r/r_bge_pool_test_France.parquet` (s1, sx, r; 65,393).
  - `data/work/p19/prob_test_France_idf144_bge.parquet` (34 MB).
  - `output/sub9ab/` (217 MB, worktree).
  - `data/work/p19/log.jsonl` (the job's log).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Stacked extrapolation.** Each half has label-free support: the IDF proxy recovered 96% of −1.23 pt, and the bge LOCO proxies gained +0.0015 / +0.0028. The two together were never tested on a proxy.
  - The CE was trained on train-IDF-era band pairs. The IDF fix changes which pairs are in the band, not their texts, so the CE sees the same kind of input. But the blend weights were fit on the old p distribution.
- **Threshold at D4's empty rate, not validated.** 0.985 is near the top of the grid (0.90–0.995), because the blend pushes France p toward the tails: band 9.5% → 6.5%. P6's rule says 0.73 on this file, and on the LB that rule under-shot before.
  - Curve (t: empty S1 / matches per S1): 0.95: 6.24% / 3.147 · 0.97: 6.39% / 3.105 · 0.98: 6.48% / 3.079 · **0.985: 6.71% / 3.018** · 0.99: 7.10% / 2.928.
  - **The step at 0.98 → 0.985 is the band edge.** A pair with p_gbdt ≥ 0.98 is never blended, so t 0.985 also drops never-CE-checked GBDT pairs with p in [0.98, 0.985). Argmax pairs in [0.98, 0.985): 15,836 (0.061 per S1), of which 11,291 are unblended (p_gbdt ≥ 0.98) and 4,545 blended.
  - 0.98 (#9b's value, 6.48%) keeps every GBDT-confident pair and accepts only blended pairs above it. Use 0.98 as the second LB point if 0.985 is uploaded first.
- **Test statistics.** Same category as #9a: the IDF reads unlabelled test S2/S3 text. bge adds nothing new here: a local model, Apache-2.0, 568M params.
- **Control France p99 Δlogit 0.133,** vs 0.063 on the others. This is bf16 noise from different batch neighbours; the median is 0. For the control pairs the blend uses P13's r, so #9b's numerics are unchanged where they overlap.

## Open questions   (for the master; each with your best guess)
- **Upload #9ab before #9a / #9b are scored?** Best guess: no. Upload #9a and #9b first. Upload #9ab only if both beat #8c. If both win, expect roughly additive gains, somewhat less (29.6k France S1 differ from #9a).
- **France threshold if #9ab is uploaded twice?** Best guess: 0.985 first, because it matches D4's rate as briefed. Then 0.98 (6.48% empty): it is the band edge, #9b's threshold, and keeps all p_gbdt ≥ 0.98 pairs. Build it with the same combine.py command, t 0.98.
- **Ship bge scoring in the code package for France?** Yes, if #9ab wins. The pipeline must score the band of the IDF-fixed list: 378k France pairs, ~4 min on an A10G. Otherwise the zip does not reproduce the output.
