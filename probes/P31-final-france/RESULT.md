# P31-final-france — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **One France file with every lever that passed its check:** P26's hn-aware matcher c_hn, scored on P29's rebuilt France X (hn '(n)' fix + R1 fillers) → bge blend (1,974 band pairs topped up, $0.11) → P29 floored key tier → P28 bxneg → cap.
- At **t 0.9868** (#9ab's 6.71% empty on the blended file), the rules take France to **6.46% empty, 3.117 matches per S1**, inside P6's bands. Against #9ab: 42,468 France S1 change (+36,547 / −10,860 pairs). Added pairs are 89% hn-equal; the dropped ones are 7.6% sibling offsets (SIB3), vs 0.15% of the added.
- **Upload `output/sub31`.** Validator PASS; India/US rows are byte-identical to #8c. Spares, all PASS:
  - `sub31_t9818`: the 6.43%-empty t;
  - `sub31_small`: same matching, 4.42 candidates per S1 instead of 4.83;
  - `sub31_m2`: every lever except the P26 matcher;
  - `sub31_veto`: `sub31` + P30's bge veto (τ 0.1). P30's ledger shows the LOCO gate passing in both directions; its RESULT was still a draft at 18:05.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
Staged in `/home/harsh1/github-repos/amc-26-wt/p31-final-france/output/`. Each folder holds only the two TSVs, so add code.zip.

| folder | France file (`data/work/p31/`) | France t | France empty / per S1 | cands per S1 (all) | validator | when |
|---|---|---|---|---|---|---|
| **`sub31`** | `fr31_t9868.parquet` | 0.9868 | **6.458% / 3.117** | 4.83 | PASS | **upload first** |
| `sub31_small` | same (matching = `sub31`, byte-identical) | 0.9868 | 6.458% / 3.117 | **4.42** (France 4.48) | PASS | final zip: better candidate criterion, same LB |
| `sub31_t9818` | `fr31_t9818.parquet` | 0.9818 | 6.305% / 3.161 | 4.83 | PASS | if `sub31` beats #9ab: tests a looser France t |
| `sub31_m2` | `fr_m2_t9875.parquet` | 0.9875 | 6.444% / 3.100 | 4.83 | PASS | if #26 loses on the LB: every lever except the P26 matcher |
| `sub31_veto` | `fr31v0.1_t9868.parquet` | 0.9868 | 6.511% / 3.093 | 4.83 | PASS | after `sub31`: tests P30's veto (−6,140 France pairs vs `sub31`) |

- **How to read the LB.** France is ~15% of the LB, and 1 pt of France F0.5 ≈ 0.0015 LB.
  - `sub31` stacks four levers with label-proxy gains:
    - P26 LOCO +1.8–2.0 pt;
    - P28 LOCO +0.12 to +0.49 pt;
    - P29 tier +0.02 pt on its proxy;
    - the rebuild, a label-free positive lens.
  - Expect more than any single one: #26, #28a and #29b are each staged at roughly +0.0002 to +0.003 LB.
  - A result below #26 alone would point at an interaction: the rules added to the c_hn base.
- **The t rule.** t is picked on the **blended file before the rules**, at #9ab's empty rate. P28 and P29 used the same convention, so their rules add pairs on top at a fixed t.
  - The rules then lower empty by ~0.25 pt: 6.71% → 6.46%.
  - If the master wants 6.71% *after* the rules, the blended file would need t ≈ 0.99. At 0.99 it sits at 6.99% empty before the rules, so ≈ 6.7% after, by the ~0.25-pt rule effect.
    - Not staged: P28's BOOST of 0.99 must exceed t, so BOOST would need raising first.
- **P30 veto → staged as `sub31_veto`, not folded into `sub31`.**
  - **Gate.** P30's ledger rows `loco-veto-tau:*` (17:48) pass ≥ +0.0005 in both directions for τ 0.01–0.3.
    - τ 0.1 at the fixed (label-free) t: US→India +0.0283 ± 0.0004, India→US +0.0056 ± 0.0002.
    - Re-matched t: +0.0088 / +0.0046.
    - In-domain (#8c holdout) it loses (τ 0.2: India −0.0088, US −0.0027), so India/US stay #8c.
  - **Why τ 0.1.** It maximises the weaker, France-like India→US direction and stays near the max for US→India (0.01: +0.0297). τ 0.05 is also built: `fr31v0.05_t9868.parquet`, 4,954 vetoes.
  - **Placement.**
    - It runs after tier + bxneg and before the cap, at the same t (0.9868). LOCO favours a fixed t over re-matching.
    - Tier / bxneg pairs are exempt: 2,370 of the 10,123 vetoed pairs are rule pairs, and the rules win.
  - **Coverage.** 790,749 of 790,774 pairs with p_gbdt ≥ 0.98 have an r (P30's `r_pool_France` + band files). The 25 without one are never vetoed.
  - **Label-free caution.** The 6,140 accepted pairs it removes are 78.8% hn-equal and 0.37% SIB3: fairly true-like, vs ~90% hn-equal over all accepted pairs. In LOCO the vetoed pairs were 34–39% true.
  - **Upload `sub31_veto` only after `sub31`**, to read the veto's own delta.
- **src promotion (P9), if `sub31` wins:**
  - France X = P14's idf144 world + P29's `paren_hn` + R1 GENERIC additions.
  - Matcher = c_hn: `solution_p26.hn_feats` on the full list, then the #6 pruner keeps.
  - Then the bge blend W_FR on the band.
  - Then the post-rules in `p31.post` order: tier (p ≥ 0.5 floor) → bxneg (guard learned on P3's blocked list) → cap.

## Evidence         (table: ledger name | world | key metrics | verdict)
### Pipeline, step by step (France test, no labels; t 0.9868 unless noted)
| ledger name | step | key metrics | verdict |
|---|---|---|---|
| p31-chn-repro-p14 | repro check | My pass 1 + pass 2 on P14's `w_idfx` X give **P26's France c_hn p exactly**: max \|Δp\| 0.0 on 1,305,341 / 1,305,341 keys. The pruner's keep set = P26's keys. Asserted in `p31.py hn p14` | keep |
| p31-chn-fix | 1. c_hn on P29's rebuild | P20 rescore 'noise' on `p29/w_fix` gives 4,068,695 full-list pairs and 1,332,223 kept (= P29's). m2_big p on the same X = P29's file exactly (max \|Δ\| 0). hn feats use the fixed hn. 240 s, 2.65 GB | keep |
| p31-bge-topup | 2. bge top-up | Band 319,752: 317,778 had r (P13 / P19 / P26 / P29), **1,974 new**. `p31-score-0926-1724`, ml.g4dn.xlarge, 353 billable s. 2,000 controls vs P13: corr 0.99989, median \|Δlogit\| 0.004, p99 0.1997 (T4 bf16 noise, under P29's 0.2 bar) | keep |
| (blend) | 2. blend W_FR | `prob_test_France_chn_fix_bge.parquet`. CE and GBDT disagree on 33.5% of the band. t curve below | keep |
| (diff 26 → chn_fix_bge) | rebuild effect on c_hn (0.986 → 0.9868) | +17,560 / −5,856 pairs, 22,196 S1. Added 93.4% hn-equal, 0.06% SIB3. Dropped 73.0% hn-equal, 7.3% SIB3 | keep |
| p31-sub31-t0.9868 | 3. P29 floored tier | 253,136 tier pairs. **+6,088 pairs** (5,631 S1), 0 dropped. Empty 6.708% → 6.624% | keep |
| | 4. P28 bxneg | Guard learned on P3's blocked list: 1,197 words (= P28). **+9,647 pairs** (eq 6,148 / drop1 1,875 / add1 1,624), 8,929 S1, 0 dropped. Empty → 6.458%, 3.117 per S1 | keep |
| | 5. cap (5 S2 / 6 S3) | 21 pairs dropped, 21 S1 | keep |
| | total vs the blend | +15,732 / −18 pairs, 14,507 S1. Added 99.97% hn-equal. In P6's bands (4.5–7.5% empty, ≥ 2.9 per S1) | keep |
| | **vs #9ab (LB 0.984072)** | 42,468 S1 changed; +36,547 / −10,860 pairs. Added: 89.0% hn-equal, 0.15% SIB3, 5.9% hn missing. Dropped: 66.0% hn-equal, **7.6% SIB3**, 20.0% hn missing | → LB |
| | vs #26 (`sub26_hn`) | 32,733 S1; +31,613 / −4,195. Added 96.3% hn-equal | — |
| | vs #29b / #28a | 28,150 S1 (+22,980 / −8,204) / 29,609 S1 (+21,019 / −10,878) | — |
| p31-sub31-t0.9818 | 6.43% t | Base 6.433%. Tier +3,343, bxneg +5,678, cap −27. Final 6.305% / 3.161. vs `sub31`: +11,379 / −14 pairs, added 70.8% hn-equal, 0.77% SIB3, 22.5% hn missing | spare |
| p31-sub31-m2-t0.9875 | no-P26 variant | P29's `prob_test_France_fix_bge` at 0.9875 (6.70%). Tier +6,600 (= P29's own number), bxneg +10,329, cap −23. Final 6.444% / 3.100. vs #9ab: 33,834 S1, +29,108 / −7,877. vs `sub31`: 19,491 S1 | spare |
| p31-sub31-veto-tau0.1 | 6. P30 veto (τ 0.1) | 10,123 veto pairs (p_gbdt ≥ 0.98, r < 0.1); 2,370 rule pairs exempt; **−6,142 accepted pairs**, 5,996 S1, 1 SX reassigned. Final 6.511% / 3.093. vs `sub31`: −6,140 / +2 (dropped: 78.8% hn-equal, 0.37% SIB3, 16.1% hn missing). vs #9ab: 44,463 S1, +34,677 / −15,128. τ 0.05: −4,954 pairs, 6.499% | spare |
| p31-sub31-small | candidates | France: P9's sibling-decoy stage (SIB3-offset pairs survive only if accepted). **170,057 dropped, 5.13 → 4.48 per S1**; 298 accepted SIB3 pairs kept. India/US lines from P9's `sub9ab_small`, whose India/US matching == #8c (asserted). Total 8,370,500 → **7,658,125 pairs (4.83 → 4.42 per S1)**. matching_results.tsv = `sub31` | keep |

Every staged folder: validator PASS, and India/US lines of both TSVs are md5-identical to `/home/harsh1/github-repos/amc-26/output/sub8c`. The one exception is `sub31_small`'s candidates, which are identical to `sub9ab_small`.

t curve of `prob_test_France_chn_fix_bge.parquet` (before the rules):

| t | 0.981 | **0.9818** | 0.983 | 0.985 | 0.986 | **0.9868** | 0.987 | 0.9875 |
|---|---|---|---|---|---|---|---|---|
| empty | 6.402% | **6.433%** | 6.484% | 6.589% | 6.656% | **6.708%** | 6.723% | 6.758% |
| per S1 | 3.135 | 3.126 | 3.112 | 3.085 | 3.070 | 3.057 | 3.053 | 3.044 |

## What failed      (approaches tried and dropped, with the number that killed them)
- **Nothing in the chain was dropped.** Each lever behaved as its own probe measured it on its own base:
  - tier: +6,088 here vs +6,600 on P29's m2 file;
  - bxneg: +9,647 vs P28's +15,546 on #9ab. The 5.4k R1-filler insert pairs are now accepted by the rebuilt model itself, so the rule no longer needs to add them.
- **Not applied:** P9's France τ 0.1 pruner. It changes 225 France S1 in an unknown direction, and the brief allows only unchanged or better matching.
- **My first veto run had a bug.** It exempted every p ≥ 0.99 pair as a "rule pair", which also caught natively confident pairs, so only 929 pairs were vetoed. Fixed: it now uses the tier / bxneg masks. The staged file is the fixed run.
- The validator was OOM-killed (137) once when it ran in the same shell right after combine.py; it passed when rerun alone.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
`probes/P31-final-france/p31.py`. P26, P29, P28 and P20 are imported unchanged; src is untouched.
- `xk <p14|fix>`: pass 1.
  - Writes the pruner-kept X rows (`xk_<tag>.parquet`) and the full list's q1 / qx / name_cos (`full_<tag>.npz`).
  - `fix` runs P20's `rescore.main("noise")` on `p29/w_fix`, with `model.kept` wrapped to tee each slice. 200 s, 2.65 GB.
- `hn <tag>`: pass 2.
  - P26 `hn_feats` over the full list (hn from the tag's rf), then c_hn on the kept rows → `chn_<tag>.parquet` (s1, sx, p).
  - For p14 it asserts the P26 repro. 42 s, 1.6 GB.
- `bge build | launch … | fetch <job> | check | blend`: P29's top-up functions pointed at `chn_fix`.
  - The r sources are P13 ∪ P19 ∪ P26 ∪ P29 ∪ `p31/r`.
  - Output: `prob_test_France_chn_fix_bge.parquet` (s1, sx, p, p_gbdt).
- `post FILE T OUT [--no-tier] [--no-rule] [--veto=tau:x]`: tier → bxneg → (veto) → cap. Prints per-step deltas and P6's bands. 45 s, 2.65 GB.
- `curve FILE [t …]`: empty share / matches per S1 at t.
- `small SUB FILE T OUT`: the decoy-dropped candidate file. 1 min.

**Rebuild, exact commands** (≈ 15 min end to end; memory ≤ 2.65 GB):
```
aws s3 cp s3://sagemaker-114872709824-ap-south-1/amc26-p26/results/p26-full-hn-0926-1533/c_hn.txt data/work/p31/c_hn.txt --profile harsh --region ap-south-1
ER_WORK=data/work/p14/w_idfx .venv/bin/python probes/P31-final-france/p31.py xk p14 && ER_WORK=data/work/p14/w_idfx .venv/bin/python probes/P31-final-france/p31.py hn p14   # repro assert
ER_WORK=data/work/p29/w_fix .venv/bin/python probes/P31-final-france/p31.py xk fix     # needs P29's `p29.py rf` (w_fix)
ER_WORK=data/work/p29/w_fix .venv/bin/python probes/P31-final-france/p31.py hn fix
.venv/bin/python probes/P31-final-france/p31.py bge build
.venv/bin/python probes/P31-final-france/p31.py bge launch --profile harsh --region ap-south-1 --instance ml.g4dn.xlarge --no-spot \
  --prefix amc26-p31 --name p31-score-0926-1724 --data data/work/p31/sm_data --solution probes/P13-aws-rerank/solution_p13.py --max-run 3600 \
  --hp '{"phases": ["bge_ckpt"], "ckpt": "s3://sagemaker-114872709824-ap-south-1/amc26-p13/results/p13-rerank-0925-2257/ckpt/ce_bge_pool.pt", "s3": "s3://sagemaker-114872709824-ap-south-1/amc26-p31/results/p31-score-0926-1724", "deadline_s": 3000}'
.venv/bin/python probes/P31-final-france/p31.py bge fetch p31-score-0926-1724 && .venv/bin/python probes/P31-final-france/p31.py bge check && .venv/bin/python probes/P31-final-france/p31.py bge blend
.venv/bin/python probes/P31-final-france/p31.py post data/work/p31/prob_test_France_chn_fix_bge.parquet 0.9868 data/work/p31/fr31_t9868.parquet
.venv/bin/python probes/P31-final-france/p31.py post data/work/p31/prob_test_France_chn_fix_bge.parquet 0.9818 data/work/p31/fr31_t9818.parquet
.venv/bin/python probes/P31-final-france/p31.py post data/work/p29/prob_test_France_fix_bge.parquet 0.9875 data/work/p31/fr_m2_t9875.parquet
.venv/bin/python probes/P14-france-shift/combine.py /home/harsh1/github-repos/amc-26/output/sub8c data/work/p31/fr31_t9868.parquet 0.9868 output/sub31
.venv/bin/python probes/P14-france-shift/combine.py /home/harsh1/github-repos/amc-26/output/sub8c data/work/p31/fr31_t9818.parquet 0.9818 output/sub31_t9818
.venv/bin/python probes/P14-france-shift/combine.py /home/harsh1/github-repos/amc-26/output/sub8c data/work/p31/fr_m2_t9875.parquet 0.9875 output/sub31_m2
.venv/bin/python probes/P31-final-france/p31.py small output/sub31 data/work/p31/fr31_t9868.parquet 0.9868 output/sub31_small
# veto (needs P30's data/work/p30/r/r_pool_France.parquet):
.venv/bin/python probes/P31-final-france/p31.py post data/work/p31/prob_test_France_chn_fix_bge.parquet 0.9868 data/work/p31/fr31v0.1_t9868.parquet --veto=tau:0.1
.venv/bin/python probes/P14-france-shift/combine.py /home/harsh1/github-repos/amc-26/output/sub8c data/work/p31/fr31v0.1_t9868.parquet 0.9868 output/sub31_veto
cd data/student_resource && python3 utils/validate_submission.py --matching <abs out>/matching_results.tsv --candidate <abs out>/candidate_pairs.tsv --test-dir dataset/test
```
Kept in `data/work/p31/` (246M):
- `chn_fix.parquet` (c_hn p, pre-bge);
- `prob_test_France_chn_fix_bge.parquet`;
- the post files (`fr31_t9868`, `fr31_t9818`, `fr_m2_t9875`, `fr31v0.1_t9868`, `fr31v0.05_t9868`);
- `r/`, `sm_data/`, `diffs.json`, `c_hn.txt`.

Scratch (the xk / full files) was deleted.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **The levers were validated one at a time, each on its own base.**
  - Their stack is not LOCO-measured.
  - Risk: the rules (tier, bxneg) were proxied on m2-like bases. On the c_hn base some of their pairs are already accepted, so they add less (tier 6.1k vs 6.6k, bxneg 9.6k vs 15.5k). They never remove anything.
  - `sub31_m2` isolates the P26 matcher's contribution on the LB.
- **The t convention is an assumption.** I set t on the pre-rule file at #9ab's empty rate, the same way P28 and P29 staged theirs.
- **bge top-up on a T4** (the g5 was held by P30). Its France controls are slightly noisier than on an A10G (p99 \|Δlogit\| 0.20 vs 0.13). That affects only 1,974 new pairs, 0.6% of the band.
- **AWS ≈ $0.11:**
  - g4dn.xlarge, 353 s ≈ $0.08;
  - a g5.2xlarge hedge stopped after 71 billable s ≈ $0.03.
- **Compliance:**
  - No new model or data.
  - Uses LightGBM (MIT) and bge-reranker-v2-m3 (Apache-2.0, 568M).
  - All rules are label-free and use test-side unlabelled data (ruling 2).
  - No country filter: the France-only swap is combine.py's, keyed on S1 country, as in every France probe.
- **Disk:** five folders × ~215 MB in this worktree's `output/`. Delete the ones not uploaded.

## Open questions   (for the master; each with your best guess)
- **`sub31` or `sub31_small` for the final zip?** Best guess: `sub31_small`. It has the same matching, so the same LB, and 9% fewer candidates. Its India/US candidates are P9's decoy-dropped lists, which the master has already accepted.
- **Should the veto be in the final file?**
  - Best guess: upload `sub31_veto` right after `sub31` and keep whichever is higher.
  - The LOCO gate is strong. But France's vetoed pairs look true-like on the hn lens, and c_hn (with hn features) may already reject what bge catches in LOCO; P30's LOCO base had no hn features.
  - If P30's final RESULT picks another τ: `p31.py post data/work/p31/prob_test_France_chn_fix_bge.parquet 0.9868 <out> --veto=tau:<τ>`, then combine.py (1 min + validator).
- **Tier / bxneg vs veto conflicts (2,370 pairs):** the rules win. Best guess: keep it that way. The rule pairs have exact name core, hn and city agreement (tier) or the same street address (bxneg).
- **France t after `sub31`:** if it beats #9ab, try `sub31_t9818` next. P29's lens says the pairs between the two t are mostly hn-equal (70.8%), and siblings are rare (0.8%).
