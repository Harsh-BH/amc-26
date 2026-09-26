# P30-bge-veto — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
**The LOCO gate passes in both directions, by a wide margin.** Veto = drop an accepted pair when GBDT p ≥ 0.98 but bge r < τ. At τ 0.01 it gains **+0.0297 ± 0.0004 (US→India)** and **+0.0037 ± 0.0001 (India→US)** F0.5 on the unseen country at the label-free t. Every τ ≤ 0.3 passes under all three t rules. On #8c's own India/US holdout the pooled bge veto also gains **+0.00034 / +0.00019** (τ 0.01, 7–8 SE).
**The France sign is not proven.** The same out-of-domain CE vetoing an *in-domain* GBDT's confident pairs loses (−0.0024 / −0.0010). The veto pays only when the confident pairs hold real GBDT errors, and France's veto rate (0.47%) sits between the two regimes. Payoff per LB point is asymmetric: about +0.0004 if France behaves like LOCO, about −0.0002 if it behaves like in-domain.
**Upload `output/sub30a`** = #9ab with 3,676 France pairs vetoed (τ 0.01, t 0.985 kept, empty 6.76% vs 6.71%). Validator PASS; India/US byte-identical to #8c. AWS: one job, **$2.07**.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**LB slot: `output/sub30a`** in this worktree (`/home/harsh1/github-repos/amc-26-wt/p30-bge-veto/output/sub30a/`). **Add code.zip**; combine.py writes only the two TSVs.
- What changed vs #9ab: France pairs with `p_gbdt ≥ 0.98` and pooled-bge `r < 0.01` get `p := 0`. Nothing else changes: same candidates, same t 0.985, same India/US rows.
  - 3,676 pairs are removed from 3,611 France S1, and 131 S1 become empty.
  - France empty rate 6.76% (#9ab 6.71%, D4 6.85%); matches per S1 3.004 (#9ab 3.018).
- How to read the LB, against #9ab's 0.984072: ΔLB ≈ 0.1497 × ΔF0.5(France).
  - ≥ +0.0001: keep it. Then try **`output/sub30b`** (τ 0.05, 5,866 pairs vetoed, empty 6.79%, PASS).
  - ≤ −0.0001: drop the France veto.
- **Why t stays at 0.985 rather than being re-matched to #9ab's empty rate.** LOCO measured both.
  - Re-matching lowers t and refills the vetoed empties with band pairs.
  - That costs half the gain in US→India (+0.0148 vs +0.0297) and a little in India→US (+0.0034 vs +0.0037).
  - The re-matched France variant would be t 0.984 (empty 6.70%). It is not built.
- **For src promotion (if the LB agrees):**
  - After `p19.blend`: `p = where((p_gbdt >= 0.98) & (r < 0.01), 0, p)`.
  - Here r is the pooled bge (`ce_bge_pool.pt`) on every France pair with p_gbdt ≥ 0.98: 771k pairs, ~8 min on an A10G. The same `bge_ckpt` path, with a wider band.
- **Rebuild:**
```
.venv/bin/python probes/P30-bge-veto/p30.py build            # 21 s, 1.6 GB -> data/work/p30/sm_data (2.44M pairs)
.venv/bin/python probes/P30-bge-veto/p30.py launch --profile harsh --region ap-south-1 --instance ml.g5.xlarge --no-spot \
  --prefix amc26-p30 --name <job> --data data/work/p30/sm_data --solution probes/P30-bge-veto/solution_p30.py --max-run 7200 \
  --hp '{"phases": ["loco_US", "loco_India", "ckpt"], "ckpt": "s3://sagemaker-114872709824-ap-south-1/amc26-p13/results/p13-rerank-0925-2257/ckpt/ce_bge_pool.pt",
         "s3": "s3://sagemaker-114872709824-ap-south-1/amc26-p30/results/<job>", "deadline_s": 6600}'
.venv/bin/python probes/P30-bge-veto/p30.py fetch <job>      # -> data/work/p30/r
.venv/bin/python probes/P30-bge-veto/eval.py ctl             # numerics vs P13's r
.venv/bin/python probes/P30-bge-veto/eval.py loco            # 14 s, 1.35 GB
.venv/bin/python probes/P30-bge-veto/eval.py ind pool        # #8c yardstick;  'ind loco' = pessimistic bound
.venv/bin/python probes/P30-bge-veto/eval.py france tau:0.01 # -> data/work/p30/prob_test_France_veto_tau0.01.parquet
.venv/bin/python probes/P14-france-shift/combine.py /home/harsh1/github-repos/amc-26/output/sub8c \
    data/work/p30/prob_test_France_veto_tau0.01.parquet 0.985 output/sub30a
cd data/student_resource && python3 utils/validate_submission.py --matching <abs>/output/sub30a/matching_results.tsv \
    --candidate <abs>/output/sub30a/candidate_pairs.tsv --test-dir dataset/test     # PASS
```
- **India/US (not built, a cheap follow-up).**
  - The pooled-bge veto at τ 0.01 gains **+0.00025** pooled on #8c's holdout: India +0.00034 ± 0.00005, US +0.00019 ± 0.00003. That is ≈ +0.0002 LB.
  - It needs pooled r for #8c's accepted India/US test pairs that have no band r: ≈ 4.3M of 4.98M. That is ~45 min on the g5, ≈ $1.4.
  - Then remove pairs with r < 0.01 from sub8c's India/US rows.
  - Not launched: the ~1.5 h round trip over a shared g5 slot and a 99%-full disk left no margin before 21:00.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| p30-build | holdout+test | tgt_India 409,139 / tgt_US 562,196 (P6 LOCO, target fold 0, p > 0.02); pool_India 277,794 / pool_US 417,204 (#8c holdout am & p_m2 ≥ .98); pool_France 773,212 (771,212 p_gbdt ≥ .98 + 2,000 ctl); P6 ids → P3 text rows round-trip asserted; France control texts = P13's | keep |
| p30-sm-done | aws | `p30-veto-0926-1651`, 4,405 billable s = **$2.07**. Train 367 / 287 pairs/s. Val AUC (CE on the unseen country): locoUS→India 0.836, locoIndia→US 0.770. France scored at 1,552 pairs/s | keep |
| p30-sm-repro | test+holdout | restored pooled bge vs P13's r: holdout corr 0.999999 / 0.999995, France (36,097 overlapping pairs) 0.999994; median \|Δlogit\| 0, p99 ≤ 0.063 | keep |
| loco-veto-tau:{0.01…0.5}, loco-veto-full | holdout-dense-full LOCO (P6 target fold 0) | table A below; gate (≥ +0.0005 both ways, matched t) passes for τ 0.01–0.3 and `full` | **keep** |
| indpool-veto-* | holdout-dense (#8c) | τ 0.01: India **+0.00034 ± 0.00005** (157 vetoed, 24% true), US **+0.00019 ± 0.00003** (143, 27% true); τ ≤ 0.2 positive, τ ≥ 0.3 negative (table C) | keep (India/US follow-up) |
| indloco-veto-* | holdout-dense (#8c), LOCO r | **pessimistic bound**, in-domain GBDT + out-of-domain CE: τ 0.01 India −0.0024 / US −0.0010, vetoed 97–98% true; worse as τ grows (table C) | drop (bound) |
| france-veto-tau:{0.01,0.05,0.1}, france-veto-full | test-France | table D | keep (τ 0.01) |
| p30-sub30a / p30-sub30b | test | validator PASS; vs #9ab: 3,676 / 5,866 France pairs removed, 0 non-France rows changed, candidates identical | keep / inconclusive |

**A. LOCO: the France procedure on P6's worlds.** P6's source-only LightGBM scores the unseen country's fold 0, with every train S1 competing. The band 0.02–0.98 is blended with a bge trained on the source only, using P13's out-of-domain France weights [0.9245, 0.2595, 0.0515]. The base t comes from P6's label-free rule (clip [t*, t* + 0.2]).
- US→India: t* 0.715, base t 0.915, empty 4.70%, F 0.92692. 283,156 confident pairs at 94.4% true.
- India→US: t* 0.72, base t 0.92, empty 5.44%, F 0.95057. 380,310 confident pairs at 97.9% true.

Gain vs the base (paired SE over all fold-0 S1). Columns:
- *fixed* = the base t.
- *matched* = t re-matched to the base's empty rate.
- *p6* = P6's rule re-run on the vetoed p.

| rule | US→India fixed | matched | p6 | vetoed (true) | India→US fixed | matched | p6 | vetoed (true) |
|---|---|---|---|---|---|---|---|---|
| no CE (ref) | −0.0395 | −0.0290 | −0.0395 | – | −0.0217 | −0.0210 | −0.0213 | – |
| full-range blend | +0.0169 | +0.0076 | +0.0169 | – | +0.0017 | +0.0015 | +0.0017 | – |
| **τ 0.01** | **+0.0297 ± .0004** | +0.0148 ± .0005 | +0.0297 | 14,239 (18.2%) | **+0.0037 ± .0001** | +0.0034 ± .0002 | +0.0037 | 3,450 (32.4%) |
| τ 0.02 | +0.0296 | +0.0135 | +0.0296 | 15,724 (23.5%) | +0.0044 | +0.0040 | +0.0044 | 4,271 (34.0%) |
| τ 0.05 | +0.0290 | +0.0110 | +0.0293 | 17,707 (29.7%) | +0.0053 | +0.0044 | +0.0053 | 5,411 (36.6%) |
| τ 0.1 | +0.0283 | +0.0088 | +0.0287 | 19,371 (34.2%) | +0.0056 | +0.0046 | +0.0056 | 6,268 (39.2%) |
| τ 0.2 | +0.0274 | +0.0066 | +0.0275 | 21,304 (38.7%) | +0.0055 | +0.0043 | +0.0055 | 7,198 (43.5%) |
| τ 0.3 | +0.0262 | +0.0047 | +0.0261 | 22,980 (42.3%) | +0.0051 | +0.0038 | +0.0050 | 8,105 (48.3%) |
| τ 0.5 | +0.0194 | −0.0073 | +0.0169 | 29,924 (54.5%) | +0.0008 | −0.0017 | +0.0002 | 14,830 (69.6%) |

**B. Precision of LOCO confident pairs (argmax & p ≥ .98) by LOCO bge r.** Unlike P27's band-selected subset, every confident pair is scored here.

| r | US→India n | true | India→US n | true |
|---|---|---|---|---|
| [0, .01) | 14,239 | **18.2%** | 3,450 | **32.4%** |
| [.01, .02) | 1,485 | 73.9% | 821 | 40.3% |
| [.02, .05) | 1,983 | 79.5% | 1,140 | 46.4% |
| [.05, .1) | 1,664 | 81.3% | 857 | 55.4% |
| [.1, .2) | 1,933 | 83.6% | 930 | 72.8% |
| [.2, .3) | 1,676 | 88.9% | 907 | 86.1% |
| [.3, .5) | 6,944 | 95.0% | 6,725 | 95.3% |
| [.5, .8) | 7,353 | 91.8% | 8,386 | 93.0% |
| [.8, .95) | 10,159 | 93.7% | 8,628 | 87.2% |
| [.95, 1] | 235,720 | 99.5% | 348,466 | 99.5% |

**C. #8c holdout (t 0.71 fixed), gain from removing predicted p_m2 ≥ .98 pairs with r < τ.**
- #8c base: India F 0.99048 (88,168 S1), US F 0.98951 (132,190 S1); confident precision 99.93% / 99.92%.

| τ | pooled bge: India | US | vetoed (true) I / U | LOCO bge (bound): India | US | vetoed (true) I / U |
|---|---|---|---|---|---|---|
| 0.01 | **+0.00034 ± .00005** | **+0.00019 ± .00003** | 157 (24%) / 143 (27%) | −0.00244 | −0.00098 | 2,697 (97.7%) / 1,590 (97.4%) |
| 0.05 | +0.00032 | +0.00015 | 231 (42%) / 206 (46%) | −0.00509 | −0.00168 | 5,346 (98.6%) / 2,702 (97.9%) |
| 0.1 | +0.00028 | +0.00015 | 293 (52%) / 259 (54%) | −0.00668 | −0.00210 | 6,805 (98.8%) / 3,305 (98.0%) |
| 0.2 | +0.00011 | +0.00009 | 461 (69%) / 427 (70%) | −0.00877 | −0.00273 | 8,639 / 4,200 |
| 0.3 | −0.00021 | −0.00126 | 785 / 2,476 | −0.01057 | −0.00337 | 10,317 / 5,086 |

**D. France (#9ab's file, pooled bge on the 771k p_gbdt ≥ .98 pairs), label-free.**
- #9ab: 783,035 accepted pairs, empty 6.71% at t 0.985.
- hn-class mix of all accepted pairs: none/eq 98.9%, +SIB 0.14%, digit 0.14%, small 0.30%, far 0.49%.

| rule | vetoed (share of accepted) | t 0.985: empty / matches per S1 | t re-matched to 6.71% | t at D4 6.85% | vetoed hn: +SIB / non-eq |
|---|---|---|---|---|---|
| **τ 0.01** | 3,676 (0.47%) | 6.76% / 3.004 | 0.984 (+3,492 −3,676 pairs) | 0.9865 | **3.7% (26×)** / 4.5% (4.2×) |
| τ 0.05 | 5,866 (0.75%) | 6.79% / 2.995 | 0.9835 | 0.986 | 2.7% / 3.4% |
| τ 0.1 | 7,664 (0.98%) | 6.82% / 2.989 | 0.9825 | 0.9855 | 2.2% / 2.9% |
| full-range blend | 8,273 removed, 7,619 added | 6.71% / 3.016 | 0.985 | 0.989 | 2.2% / 2.7% |

France r histogram of the accepted confident pairs:

| r | [0, .01) | [.01, .02) | [.02, .05) | [.05, .1) | [.1, .2) | [.2, .3) | [.3, .5) | [.5, .8) | [.8, .95) | [.95, 1] |
|---|---|---|---|---|---|---|---|---|---|---|
| pairs | 3,676 | 735 | 1,455 | 1,798 | 2,325 | 2,827 | 16,270 | 6,081 | 8,319 | 716,390 |

## What failed      (approaches tried and dropped, with the number that killed them)
- **Re-matching t to the base's empty rate after the veto** is worse than keeping t, in every LOCO cell. At τ 0.01: US→India +0.0148 vs +0.0297, India→US +0.0034 vs +0.0037. At τ 0.5 it turns negative (−0.0073 / −0.0017). sub30a therefore keeps 0.985.
- **Full-range blend** (the OOD weights applied to p ≥ 0.98 as well) is positive but a third to a half of the hard veto: +0.0169 / +0.0017 vs +0.0297 / +0.0037.
- **τ ≥ 0.3:**
  - It loses in-domain (#8c −0.0002 / −0.0013).
  - At τ 0.5 the LOCO gain collapses.
- **The veto as a general rule for an in-domain GBDT with an out-of-domain CE** is negative (table C, right). That is exactly the France risk (see Risks).
- **Operational:**
  - The first launch hit `ResourceLimitExceeded`: P29's job held the single g5 for ~4 minutes. `launch_retry.sh` launched at the next free minute.
  - The master's disk offload removed `data/work/p27/` (P6's frames) mid-probe. The two frames were copied back from the offload into `data/work/p30/p6/`.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `probes/P30-bge-veto/p30.py`:
  - `build()`: the channel. LOCO targets from P6 `S_holdout_<src>` (target fold 0, p > 0.02); in-domain from P23's #8c dump; France from #9ab's file. Texts via P4b's `rerank.texts`. 21 s, 1.6 GB.
  - `launch`: `p13.launch` with `solution_p30.py`.
  - `fetch(job)` → `data/work/p30/r`.
- `probes/P30-bge-veto/solution_p30.py`: P13's job, with only `Job.run` replaced.
  - `loco_<src>`: fine-tune bge on train_<src> (P13's recipe: 1 epoch, bs 128, lr 2e-5), score `tgt_<other>`, save weights to S3 `ckpt/ce_bge_loco<src>.pt`.
  - `ckpt`: restore the pooled checkpoint and score controls + `pool_*`.
  - A10G: train 287–367 pairs/s at 13 GB; score 1,178–1,724 pairs/s.
- `probes/P30-bge-veto/eval.py` (CPU, ≤ 1.35 GB, 3–14 s each):
  - `loco [US|India|US,India]`, `ind pool|loco`, `ctl`, `france <rule> [ref_t]`, `selftest`.
  - Rule strings: `tau:<x>` (veto p_gbdt ≥ .98 & r < x → p 0) or `full`.
  - `france` writes `data/work/p30/prob_test_France_veto_<rule>.parquet` [s1, sx, p, p_gbdt]. Rows and order = #9ab's file.
- Outputs:
  - r files: `data/work/p30/r/r_{locoUS_tgt_India, locoIndia_tgt_US, pool_India, pool_US, pool_France, holdout_India, holdout_US}.parquet`.
  - JSON: `loco_US_India.json`, `ind_pool.json`, `ind_loco.json`.
  - Submissions: `output/sub30a`, `output/sub30b`.
- LOCO weights `s3://…/amc26-p30/results/p30-veto-0926-1651/ckpt/ce_bge_loco{US,India}.pt` now exist for any later LOCO work.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **The France sign rests on which regime France is in.** The CE's false-veto rate on *true* confident pairs is stable per CE × country. At τ 0.01 it is 0.95% (India) / 0.30–0.37% (US) in both the P6 world and the #8c bound. The pooled CE in-domain is 0.01%.
  - France vetoes 0.47% of its accepted pairs, which is inside that range. If the pooled CE's false-veto rate on France is ≈ 0.1%, the vetoed pairs are ~80% false. If it is ≥ 0.4%, most vetoed pairs are true.
  - Per vetoed pair, the measured value is ≈ +0.16…+0.19 S1-units at 18–32% true, and ≈ −0.08 at ~97.5% true. Break-even is ~75% true.
  - For France's 3,676 pairs that means ≈ +0.0004 LB (LOCO-like), ≈ +0.0002 (half true), or ≈ −0.0002 (in-domain-like). The upside is about 2× the downside, and τ 0.01 caps the exposure.
- **For the favourable regime:**
  - The LB puts France around F0.95, near the LOCO targets (0.927 / 0.951), not near in-domain 0.99.
  - The vetoed pairs are 26× enriched in +SIB house-number mismatches, the generator-sibling class that P27 found only 19% precise at p ≥ .98 in India→US.
- **Against it:**
  - France's veto rate is lower than both LOCO worlds' (5.0% / 0.9% at τ 0.01).
  - The CE is further out of domain on France (band disagreement 33% vs 20–26%). That can raise its false-veto rate.
- **Proxy selection.** τ was picked on the same LOCO data. Every τ from 0.01 to 0.3 passes, so the selection effect is small. τ 0.01 was chosen as the most conservative: it gains in LOCO and in-domain and loses least in the bound.
- **LOCO mechanics (assumed).**
  - The argmax is P6's `am_all`, fixed on the GBDT p: the band blend does not re-rank, and a veto removes the pair without reassigning the SX.
  - France uses combine.py's recomputed argmax. After the τ 0.01 veto, 0 new argmax pairs are accepted at t 0.985.
  - The LOCO bge models are retrains of P13's recipe (same seed and data), not P13's weights. Val AUC is similar (India 0.836 vs P13's band 0.821).
- **Holdout optimism** on the #8c yardstick is the same as P13's: shared S2/S3 corpus.
- **Licence / rules.** Same Apache-2.0 bge-reranker-v2-m3 (568M) and the same checkpoint. No external data. France is not filtered: every France row is present, and the validator passes.

## Open questions   (for the master; each with your best guess)
- **Upload sub30a before or after P29's France rebuild?** Best guess: now, against #9ab, so the ΔLB is the veto alone. The veto composes with P29: it removes at p_gbdt ≥ 0.98, while tiers add below t. If P29's file becomes the base, re-run `eval.py france tau:0.01` on it after pointing the loader at the new file. The r already covers every #9ab p_gbdt ≥ 0.98 pair; new pairs would need scoring.
- **If sub30a wins, sub30b next?** Yes. τ 0.05 was the LOCO maximin (+0.0290 / +0.0053) at 60% more vetoes. If sub30a loses, stop: τ 0.05 loses more under the unfavourable regime.
- **India/US veto?** Best guess: worth one job (~$1.4, ~1 h) if a g5 slot and an LB slot are free. It is holdout-validated at +0.00025 (7–8 SE), ≈ +0.0002 LB, and independent of the France question. The spec is in Recommendation.
