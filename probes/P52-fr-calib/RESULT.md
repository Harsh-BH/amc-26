# P52-fr-calib — RESULT (final, 27 Sep 05:45)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **The calibration arms fail as briefed.** Arm 1 (Saerens EM prior shift), arm 3 (adversarial ranks + refit) and arm 4 (top-1 quantile map) all fail. Arm 2 (importance weighting) fails source→target on the chain analogue.
  - The LOCO target is **over-confident in the band, not prior-shifted**: π_T = π_S = 0.726, and p 0.5–0.9 pairs are only 20–44% true.
  - So any "calibrate, then apply the source bars" rule admits false pairs: EM −0.016 / −0.014 on chain; the true-π shift −0.009 / −0.013. EM's π̂ is 0.76–0.83 against a true 0.726.
- **Importance weighting is the one real transfer signal.** Moving bars from one unseen-country frame to the other, it beats P42's density map in all 4 readings (+0.0036 / +0.0068 chain, raw bars; +0.0017 / +0.0057 through the density map).
  - On France (LOCO S1s weighted toward France, ESS 11–12%) it prefers hn extras off or at ≥ 0.9.
  - That fails the unweighted India→US gate (−0.0027 / −0.00065), so it is **not shipped**: it is a risk flag on the hn stage.
- **Ship: `output/sub52cal`** = sub52_max + 1,475 France pairs. Rescue 0.90 → **0.80** and gc 0.70 → **0.50**, with P48's veto re-applied to the new stages.
  - Gate (chain, cell picked on h0, read on h1): **+0.00024 / +0.00033**; m2 +0.0013 / +0.00066; France-weighted ≈ 0 to +0.0005.
  - Expected LB **≈ +0.00005** (up to +0.0001). Official validator `--check-ids` **PASS**.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**Upload / integrate:** `output/sub52cal` (base `sub52_max`; only France lines change, only by additions).
- Change set: `data/work/p52/sub52cal_changes.parquet` (op, s1, sx, p): 1,672 recipe adds, 197 of them already in sub52_max.
- **Rule:** the France recipe as now (t 0.99 on post-rules p, caps 5/6, τ 0.1, hn extras 0.80), with **rescue bar 0.80** (was 0.90) and **gc bar 0.50** (was 0.70). Then P48's veto on the new stages: a band pair (0.02 < p_gbdt < 0.98, p = OOD blend) that the rescue or gc stage added leaves if bge r < 0.2.
- **Why now:** P42 tuned these two bars without the veto. The veto removes exactly the low-CE false adds that lower bars let in, so the joint optimum moves down.
- **Builder** (any base; deltas vs the current recipe; SX-owner and cap guards; 6 s, 0.9 GB):
  `.venv/bin/python probes/P52-fr-calib/build52.py <BASE> <OUT> t=0.99,r=0.8,gc=0.5,hn=0.8`
- **run.py / src port:** in France's P42 `decide(...)` call, pass `t_r = 0.80, t_gc = 0.50`. P48's veto already runs after `gc_add` on the rescue + gc stage pairs, so nothing else changes.

**Do not adopt** (evidence below):
- EM or BCTS recalibration;
- quantile-mapped or source-optimal bars;
- the rank-transformed frame scorer;
- the France-weighted picks with the hn stage off.

## Evidence         (table: ledger name | world | key metrics | verdict)
France-F0.5 dF vs the current recipe analogue:
- P42 density-mapped t .99 / r .9 / gc .7 / hn .8, caps 5/6, τ 0.1, plus P48's rescue/gc band veto at r < 0.2;
- each cell reads US→India / India→US (the LOCO target country is after the arrow);
- SE 0.0001–0.0006 unless given.

**Arms as briefed** (source = the source country's own fold 0, labelled, from `prep52.py`; target = P42 LOCO frame):
| ledger name | arm | chain | m2 | verdict |
|---|---|---|---|---|
| arm1-em-prior-shift-src2tgt | EM (BCTS + Saerens), source-optimal bars in q | −0.0159 / −0.0142 | −0.0461 / −0.0023 | drop |
| (same row) | true-π shift (the ceiling of any prior correction) | −0.0091 / −0.0130 | −0.0371 / −0.0014 | — |
| (same row) | BCTS only | −0.0071 / −0.0129 | −0.0351 / −0.0013 | — |
| (same row) | π̂ vs true π_T | 0.796 / 0.758 vs 0.726 | 0.829 / 0.754 vs 0.726 | EM follows the over-confidence |
| arm4-qmap-top1-src2tgt | top-1 quantile map + source bars | +0.0016 / −0.0058 | −0.0055 / −0.0020 | drop |
| (same row) | naive (source bars, raw p) | −0.0105 / −0.0089 | −0.0327 / −0.0005 | — |
| arm2-iw-bars-src2tgt | source bars re-tuned with S1 weights p(T\|x)/p(S\|x) (AUC 0.89 / 0.92, ESS 0.18 / 0.17) | +0.0020 / **−0.0091** | +0.0059 / +0.0098 | drop (chain fails) |

- The source-optimal bars are low: t 0.7–0.8, in-domain M2 is sharp.
- Target reliability (chain, US model on India): p 0.5–0.7 is 20% true; 0.7–0.9 is 44%; g < 0.02 is 2.6% (source 0.3%).

**Transfer rules between the two unseen-country frames** (`xfer-rule-tournament-frameA2frameB`, `xfer-iw-then-density`). A's oracle bars → B; B's labels only score. P42's density map is the rule France uses:
| rule | chain India-tgt → US-tgt | chain US-tgt → India-tgt | m2 India-tgt → US-tgt | m2 US-tgt → India-tgt |
|---|---|---|---|---|
| B oracle (in-sample ceiling) | +0.0022 | +0.0034 | +0.0183 | +0.0140 |
| raw (same bars) | −0.0041 | −0.0029 | +0.0034 | +0.0006 |
| **density (P42)** | −0.0039 | −0.0041 | −0.0031 | −0.0009 |
| qmap top-1 | −0.0039 | −0.0031 | +0.0023 | +0.0036 |
| em | −0.0028 | −0.0045 | −0.0021 | −0.0062 |
| **iw, raw bars** | **−0.0003** | **+0.0027** | **+0.0161** | **+0.0074** |
| **iw → density** | −0.0022 (+0.0017 vs density) | +0.0016 (+0.0057) | −0.0012 (+0.0019) | +0.0006 (+0.0015) |

**Arm 2 on France** (`arm2-fr-iw-grid`, `arm2-lb-backtest-weighted-loco`, `cands-r80gc50-with-p48-veto`):
- LOCO S1s are weighted toward France's argmax S1s: AUC 0.956 / 0.945, ESS 12.6% / 10.7%, 99th-pct weight ≈ 6.
- The 2,401-cell P42 grid (density-mapped, chain, veto on) picked on the weighted tune half:
  - picks: **hn off, rescue 0.97, t .99, gc .7**, weighted **+0.0028 ± 0.0004 / +0.0003 ± 0.0004**;
  - but unweighted +0.0026 / **−0.0032**.

**LB back-test.** France-weighted LOCO is only a partial LB predictor:
| lever | LB (France-F) | unweighted | weighted |
|---|---|---|---|
| rescue90 | +0.0021 | −0.0037 / +0.0010 | −0.0017 / +0.0010 |
| gc70 | +0.0023 | −0.0010 / +0.0022 | +0.0005 / +0.0012 |

- The weighted frame-mean is closer to the LB.
- Unweighted India→US alone still called gc70 best.

**Candidate cells** (vs current; u = unweighted, h1 = report half; full / sqrt / clip90 / nop = France-weighting variants):
| cell | chain u (h1) | chain weighted full / sqrt / clip90 / nop | m2 u | m2 weighted (4) |
|---|---|---|---|---|
| **A: r .8, gc .5 → sub52cal** | **+0.00016 (+0.00024) / +0.00034 (+0.00033)** | −0.00007/+0.00013/+0.00014/−0.00009 ; +0.00003/+0.00033/+0.00052/−0.00014 | +0.0013 / +0.00066 | all ≥ 0 except nop India→US −0.00004 |
| F: r .8, gc .5, hn .9 | +0.0013 / +0.00006 | all 8 > 0 (+0.0003 to +0.0020) | +0.0019 / +0.00025 | all 8 > 0 |
| B: hn .95 | +0.0022 / **−0.00065** | US→India all > 0; India→US +0.0008 / −0.0002 / −0.0003 / +0.0016 | +0.0011 / −0.00087 | mixed |
| E: r .97, hn off (weighted top) | +0.0026 / **−0.0032** | India→US +0.0003 / −0.0022 / −0.0027 / +0.0014 | +0.0013 / −0.0038 | mixed |

**Arm 3** (`arm3-adversarial-rank5-refit`):
- Adversarial AUC on P39's 20 frame columns: France vs LOCO **0.9995**; US target vs India target 0.994.
- Top shifted columns: lt 0.27, q 0.21, best_oth_lt 0.20, state_eq 0.14, best_src_ex_lt 0.05.
- P39 scorer refit, add at 0.8 on top of the recipe:

| train labels → apply | raw | rank5 | drop5 |
|---|---|---|---|
| US → India | −0.087 | −0.058 | −0.083 |
| India → US | +0.0023 | +0.0014 | +0.0006 (addrm 0.9 +0.0080) |

- It fails both ways.

**Build** (`build-sub52cal`):
- sub52_max + 1,475 pairs (1,672 recipe adds; 197 already present), 1,438 S1, 282 empty S1 filled, 0 removed.
- Guards: 0 SX-owner skips, 0 cap skips.
- `check.py` (P47): 0 non-France diffs, 0 SX twice, 0 over cap, 0 adds outside `candidate_pairs.tsv`. `check48.py`: additions only.
- md5 30834a0f…; `candidate_pairs.tsv` / `code.zip` are hard links to sub52_max's (= sub48's inode).
- Official validator `--check-ids`: **PASS** (SageMaker `p52-val-sub52cal-0526`, ml.m5.xlarge spot, ≈ $0.01): 1,732,544 rows, 100,489 empty (= sub52cal locally; sub52_max has 100,771).

## What failed      (approaches tried and dropped, with the number that killed them)
- **EM prior re-estimation (arm 1):** the label-shift assumption is false here, since π_T = π_S.
  - π̂ is off by +0.03 to +0.10 because the target is over-confident.
  - Even the true-π shift loses −0.009 / −0.013 (chain), because the source-optimal bars (t 0.7–0.8) are far too low for an unseen country.
- **Top-1 quantile map (arm 4):** −0.0058 India→US (chain). As a frame→frame transfer rule it only ties the density map on chain.
- **IW source→target (arm 2 as briefed):** −0.0091 India→US (chain).
  - The in-domain source cannot be weighted into looking like an unseen country: its p→y relation differs.
- **IW-weighted France picks** (hn off / r .97): −0.0032 unweighted India→US. Not robust across weighting variants: sqrt and clip90 flip India→US negative.
- **Arm 3 (rank5 / drop5):** −0.058 / −0.083 US→India, as P39's country-specificity.
- **Spare `sub52cal_hn90` (cell F):** built and checked, then deleted.
  - Its 5,184 hn-extra removals can hit P41 gib / P46 swap tier pairs, which the builder cannot tell apart from recipe pairs.
  - It fails the India→US gate anyway (+0.00006).

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `prep52.py US|India`: source in-domain frames → `data/work/p52/src_<src>.npz`. P6 am_all rows of the source's own fold 0, with r = P45 r_all; 19 s, 1.2 GB.
- `prep52.py france`: → `data/work/p52/france_hn_rel.npy` (P39 hn classes; asserted equal to P42's hn_eq / sib).
- `p52.py`: numpy / sklearn. `python p52.py` runs the self-check. Subcommands:
  - `loco` (arms 1 / 2 / 4) → `loco.json`;
  - `xfer`, `xfer_iwd` → `xfer.json`;
  - `fr_iw` (2,401-cell France grid, 10 min, 0.6 GB) → `fr_iw.json` + `data/work/p52/fr_iw_grid.parquet`;
  - `lbcheck` → `lbcheck.json`;
  - `cands` → `cands.json`.
- `p52.py` functions:
  - `run(z, p, bars)` (P42 decide + P48 veto);
  - `tune(z, p, w)` (coordinate descent);
  - `bcts`, `em_prior`, `shift`, `qmap`;
  - `iw_weights(zs, ps, zt, pt)` (S1-level HistGB, 2-fold cross-fit, clip 99.5%);
  - `dmap`, `qbars`.
- `arm3.py` → `arm3.json` (5 min, 0.8 GB).
- `build52.py BASE OUT SPEC`: 6 s, 0.9 GB.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Nothing new ships but two bars.** No model, no data, no country key: the recipe runs on unseen-country S1 as before. sklearn HistGradientBoosting (BSD) is used only in the probe.
- **sub52cal's gain is small and in-sample-ish.**
  - It was picked from 2,401 cells on h0; h1 replicates (+0.00033 India→US).
  - France-weighted chain full is ≈ 0 (−0.00007 / +0.00003).
  - Downside ≈ −0.00001 LB.
- **Interaction with sub52_max's non-recipe France pairs** (P47 cls80, tiers): the LOCO frames have none of them. The adds are guarded (owner / caps) and 197 of them were already present.
- **The hn stage (hn extras 0.80) is the soft spot.** Under France-like weights its India→US value vanishes (≈ 0 to −0.0004) and its US→India cost (−0.0029) remains. This is the "France India-like on hn" risk P42 named.
- **Source-side chain p** uses P45's in-domain CE (r_all), not P30's source-only CE: the latter has no source-side scores. It affects the source→target arms only.
- **Disk:** sub52cal = 98 MB (matching only); 3.1 GB free after.

## Open questions   (for the master; each with your best guess)
- **Test the hn bar on the LB?** hn 0.9 (cell F) is positive in 16 of 16 LOCO readings except that unweighted India→US is only +0.00006. Best guess: worth one diagnostic upload after sub52cal, as `build52.py <base> <out> t=0.99,r=0.8,gc=0.5,hn=0.9`, but only once the builder exempts P41 gib / P46 swap tier pairs from removal. Expected ≈ +0.0000 to +0.0002.
- **Use IW weights as a second LOCO reading in future France gates?** It beats the density map at transferring bars between frames (4 of 4), but it back-tests only partially on the LB. Best guess: report it next to the unweighted gate, and do not gate on it alone.
