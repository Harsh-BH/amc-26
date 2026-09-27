# P76-segment-hill — RESULT (final, 27 Sep 19:15 IST)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Company TYPE does not drive the loss, and no per-segment India/US knob beats one t 0.71.**
  - Org-word type and legal form have |lift| ≤ .036. The loss is FN_up (the true SX is not the S1's argmax pair) in
    common-name twins (34% of S1, 51% of loss) and in small groups.
  - Every hill-climb arm is negative out of sample, and the permuted-label nulls reach the same fit. joint_dek (133–143 knobs)
    reads −0.000100 / −0.000148; joint_d (89) reads −0.000051 / −0.000109.
- **France: no fitted per-segment arm passes the gate within rescue ≤ .90 / gc ≤ .70.**
  - joint_lo reads India→US +.0041 / +.0036 but US→India −.00007 / −.00045.
  - LOCO's India→US direction retrodicts the two LB-measured bar steps (gc70 +.0021 vs LB-implied +.0023). US→India gets both
    signs wrong.
  - One simple rule, R3 (t .9868 → .9925 for France S1 with exactly 2 argmax pairs), is positive in all 8 LOCO readings and beats
    its label null: chain +.00015 / +.00015 US→India, +.00098 / +.00103 India→US.
- **Recommendation:** no India/US change. Keep `output/sub82r3` (sub77 − 260 France pairs, validator PASS) only as an optional
  bundle-in. It is worth ≈ +0.00004–0.00009 LB, below single-upload resolution, so it should not take a slot of its own.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **India/US:** nothing. Keep one t 0.71 on the sub54 combiner.
- **France:**
  - No chain-bar change.
  - Optional, removal-only: **`output/sub82r3`**. It is `probes/P76-segment-hill/build_r3.py output/sub77 output/sub82r3`
    (2 s), and its change rows are `data/work/p76/r3_changes.parquet` (step `76r3`, 260 rows).
  - To make it rebuildable in final_build, add `"76r3"` to `STEPS` (after `41`) and to `REMOVE`, then append those rows to
    final_changes.parquet. That is a src/ edit, so it is left for the master.
  - Bundle it into whichever France candidate uploads next, or leave it out: the expected gain is ≈ +0.00004–0.00009.
  - The rule: in P69's t9868 France chain frame (`amc26-offload/work/p69/t9868/p69_frame.npz`), take the S1 with exactly 2
    SX-argmax pairs. Their base-stage (acc0) pairs with post-rules p in [.9868, .9925) leave, unless the pair is its source's
    last member on the line (rescue / gc would re-add it: LOCO's net effect).
- Do not stage a tightened France chain (t .99, rescue .97, gc .9). A global-only control gets most of the unbounded arms' gain,
  and the move contradicts LB-measured rescue90 / gc70.

## Evidence         (table: ledger name | world | key metrics | verdict)
**Frame.**
- p54 = sub54's India/US recipe, out of fold on P57's full holdout world. Half-1 S1 get the h0-fit boosters (P57's cached
  `b_sub54_h0_s*`), half-0 S1 get the h1 fit.
- It reproduces P57's `full|orig|sub54` deltas vs #34 exactly (India +0.000635 / +0.000873, US +0.000602 / +0.000712).
- Base = p54 ≥ 0.71 per argmax pair.
- Loss = 1 − per-S1 F0.5. Pooled rows weight India 0.468 and US 0.383 per S1 share.

**Loss columns.**
- FN_below = a true argmax pair with p54 < 0.71. A lower t can reach it.
- FN_up = a true SX that is not in the S1's argmax pairs. No decision knob reaches it.
- Base totals: India FP 310, FN_below 1,539, FN_up 6,104. US FP 581, FN_below 3,863, FN_up 8,073.

**1. Deep dive: top segments by lift = loss share − S1 share** (pooled, n ≥ 500; full tables in data/work/p76/dive_top.md; raw
error examples, 2 per error kind per top segment, are in the gitignored data/work/p76/examples.md)

| segment | n S1 | S1 share | loss share | lift | FP | FN_below | FN_up |
|---|---|---|---|---|---|---|---|
| twin 5+ (name core shared by ≥ 5 train S1 of the country) | 70,260 | .338 | .509 | +.171 | 215 | 1,884 | 8,509 |
| 0 accepted pairs | 12,503 | .057 | .214 | +.158 | 0 | 159 | 368 |
| 1 accepted pair | 13,446 | .061 | .188 | +.127 | 40 | 592 | 1,499 |
| 1 argmax pair | 11,594 | .052 | .148 | +.096 | 20 | 59 | 819 |
| address-less SX share ≥ .5 of the S1's argmax pairs | 3,749 | .016 | .106 | +.090 | 61 | 625 | 267 |
| twin 2–4 | 42,960 | .186 | .274 | +.087 | 276 | 2,657 | 2,889 |
| address-less SX share in (0, .5) | 28,690 | .125 | .200 | +.075 | 433 | 3,744 | 1,378 |
| 2 argmax pairs | 26,867 | .122 | .196 | +.074 | 73 | 307 | 2,090 |
| 0 argmax pairs | 5,035 | .023 | .084 | +.061 | 0 | 0 | 203 |
| 2-token name | 30,655 | .127 | .165 | +.038 | 154 | 791 | 2,837 |
| no legal form | 70,626 | .280 | .316 | +.036 | 359 | 1,896 | 5,063 |
| under-represented: unique core (twin 1) | 107,138 | .476 | .218 | −.258 | 400 | 861 | 2,779 |
| under-represented: no address-less argmax SX | 182,884 | .837 | .611 | −.226 | 397 | 1,033 | 12,329 |

**Org-word types (per country, lift).**
- US: medical +.025 (12.1% of S1 → 14.6% of loss), trust/society +.011, club/association/sports +.009, school +.006,
  religious +.006, dental +.006.
- India: services +.016, traders/industries +.014, trust/society +.007, school +.005.
- Everything else is within ±.005. "other" (no org word) is −.035 India and −.068 US.
- Legal form: 'none' +.036 is the only positive class. pvt ltd −.024, llc −.017. Every state is within ±.006.

**WHY.**
- **FP is ≈ 5% of error units** (891 FP vs 19,579 FN). The loss is recall on multi-member clusters.
- **Twins lose to FN_up.** 81% of twin-5+ misses are FN_up.
  - 8–12% of FN_up SX are the argmax pair of another holdout S1: 6–11% one with the same core (a same-name twin).
  - The rest went to a non-holdout competitor S1 or were never ranked.
  - It is an assignment / blocking problem (P49 / P58 territory), not a threshold one.
- **Small groups (0–1 accepted, 1–2 argmax pairs) lose to FN_up the same way.** Their lift is partly tautological: an S1 with
  few members recovered is the loss.
- **The address-less cell is the only FN_below-dominated cell** (4,369 FN_below vs 494 FP), but p54 is calibrated there:

| country | address-less pairs, p band | n | precision | mean p |
|---|---|---|---|---|
| India | [.5, .6) | 301 | .50 | .55 |
| India | [.6, .71) | 244 | .67 | .65 |
| India | [.71, .8) | 170 | .82 | .76 |
| US | [.5, .6) | 1,063 | .51 | .55 |
| US | [.6, .71) | 680 | .60 | .65 |
| US | [.71, .8) | 398 | .76 | .75 |

  Those FN are genuinely uncertain pairs. Lowering their bar adds pairs at ≈ p precision, which is below the break-even for
  an S1 that already holds members.
- **Company kind.** Medical, trust, club and school S1 in the US and services / traders in India carry slightly more loss
  (≤ 1.25× their share), again FN_up-dominated. They are over-represented among twins (generic org-word names shared by many S1),
  not a separate failure mode.

**2. India/US hill climb**: see What failed (every arm drops). Knobs per (country, family, value) with ≥ 2,000 S1:
- a t offset, clipped to 0.71 ± 0.2;
- an extra-member bar for non-top same-source pairs;
- a cap level (S2 ≤ 5 / S3 ≤ 6, or 4 / 5).
The search uses step 0.005, 20–24 random restarts + coordinate ascent, and 10–20 permuted-label nulls per direction
(SageMaker spot c6i / m5.4xlarge).

**3a. France on test (label-free, sub76)** (data/work/p76/frtest.md)
- France accepts 3.20 SX per S1 vs India/US 3.39. Its empty rate is .061 vs .058, 1-member .075 vs .061, one-sided .171 vs .152.
- Every segment (legal form incl. sarl / sas / eurl / sa / sasu / sci / ei, org type, name length, twin, hn) sits inside the
  same gap. Acc ranges 3.08–3.27 vs 3.32–3.44.
- The −0.19 SX/S1 gap matches France's lower SX density: 5.53 SX/S1 vs 5.8, and 3.39 × 5.53 / 5.8 = 3.23.
- No company kind is singled out as under- or over-matched. The largest deviations are 5+-token names (acc 3.08 vs 3.42,
  1-member .090 vs .058) and twins 5+ (1-member .083 vs .066). France's S1 is 99.4% hn, so hn / unit / floor carry no signal there.

**3b. France LOCO (P42 frames; sub66_final chain analogue t .9868 / rescue .9 / gc .7 / hn off, caps 5/6).**
- Knobs = per-segment integer shifts of each stage's bar along the France level grid, for twin, ntok, legal form y/n, S1
  address-less share and # argmax pairs; values need ≥ 2,000 S1 in both targets.
- The density map is P42's rescaled to sub66_final's own France density. P69 measured 0.858–0.876× P42's below-anchor mass.
  The first runs used P42's map, which made the base looser than sub66's chain and inflated every "stricter" gain.
- Nulls: the same search with S1 segment labels permuted, at the same number of restarts.

| arm | map | knobs | chain US→India (fit h0 / h1) | chain India→US | m2 US→I / I→U | null p95 US→I / I→U | verdict |
|---|---|---|---|---|---|---|---|
| global (no segments, unbounded) | sub66 | 4 | +.0032 / +.0034 | +.0015 / +.0012 | +.0015 / +.0006 · +.0023 / +.0018 | — | the tightening you ruled out (t .99, rescue .97, gc .9/.95, hn on .95/.97) |
| global, P42 map | P42 | 4 | +.0037 / +.0044 | +.0017 / +.0015 | | | map-inflated |
| fam_lf2 / n_am / noaddr / ntok / twin (unbounded) | sub66 | 6–17 | +.0026 to +.0035 | +.0013 to +.0036 | | +.0028 to +.0034 · +.0015 to +.0026 | tighten rescue/gc; ≈ global + segment noise on US→India |
| joint (unbounded) | P42 | 53–58 | +.0043 / +.0048 | +.0041 / +.0042 | +.0037–.0041 · +.0057–.0063 | +.0037 / +.0027 (1 restart) | tightens rescue/gc |
| **global_lo** (rescue ≤ .90, gc ≤ .70) | sub66 | 0 moved | 0 | 0 | 0 | — | the recipe is locally optimal |
| **joint_lo** (bounded) | sub66 | 22 / 21 | **−.00007 / −.00045** | **+.0041 / +.0036** | +.0009–.0011 · +.0055–.0056 | −.00009 / −.00018 · +.0010 / +.0008 (n 3) | **fails US→India** |
| joint_lo2 (t ≤ .9868 too) | sub66 | 13 | −.00016 / −.00028 | +.0006 / +.0010 | −.0004 to −.0002 · +.0006 / +.0011 | −.00001 / −.00003 · +.0004 / +.0006 (n 12) | fails US→India |
| ONE best single change (unbounded) | P42 | 1 | +.0021 / +.0015 | −.00014 / +.00003 | | | fails India→US (it is "rescue off" for one segment) |
| **R3: t +3 (.9925) on n_am = 2** (fixed rule, both halves) | sub66 | 1 | **+.00015 / +.00015** | **+.00098 / +.00103** | +.0003 / +.0004 · +.0021 / +.0018 | 8 permuted labels: max −.00017 · +.00011 (m2 +.0003 · +.0005) | **passes, beats null** |

**Retrodiction** (retro.py: each LB-measured step on the same chain analogue).

| step | LB-implied France-F (ΔLB / 0.1498) | chain US→India | chain India→US | m2 US→I / I→U |
|---|---|---|---|---|
| rescue90 | +.0021 | −.0029 / −.0029 | +.0010 / +.0011 | −.0021 / +.0010 |
| gc70 | +.0023 | −.0004 / −.0008 | +.0021 / +.0022 | +.0005–.0011 / +.0019 |

India→US retrodicts both LB steps (gc70 almost exactly). US→India gets both signs wrong. The brief's gate leg "US→India ≥ 0"
therefore leans on the direction the LB contradicts. R3 passes it anyway. joint_lo fails it, but its India→US gain
(+.0041 / +.0036, 4× its null) is the largest LB-consistent LOCO signal seen.

**R3 on sub77 (output/sub82r3).**
- 1,243 candidates → **260 France pairs removed on 259 S1**. 958 were kept as their source's last member, 25 were not on the
  line, and no line goes empty.
- That is 0.0010 removals per France S1, inside LOCO's 0.0006–0.0017 (chain). LOCO's removed pairs are 3–23% true.
- The removed France pairs are CE-healthy, like the whole band (bge r median .998, Qwen median .989 on 35% coverage). CE neither
  supports nor refutes them.
- India/US lines are byte-identical. Validator PASS --check-ids, md5 c1b92237….
- Expected LB ≈ 0.1498 × (removals/S1 × LOCO gain per removal) ≈ **+0.00004 to +0.00009**.

## What failed      (approaches tried and dropped, with the number that killed them)
- **ONE best single-segment t change.**
  - The h0 fit picks US rank-0 pairs −0.035: h1 **+0.000006 ± 0.000010**.
  - The h1 fit picks India address-less pairs −0.025: h0 **−0.000009 ± 0.000009**.
  - The two halves pick different segments. Noise.
- **joint_dek: t offsets + extra-member bars + per-segment caps over all families jointly** (133 / 143 free knobs; SageMaker
  p76-iu-dek-1729; 24 restarts, 20 nulls per direction): fit +0.000296 / +0.000308 vs null p95 +0.000322 / +0.000301. Report
  **−0.000100 ± 0.000038 / −0.000148 ± 0.000044**.
- **joint_d: t offsets over all 9 families jointly, per country** (lf, org, twin, S1 address-less share, ntok, hn, top-10
  state, pair address-less, pair rank; 89 / 88 free knobs; step 0.005; 24 restarts + coordinate ascent; 20 label-permuted
  nulls per direction).
  - Fit gain +0.000236 (h0) / +0.000258 (h1). The **null's fit gain is the same**: mean +0.000244 / +0.000217, p95
    +0.000270 / +0.000235. The real segment labels buy nothing that random labels don't.
  - Report **−0.000051 ± 0.000041 (h0→h1) / −0.000109 ± 0.000046 (h1→h0)**. Null report mean −0.000163 / −0.000112.
- **One family at a time, and caps / extra-member bar alone** (SageMaker p76-iu-rest-1729, ml.c6i.4xlarge spot; 20
  restarts; 10 nulls per direction). Report = LB-eq on the other half, h0→h1 / h1→h0. "fit vs null p95" = the in-sample gain
  against the null's 95th percentile.

| arm (free knobs) | report h0→h1 | report h1→h0 | fit vs null p95 (h0 / h1) |
|---|---|---|---|
| caps_only: S2 ≤ 5 / S3 ≤ 6 or S2 ≤ 4 / S3 ≤ 5 per S1 segment (≤ 2) | +0.000000 | +0.000000 | +0.000002 vs +0.000002 / 0 vs 0 (caps never bind profitably) |
| extra_only: extra bar for non-top same-source members (56) | −0.000055 ± 0.000017 | −0.000049 ± 0.000018 | +0.000139 vs +0.000151 / +0.000117 vs +0.000126 |
| fam_lf (9) | −0.000055 ± 0.000039 | −0.000016 ± 0.000022 | +0.000064 vs +0.000092 / +0.000073 vs +0.000080 |
| fam_org (21) | −0.000131 ± 0.000032 | −0.000087 ± 0.000027 | +0.000101 vs +0.000130 / +0.000096 vs +0.000112 |
| fam_twin (6) | −0.000113 ± 0.000035 | −0.000032 ± 0.000023 | +0.000062 vs +0.000072 / +0.000055 vs +0.000063 |
| fam_noaddr, S1 address-less share (5) | −0.000064 ± 0.000027 | −0.000051 ± 0.000024 | +0.000051 vs +0.000055 / +0.000027 vs +0.000061 |
| fam_pair_na, pair address-less (4) | −0.000030 ± 0.000021 | −0.000030 ± 0.000019 | +0.000042 vs +0.000036 / +0.000050 vs +0.000053 |
| fam_rk, pair rank in S1 = group state (10) | −0.000018 ± 0.000045 | −0.000037 ± 0.000032 | +0.000108 vs +0.000102 / +0.000101 vs +0.000090 |
| fam_ntok (8) | −0.000116 ± 0.000041 | −0.000035 ± 0.000018 | +0.000059 vs +0.000089 / +0.000059 vs +0.000072 |
| fam_hn (3) | −0.000117 ± 0.000035 | −0.000029 ± 0.000013 | +0.000030 vs +0.000039 / +0.000033 vs +0.000037 |
| fam_state, top-10 states (21) | −0.000064 ± 0.000027 | −0.000101 ± 0.000027 | +0.000126 vs +0.000144 / +0.000114 vs +0.000129 |

  Every arm is negative on its report half. Only fam_rk (both fits) and fam_pair_na (h0) fit above the null's p95, and both
  still lose out of sample. The group-state result repeats P56's.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `seg.py`: `segments(names, addr, country, twin_counts)`, `twin_counts(names, c)`, `org_type`, `lf_class`. Label-free, any
  country. It has a self-check.
- `frame.py`: F_{c} / S_{c} parquet (p54 out of fold + segments). 50 s, 1.2 GB.
- `dive.py`: the deep-dive tables and examples. 20 s.
- `hill.py`: India/US knobs (t offset d, extra-member bar e, cap level k) per (country, family, value).
  - Random restarts + coordinate ascent at step 0.005, label-permuted nulls. `selftest` asserts that all knobs at 0 reproduce the base.
  - One eval is 0.9 ms: only the ~8k band / cap pairs and their S1 are re-scored.
- `fr_hill.py`: France chain bars as per-segment level shifts on P42's LOCO frames.
  - `fast_decide` is O(n). It is asserted equal to `solution_p42.decide` under random shifts.
- `frtest.py`: the label-free test view of any submission. 4 min, 1.5 GB.
- `fr_hill.py` arms: `fam_<f>`, `global`, `joint`, and the bounded `*_lo` / `*_lo2` (BOUND: stages never stricter than the
  recipe). `P76_DENS=p42` switches back to P42's density map.
  - The default map is `data/work/p76/fr_dens_sub66.json`: P42's map × the P69 sub66 / P42 ratio, interpolated.
  - `single` runs the one-best-change arm.
- `retro.py` (LB retrodiction), `rules.py` (the fixed rules R1–R6), `r3null.py` (R3's label null). About 1–3 min each, 1.1 GB.
- `build_r3.py <base> <out>`: R3 on any France-lines base. 2 s. It needs `data/work/p76/p69/p69_frame.npz` (from S3
  `amc26-offload/work/p69/t9868/`).
- SageMaker: `launch76.py <name> <instance> <iu|fr> <arm:restarts:nulls,...>` → `solution_p76.py` (sm_train main contract;
  the channel is s3 `amc26-p76/ch/`, repo-relative frames).
  - `fetch76.py <job>` splits the results per arm into `data/work/p76/{hill,frhill}_<arm>.json` and prints the null table.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Analysis only: numpy / pandas / LightGBM predict. No external data.
- Twin counts are fit on each country's own S1 pool: train for the holdout, test for test. Train pools are larger
  (US 1.32M vs test 658k), so the holdout's twin share is somewhat higher.

- France LOCO transfer is the weak point, as in P56 / P69. The two directions disagree on rescue / gc. R3 was read off
  joint_lo's knobs (both independent fits chose +3 / +4 on n_am = 2), so a little selection leaks in. Its 1-knob label null
  and 8/8 sign agreement are the defence.
- n_am (# SX-argmax pairs per S1) depends on candidate density. France has 5.53 SX/S1 vs India/US 5.8, so the n_am = 2 segment
  is a slightly different population in France.
- AWS: 9 SageMaker spot CPU jobs (c6i / m5.4xlarge), ≈ $3 in total. The local laptop reboot at 18:31 lost nothing: every result is on disk or in S3.

## Open questions   (for the master; each with your best guess)
- Should the France gate weight India→US over US→India, given that the retrodiction shows India→US tracks the LB? Best guess:
  yes. Under that reading, joint_lo (+.004 India→US) is the next France lever to study, but it is 21–22 knobs. Only a
  simplification of it (like R3) is safe this late.
- Upload R3 on its own? No. At +0.00004–0.00009 it cannot be resolved on the public LB. Bundle it into the next France
  candidate (e.g. sub81 − R3's 260 pairs, if those pairs are still on its lines), or skip it.
