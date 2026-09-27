# P80 stage C: rival-style re-decision on honest out-of-fold M2 scores

`stagec.py`. Post-deadline research. It touches nothing in `output/`. CE_NOTES.md covers the CE cross-fit, which is
separate work.

## Scores feeding stage C (stage 1: solution_p80.py, jobs p80-cfA-r-0927-2123 / p80-cfB-r-0927-2123)

M2 (model.PARAMS, 65 features) is cross-fit twice. m2_cf_A is fit on folds {2,3} and scores {4,5}. m2_cf_B is fit
on {4,5} and scores {2,3}. Both early-stop on a 25% sample of fold 1 (about 3.6-3.7k trees, fold-1 logloss 0.0241;
m2_big on 8 folds reaches 0.0229). Neither booster saw folds 0, 1 or 6-9, so those folds take the **logit-space mean
of both**, the same rule `score_test.py` applies to test. Every pair's p is out of sample for that pair.

## Frame (`stagec.py frame <c>` -> data/work/p80/sc_<c>.parquet)

One row per pruned candidate of every fold 0-9. The pruned set is the P4 X pairs with p >= 0.01: 30% of the rows,
99.9% of the true pairs.
- ids and labels: q1, qx, fold, y, s3 (source 3), p
- M2 pair features: name_tset, core_tset, addr_tset, hn_eq
- anchors: a_n = the S1's other candidates with p >= .9. a_name / a_addr = the best rapidfuzz token-set ratio of this
  SX's P3 name / address to those anchors (address is NaN when empty).

`feats()` derives these from the frame at fit time (34 columns):
- within the S1: rank, size, gap above and below, best other p, p-mass of the others
- within the S1 and source: rank, size, size of the other source
- within the SX: rank of this S1, number of S1, best competitor p `bx`, margin `mx = logit p - logit bx`, competitors
  with p >= .5, competitor mass
- rival pair (the SX's best other S1): its rank inside the rival S1, the rival S1's size and anchor count, the rival
  pair's anchor similarities, and own-minus-rival differences of the four pair features and of a_name.

## Stage C and decision (`stagec.py fit` -> stagec.json, boosters data/work/p80/sc_C{A,B}_<c>.txt)

- The model is a LightGBM per country (lr .05, 63 leaves, min_data 200, ff/bagging .8), early-stopped on fold 1.
- It is cross-fit the same way as stage 1. C_A is fit on {2,3}, C_B on {4,5}. Folds 0, 1 and 6-9 take the mean of
  both. So every competitor's stage-C score is out of sample too. This matters because 90% of an SX's rival S1 sit
  in other folds.
- Owner re-decision: each SX goes to its best-scoring S1 among all its pruned candidates. The pair is kept at s >= t,
  with caps of 5 S2 / 6 S3 per S1, best first. t is tuned on fold 1, using per-S1 F0.5 over every fold-1 S1 (n_true
  from gt_pairs).
- Report: per-S1 F0.5 on all fold-0 S1 (88,168 India / 132,190 US), paired against today's decision. That decision
  is the P76 frame: sub54's p54 >= .71, out of fold by half. On fold 0 it scores India 0.991369 and US 0.990507.
  stageB is the same owner-and-threshold rule applied to the M2 p alone.

## Results

Fold 0 covers all 88,168 India and 132,190 US S1. Deltas are per S1, paired, mean ± SE. LB-eq = 0.468 India + 0.383 US.

| | India | US | LB-eq |
|---|---|---|---|
| today (p54 >= .71) F0.5 | 0.991369 | 0.990507 | |
| stage B (cf-M2 argmax, t .72 / .73) | 0.982107 | 0.983095 | |
| **stage C** (t .68 / .70) | 0.984001 | 0.984242 | |
| **C - today** | **-0.00737 ± 0.00021** | **-0.00626 ± 0.00016** | **-0.00585 ± 0.00012** |
| C - today, half 0 / half 1 | -0.00756 / -0.00717 | -0.00663 / -0.00590 | |
| C - B | +0.00189 ± 0.00016 | +0.00115 ± 0.00010 | +0.00133 ± 0.00009 |

- **Verdict: it fails the gate.** Stage C recovers about 18% of the 0.0072 LB-eq gap between cf-M2 and today's decision.
  The rest is the CE: #8c minus M2 alone is about +0.007 on fold 0 (P44). Stage C is not a candidate without CE features.
- Gain share: lp .78, mx (margin vs the rival) .16, then up1, a_addr (anchor address), addr_tset, r1s and dn1 at about
  1% each. Early stopping ends at 245-413 trees.
- Stage C moves the SX owner on 11% (India) and 10% (US) of pruned rows relative to M2 argmax.
- **Hybrid** (`stagec.py hybrid` -> hybrid.json): today's p54 decision edited by stage C, with the knob tuned on one fold-0
  half and reported on the other. No arm helps.
  - veto drops pairs with s < v: LB-eq -0.000012 / -0.000006.
  - steal-veto drops pairs whose SX stage C gives to another S1: -0.000029 / -0.000039.
  - rescue adds argmax pairs that stage C owns with s >= u: -0.000075 / -0.000080.
- Check: my fold-0/1 logit-mean p matches the coordinator's AWS p80-sc01 scores to 1e-5.
- Cost: $0 of my own. Everything ran on the laptop:
  - frame: 24 min India / 31 min US, run in parallel. Most of it is scoring folds 0, 1 and 6-9 with two ~3.7k-tree
    boosters on 8 threads.
  - anchors: 11.7M / 17.6M pairs via rapidfuzz cpdist in about 20 s.
  - fit: 2 min. Peak RSS is 2.1 GB for the frame and 2.85 GB for the US fit.
  - disk: sc_India / sc_US parquet 69 / 105 MB, kept in data/work/p80 for the CE step.

## Stage C + CE (`ce_r`, `fit --ce`, `hybrid --ce`, `blend --ce`)

**Where the CE scores come from.** The p80ce cross-fit supplies r: CE_A is out of fold on folds 4-5 and CE_B on 2-3,
and folds 0-1 take the A/B logit mean.
- `ce_r` reads them straight from S3, keeping only the frame's rows: `oof/`, or `r/` by entity id while `oof/` is
  not written yet.
- Folds 6-9 stay NaN until the p80cs jobs land. `ce_cs <c> --jobs ...` then writes sc_r69_<c> for train and sc_rT_<c>
  for test.

**Six extra columns:** lr = logit r; its rank in the S1 and in the SX; rg1 = the S1's best lr minus lr; rmx =
margin vs the p-rival's lr; rmx2 = margin vs the SX's best other lr.

**India, fold 0** (88,168 S1; lr covers folds 0-5 fully, folds 6-9 not at all):
- **Stage C** (t .69) scores 0.991310 against today's 0.991369: **−0.000059 ± 0.000088**. By half: h0 +0.000035 ±
  0.000129, h1 −0.000154 ± 0.000121. Against stage B: +0.00920. So CE closes the whole gap to today, but no more.
- **Hybrid edits** of today's decision (veto / steal / rescue) are all within ±0.00004, and within 1 SE.
- **Blend** (`blend`): q = w·logit(p54) + (1−w)·logit(stage-C p) on fold-0 pairs. A pair with only one score keeps
  that side's logit, and rivals in other folds keep stage C. Then the owner re-decision and caps; (w, t) is tuned on
  one half and reported on the other (blend_ce_India.json):

  | w | h0→h1 | h1→h0 |
  |---|---|---|
  | 0 | −0.000273 ± 0.000121 | +0.000056 ± 0.000130 |
  | .5 | +0.000054 ± 0.000101 | +0.000354 ± 0.000105 |
  | .7 | +0.000109 ± 0.000090 | +0.000283 ± 0.000100 |
  | 1.0 | +0.000112 ± 0.000066 | +0.000297 ± 0.000079 |

  w = 1.0 leaves p54 untouched: stage C only scores the pairs p54 doesn't cover (non-argmax pairs and rivals in other
  folds) inside the owner re-decision. It matches w = .7, so the gain comes from re-deciding the owner around p54
  rather than from averaging the two models. The h0→h1 direction is weak (1.2-1.7 SE).

**US, fold 0** (132,190 S1).
- **First fit**, with no CE score on folds 6-9 (blend_ce_US_nan69.json):
  - Stage C alone: −0.000589 ± 0.000089.
  - Blend w .7, half-tuned: +0.000352 ± 0.000085 / +0.000334 ± 0.000080.
  - Blend w 1.0, half-tuned: +0.000248 ± 0.000069 / +0.000303 ± 0.000044.
- **Refit** with CE scores on folds 6-9 (blend_ce_US.json, stagec_ce_US.json). Those scores are CE_A-only, from
  cs_US_A_only.parquet (the p80cs ckpt-A pass; test uses the same CE_A-only scores), so CE coverage is 100% on train
  and test and the rival features match:
  - Stage C alone: −0.000548 ± 0.000088.
  - **Blend w 1.0, t .72 tuned on all of fold 0: +0.000287 ± 0.000033 (h0 +0.000301, h1 +0.000274).**
  - Blend w 1.0, half-tuned: +0.000216 ± 0.000072 / +0.000301 ± 0.000048.
  - Blend w .7, t tuned on all of fold 0: +0.000338 ± 0.000056.

**US test (`w1_test US` -> data/work/p80/w1_US_pairs.parquet, w1_test_US.json).**
- Today's shipped P70 ta p on #8c's argmax pairs, and stage C + CE on the rest. Then the owner re-decision, caps 5/6
  and t .72.
- Result: 2,250,236 pairs over 624,727 S1. That keeps 2,247,909 of sub89's 2,248,333 ta-rule pairs, adds 2,327 and
  drops 424.
- sub89's US lines = sub88's = the ta rule plus 5 dups (iu_dup_sub88.parquet).
- The coordinator builds output/sub90u on sub89 (US lines only).
- India is not shipped tonight. Its fold-0 blend passes on both halves, but its test CE scores are not in yet.
- Expected US-only LB-eq ≈ 0.383 × 0.00029 ≈ +0.00011. Its t is tuned on fold 0 itself.

## Rehearsal (`--ho`: fold 0 alone, m2_big p, 6 pseudo folds by S1 hash; stagec_ho.json)

This ran before stage 1 landed. Only fold-0 competitors exist in it, so B and C both miss the stealing by other
folds. Only the C-vs-B column is informative.

| | B vs today | C vs today | C vs B |
|---|---|---|---|
| India (14.6k S1) | -0.00907 | -0.00745 | +0.00163 ± 0.00034 |
| US (22.2k S1) | -0.00717 | -0.00672 | +0.00046 ± 0.00028 |

Gain share goes to lp .68 / .59 and mx .26 / .34. The anchor address similarity a_addr, up1/dn1 and addr_tset come
next.

## Next steps to a test file

1. **Test scores: done.** The coordinator's p80-sct job wrote
   out/p80-sct-g4dn16xlarge-0927-2143/sct_test_<c>_sx_*.parquet, with the cf boosters' logit mean. That is the same
   distribution as fold 0 here.
2. **Test decision.** `stagec.py test <c> --test-p out/p80-sct-g4dn16xlarge-0927-2143` builds the test frame from the
   test X files, prunes at p >= .01, computes anchors on P3 txt_test and scores with the mean of sc_CA / sc_CB. It then
   runs the owner re-decision with fold-1's t and the caps, and writes sc_test_<c>.parquet with (s1, sx, s). This has
   not been run on the full test set, because the gate failed.
3. **CE integration (the next step).** Today's decision is ahead because of the bge CE; #8c minus M2 alone is +0.007 /
   +0.006 on fold 0. The p80ce cross-fit (CE_NOTES.md) scores every X pair of folds 0-5 out of fold. Join its r onto
   sc_<c> by (q1, qx), then add as stage-C columns:
   - lr = logit r
   - r's rank within the S1 and within (S1, source), and its gap to the next pair
   - margin vs the rival's r, and r-mass on the SX side
   Folds 6-9 have no r. Score them with both CE models (the mean, as M2 does here), or leave them NaN; NaN weakens the
   rival-r features for about 40% of rivals. Then rerun `fit` and `hybrid`. `apply_test` needs the test r joined the
   same way.
4. **France.** The per-country boosters do not apply there. France needs a pooled, country-invariant stage C, gated
   the way P55/P70 gate their transfers.
