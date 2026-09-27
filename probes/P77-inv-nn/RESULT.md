# P77-inv-nn — RESULT (draft, 27 Sep 17:10)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- With **country-invariant inputs only**, P57's neural set-matcher **transfers across countries**.
  - Regime: France-like, M2 p only, no CE; label-free t = the base's empty-S1 rate. Numbers are vs M2 at t 0.71.
  - US→India: **+0.00108 ± 0.00016**, India→US: **+0.00190 ± 0.00013**.
  - P57's full-feature net: −0.00008 / +0.00067. P17's GBDT stacker (P17 LOCO): +0.00073 / +0.00203.
- Not yet measured on the yardstick that matters for France: P55's LOCO chain frame (the France recipe analogue +
  P48 veto). That frame's inputs (P42 / P48 / P52 data) are not on this laptop.
- **Proposal:** drop the net in for `p55.fit` (LightGBM) on P55's frames, where they exist. If it beats P55's
  +0.0052 / +0.0046 there, build the France add-on with `p55.py build`.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- Code: `probes/P77-inv-nn/train.py base=m2 inv=1 fit=US|India|India,US`.
- Inputs: P57's `data/work/p57/holdout_m2_<c>.parquet`, from `build.py India US base=m2`.
- Invariant columns (INVC): lp, hn class / delta / eq, P17 support (n_same, p_same, n_s1hn), SX competition (count,
  rank, best other, gap), is_s3, M2's context columns (s1_/sx_ n / gap / margin, block ranks, blocked, rule_acc), and
  the S1's same-source structure. g_src_n = other same-source candidates with p > 0.5; g_src_max = their best logit.
- Net: L=2 attention across the S1's candidates, d 128, 10 epochs, residual on logit p, rank-gauss inputs.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| nn-L2-d128-fitUS_m2-inv | LOCO US→India (India unseen: all S1) | label-free t 0.78: **+0.00108 ± 0.00016**; US in-country h1 +0.00353 | keep (transfers) |
| nn-L2-d128-fitIndia_m2-inv | LOCO India→US (US unseen: all S1) | label-free t 0.77: **+0.00190 ± 0.00013**; India in-country h1 +0.00176 | keep (transfers) |
| nn-L2-d128-fitIndia+US_m2-inv | in-country h1 | +0.00304 ± 0.00015 (India +0.00209, US +0.00358); full-feature P57 +0.0034 | reference |

## What failed      (approaches tried and dropped, with the number that killed them)
- P57's full-feature net in the same regime: US→India −0.00008 ± 0.00018. The raw country-specific features (script
  flags, state / city agreement, name-token overlap, legal forms) do not transfer.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- The M2-only regime is a proxy. France's real chain (c_hn + bge out-of-domain blend + tiers / veto / cap / rescue /
  gc / xfc / P64) already absorbs part of this signal. The gain over sub66_final's France is unknown until the net
  runs on P55's LOCO chain frames.

## Points 1-6 of the loss list (27 Sep evening, on the latest main)
- **1 (address-less same-name twins):** closed. P28-noaddr (sh-noaddr) put the ceiling at +0.000015 F0.5, and the
  net on a random sample was ≈ −0.0001. P23 / P49 agree.
- **2 (gibberish brand, street only):**
  - France: P41's brand recovery is not in sub66_final. Re-derived (+5,742, exactly P41's) and grafted: `output/sub66_brand`
    (+5,729 pairs, 5,629 S1). Validator `--check-ids` PASS; expected LB +0.00005 to +0.00012.
  - India / US: P36 −0.000038 / −0.000063. Closed.
- **3 (one-digit house-number fakes):** the hn_cls gap (a middle-digit drop/add counts as class 4 "far") is real but tiny.
  - Holdout: 104 / 107 such pairs, with 1 (India) / 11 (US) wrong decisions.
  - Ceiling ≈ 0.00003. Not worth a retrain.
- **4 / 5 (France extra same-source copies, Groupe / inserted words):** the gate-passing slices are already staged on the
  team box (t975ce, P70a, P71a, N54, P73e in sub75a / sub76). Their LOCO frames (data/work/p42, p55, p69, p70, p74) are
  not on this laptop, so nothing new was testable here.
  - P77's invariant net passes the M2-regime LOCO both ways (+0.00108 / +0.00190). It is the candidate drop-in for
    `p55.fit` on those frames.
- **6 (heavy rewrite, never shortlisted):** P68's retrieval tier got +0.00004. Ceiling small. Closed.
- Note: main's P57-iu-transfer shares `data/work/p57` with sh-neural's P57-neural inputs. The file names differ, so
  nothing clashes.
