# P35-twin-propagate — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Drop: the twin hypothesis is false on labels.** An address-less S2/S3 record whose near-identical addressed twin is
  confidently matched to S1 X belongs to X only **31% (India) / 26% (US)** of the time on the #8c holdout. The
  break-even is 0.67–0.75. The rest belong to a *different same-name S1*: 38–44% among their candidates, 27–29% outside
  them. Only 2% are distractors.
- Every form loses or adds nothing. Propagate rules: −0.00002 to −0.0025 LB-eq. Conflict guard: −0.00006. Blocking-miss
  propagation: 1–4% precise. LOCO India→US: −0.0005 to −0.0055. Group-decision GBDT: +0.00030 / +0.00024 LB-eq, but a
  twin-free context arm matches it and re-assignment adds +0.000016. P35 columns on P34's `twin_p2mask`: +0.00001 / −0.00004.
- **Recommendation: nothing to integrate, and no sub35 was built (gate failed).** P34's a10 stacker already takes the
  only real signal. Stop spending on the address-less bucket: its recoverable part is name-only look-alikes the matcher
  already scores correctly as ambiguous.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Adopt nothing from P35.** No India/US or France file, and no `output/sub35`.
  - The India/US gate needs ≥ +0.0003 at ≥ 3 SE, carried by twin reasoning. The best arm clears +0.0003 pooled in one
    direction only (h0→h1). The gain comes from competitor-p context columns that P34 already ships, masked there. The
    twin columns add ≈ 0.
  - The France gate needs LOCO ≥ +0.0005 in both directions. India→US is negative for every twin rule.
- **For P34 / P9:** do not add twin or S1-load columns to the a10 stacker. They were measured on its exact recipe, with
  no gain (table below).
- **For the error budget (P23-research row "address-less"):** read the stolen part of the address-less bucket as a holdout
  artefact.
  - The "thief" is a competing train S1 in 84–88% of stolen address-less SX. Its row is e5-blended in 82–87% of them;
    test is all bge.
  - Its p is low anyway: median 0.26–0.29, and only 2–4% are ≥ t.
  - Re-assignment gains measured on the holdout frame therefore do not transfer. The underlying loss (low-p name-only
    records among same-name S1) would show on test as low-p FN, and no label-free signal separates those S1.

## Evidence         (table: ledger name | world | key metrics | verdict)
All on the #8c holdout yardstick (`data/work/p23` dump; `ev.py` reproduces #8c exactly: India 0.99048 / US 0.98951,
half 1 0.99004 / 0.98924) unless noted. LB-eq = 0.468·India + 0.383·US. "h0→h1" = fit or tune on half 0, report half 1.
The twin is an addressed SX of the same candidate S1, at its argmax with p ≥ 0.71, and Indel(core) ≥ 0.85.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| twin-rule-A-precision | #8c holdout | raise-pool (address-less argmax, p < t, twin): India 4,089 pairs, **precision 0.308**; US 10,787, **0.260**. p 0.5–0.71 with one supported S1: 0.678 / 0.619. Switch-pool (non-argmax + twin): 0.083 / 0.094. Guard-pool (accepted, no twin): 0.973 / 0.953 true | drop |
| twin-negatives-where | #8c holdout + train truth | raise-pool negatives: other S1 among candidates 0.378 / 0.437, other S1 not a candidate 0.291 / 0.275, distractor 0.023 / 0.027. Address-less SX overall: distractor 2.0% / 1.9% (addressed SX: 13.8% / 12.4%) | — |
| noaddr-oracle-headroom | #8c holdout | oracle over address-less SX: accept/reject argmax only +0.00186 / +0.00310; re-assign among candidates +0.00358 / +0.00494 | — |
| noaddr-budget-artefact-aware | #8c holdout | oracle-fix per category (India / US): low-p FN +0.00158 / +0.00239; **stolen with e5-blended thief +0.00152 / +0.00154**; stolen with clean thief +0.00024 / +0.00037; FP +0.00031 / +0.00078. 92% / 87% of raise-pool pairs already have SX core ≈ S1 core (sim ≥ 0.95), so the twin adds no link the matcher lacks | — |
| stolen-noaddr-anatomy | #8c holdout | stolen address-less SX 1,723 / 2,468. The thief is a train S1 in 0.879 / 0.836 of them, with an e5 row in 0.868 / 0.824; thief p median 0.26 / 0.29. A twin sits at the true S1 only (not at the thief) in 0.536 / 0.326 | drop |
| twin-rule:propagate-* | #8c holdout | p := 0.999 for twin-supported candidates with p ≥ floor. LB-eq h0 / h1: floor 0.3, one supported S1: −0.00134 / −0.00117; floor 0.5: −0.00023 / −0.00022; floor 0.6: −0.00003 / −0.00002 | drop |
| twin-rule:conflict-guard | #8c holdout | drop an accepted pair without a twin when another candidate has one: −0.000055 / −0.000074 | drop |
| twin-propagate-blocking-misses | #8c holdout + train corpus | same-core address-less SX that are not candidates of the twin's S1: 68,915 / 36,408 pairs, **precision 0.012 / 0.016**. Best slice (unique core, unassigned, one claimant): 0.031 / 0.040, dF0.5 −0.0026 / −0.0032 | drop |
| loco-twin-rules | LOCO (P6 M2 frames, t_profile 0.81 / 0.97) | India→US: raise-pool precision 0.319; A_all −0.00553, A p ≥ 0.5 −0.00073, guard −0.00013. US→India: A_all −0.00412, A p ≥ 0.5 +0.00099. The t 0.97 there is strict: [0.7, 0.97) pairs are 92% true with or without twins. Fails both-directions | drop |
| gdec-gbdt:* | #8c holdout, pooled fit | residual LightGBM on every candidate row of address-less SX, then re-argmax, t 0.71 (table below) | drop |
| gdec-decomp | #8c holdout | full arm h0→h1, LB-eq +0.000305. Accept/reject on the same argmax: +0.000289. **Re-assignment: +0.000016** (swap +0.000003). SX without an e5 row: +0.000113; SX with an e5 row: +0.000193 | drop |
| a10-twin_p2mask+P35-columns | #8c holdout, a10 frames | P34's arm reproduced: +0.000308 ± 0.000064 (h0→h1, t 0.71), +0.000227 (swap). + P35 columns: +0.000319 ± 0.000068 / +0.000192 | drop |
| empty-s1-rescue | #8c holdout | an empty S1 takes its best candidate at p ≥ t_r (0.4–0.65): LB-eq −0.00001 to −0.00038 in every variant | drop |

Group-decision GBDT arms (pooled India + US per-S1 mean, ± SE; LB-eq in brackets):

| arm | columns | h0→h1 | h1→h0 |
|---|---|---|---|
| full | 18 (p ctx + twin + S1 load + group) | +0.000392 ± 0.000052 [+0.000305] | +0.000279 ± 0.000046 [+0.000244] |
| notwin | full − 5 twin columns | +0.000412 ± 0.000052 [+0.000323] | +0.000264 ± 0.000047 [+0.000232] |
| twin_only | lp, rank, margin, n_cand + twin | +0.000496 ± 0.000058 [+0.000400] | +0.000258 ± 0.000052 [+0.000227] |
| **ctx_only** | **lp, rank, margin, n_cand** | **+0.000465 ± 0.000056 [+0.000373]** | **+0.000283 ± 0.000051 [+0.000250]** |
| nomargin | full − margin / rank / x_best | +0.000107 ± 0.000041 [+0.000089] | +0.000027 ± 0.000045 [+0.000012] |
| nomargin_notwin | nomargin − twin | +0.000129 ± 0.000043 [+0.000103] | −0.000001 ± 0.000046 [−0.000009] |

Per country, full arm h0→h1: India +0.000259 ± 0.000073, US +0.000480 ± 0.000071. Swap: +0.000311 / +0.000257.
- `ctx_only` equals or beats every arm with twin columns, so the gain is the SX's competitor-p context. That is P23's
  a10 CTX, which P34 ships.
- Remove the competitor-p columns (which lean on e5-blended competitor rows that exist on holdout only) and the gain
  falls to +0.0001 / 0.0000. Twins do not restore it.

## What failed      (approaches tried and dropped, with the number that killed them)
- **(a) Propagate: raise / accept the twin-supported candidate.**
  - Raise-pool precision is 0.31 / 0.26.
  - Even the p ≥ 0.6 slice loses (−0.00002 LB-eq).
  - In LOCO India→US it is −0.0005 to −0.0055.
- **(b) Conflict guard.** The pairs it drops are 95–97% true, so −0.00006 LB-eq. In LOCO −0.00013 / −0.00021.
- **Switch the argmax to the twin-supported S1.**
  - Switch-pool precision is 0.08–0.09.
  - In the GBDT, re-assignment adds +0.000016 LB-eq.
- **Twin propagation into blocking misses.** 1–4% precise.
- **Residual GBDT with twin columns.** Twin columns add nothing over `ctx_only` or `nomargin`, and nothing over P34's
  a10 `twin_p2mask`.
- **Cross-source vs same-source twins.** Precision 0.31 vs 0.29 (India) and 0.27 vs 0.24 (US).
- **Exact vs fuzzy core.** Indel ≥ 0.999: 0.30 / 0.26. 0.85–0.95: 0.35 / 0.29.
- **Why it fails.** The data holds many same-name S1. An address-less record's name matches its twin's and the S1's
  equally well (sim ≥ 0.95 in ~90% of cases). So the twin carries no information the name-only matcher lacks, and the
  matcher's low p on these pairs is correct.
- **Empty-S1 rescue** (an adjacent idea, same bucket). −0.00001 to −0.00038 in every variant: empty #8c S1 with a
  0.4–0.71 candidate are mostly true singletons.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `frame.py [India US]` builds `data/work/p35/ho_<c>.parquet`: the p23 pairs plus src, noaddr (empty or NULL
  placeholder), core, city and hn. About 1 min and < 1.5 GB per country.
- `twins.py`:
  - `twin_frame(d, am, tau1=0.9, tau_sim=0.9)` returns per-(s1, sx) twin support: tw_p, tw_sim, n_tw, tw_x.
  - `argmax_mask(sx, p)`.
  - The `__main__` self-check passes.
- `ev.py`:
  - `load(c)`, `f_of(S, pred)` and `gain(S, pred)`: paired per-S1 F0.5 on the #8c yardstick.
  - `__main__` asserts the #8c reproduction.
- `gdec.py`:
  - `features(d, core1, t)` builds the 18 label-free group columns on address-less rows.
  - `decide(d, x, p_new, t)` re-argmaxes address-less SX.
  - The `__main__` self-check passes.
  - The features cost ~5 s per country.
- `fit.py feats | eval <arms>`: the GBDT arms, about 2 min per arm pair.
- `loco.py`: twin rules on the P6 LOCO frames, ~40 s and ~1.5 GB per direction.
- Analyses: `a1_profile`, `a2_prec`, `a3_where`, `a4_oracle`, `a5_decomp`, `a6_stolen`, `a7_budget`, `a8_block`,
  `a9_rules`, `a10_plus` (imports P23's `a10_stack` from the main checkout), `a11_rescue`.
- Logs: `probes/P35-twin-propagate/logs/*.log`, aggregates only.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Nothing ships, so there is no rule risk. rapidfuzz and LightGBM are MIT. No external data.
- **Assumptions:**
  - "Address-less" means the raw address is empty or a placeholder (NULL, <NULL>, n/a, -). That is 10.5% of India and
    12.7% of US SX in the frame.
  - A record whose address is only a state was treated as having an address. The conclusion does not hinge on this: the
    precision is flat across every twin slice.
- **Holdout-frame caveats** (these apply to any re-assignment study on the p23 dump):
  - Competing train-S1 rows carry e5-blended p in the band, 29–32% of competitor rows.
  - Competing S1 have partial candidate lists in the frame, so S1-load features are undercounted for them.
  - Errors on competing S1 are not scored.
  - In-sample folds 2–5 look the same as out-of-sample folds on band share (0.289 vs 0.295 India).

## Open questions   (for the master; each with your best guess)
- **Is any address-less lever left for India/US?**
  - Best guess: only what P34's a10 already takes.
  - The remaining oracle room is ambiguity among same-name S1 with no text signal: low-p FN +0.0016 / +0.0024, stolen
    +0.0018 / +0.0019.
  - Blocking for address-less records (P24's domain) is the only other bucket, and exact-core twin propagation into it
    is 1–4% precise.
- **Does P23's "address-less = 59% / 71% of the loss" need a correction?**
  - Best guess: the total stands, but not its split. 41% (India) / 30% (US) of the on-candidate address-less room is "stolen by an
    e5-blended train-S1 row", which test does not have.
  - On test those records most likely become low-p FN. That loss is not recoverable label-free.
