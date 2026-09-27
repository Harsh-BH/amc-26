# P58-twin-cands — RESULT (final, 27 Sep ~06:10 IST)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Drop.** Twin candidate expansion does not open the address-less floor. The brief's premise is mostly wrong: when the
  owner shares the SX's key (groups of ≤ 10), it is **already a candidate 91% of the time**. The "38–43% outside the set"
  splits into owners under a *different* key (≈ 19–20% of owned SX) and groups too large to decode (K > 10).
- The best decode on the expanded sets is **+0.000019 ± 0.000024 / +0.000001** (raw arm) and **−0.000004 / +0.000034 ± 0.000014**
  (raw + #8c p arm) LB-eq, both directions. The expected-F0.5 decode comes out −0.00006. The gate is +0.00015 at 3 SE.
- **Why it cannot pass:** 87–95% of the oracle sits on raw-identical twins. The whole raw-differing slice is worth ≤ +0.00020
  LB-eq *with perfect decisions*, and new-owner pairs ≤ +0.00018. **No sub58 was built. Stop spending on this bucket.**

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Nothing.** No output/sub58, no validator run, no change set to P50. The gate failed in both directions and on every arm.
- For the approach doc: the address-less same-name bucket is a measured Bayes floor.
  - Twins share the normalised name, and in 87–95% of the oracle's SX the raw name too. S1 addresses are always present.
  - Only the group-size prior separates them, and it is weak (P49 T1).
  - Blocking is not the bottleneck: in groups of ≤ 10, the owner is in the candidate set 91% of the time given it shares the key.

## Evidence         (table: ledger name | world | key metrics | verdict)
**Frame (`frame58.py`).** Every address-less train SX (P35 placeholders), expanded to every train S1 whose P40 key equals the key
of the SX's #8c argmax S1 (or the SX's own key if it has no argmax). Groups of 2–30 are kept. India has 27,248 SX and 178,606 rows
(+62,153 new pairs); the US has 48,174 SX and 243,593 rows (+72,137).

**Base.** sub44r out of fold on both halves (P49 `swapfit`). It stands in for sub54's India/US decision, because sub52_max's zoo
combiner was refit on the full holdout (no OOF), and P50 measured it within +0.00004 of sub44r. Deltas are per-S1 F0.5 on the
report half with paired SE; LB-eq = 0.468 India + 0.383 US.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| p58-twin-expansion-coverage | holdout (train keys) | Owner in the key group: India 0.808, US 0.814 of owned SX (98% of these SX are owned). **Given the owner is in the group, it was already a candidate 0.905 / 0.950.** By group size K (both countries): set given group is 0.91 for K ≤ 10, 0.45–0.50 for K 11–30, 0.19 / 0.09 for K 31–100, 0.03 for K > 100. 7,666 India / 11,474 US SX sit in groups > 30 (not expanded) | drop |
| p58-decode-raw | holdout, sub44r OOF | LightGBM on twin rows (raw-name distances, closest-twin uniqueness, case, length, group size, in-set, size prior r(k), counts), OOS per half; best twin iff s ≥ τ (τ_e for empty twins). **h0→h1 +0.000019 ± 0.000024; h1→h0 +0.000001 ± 0.000025**. India −0.00001 / −0.00004, US +0.00006 / +0.00006 | drop |
| p58-decode-rawp | holdout, sub44r OOF | the raw arm + #8c logit / gap / argmax flags (NaN on new pairs). **h0→h1 −0.000004 ± 0.000032; h1→h0 +0.000034 ± 0.000014 (2.4 SE)** | drop |
| p58-decode-ef-rawp | holdout, sub44r OOF | Per twin, the expected F0.5 gain from its label-free state (other accepted pairs, λ missed members, singleton probability for empty twins), with (λ, ps) tuned on the fit half. **−0.000067 ± 0.000046 / −0.000056 ± 0.000043**. On the holdout, 96.5% of base-empty S1 are true singletons, so empty twins should never receive | drop |
| p58-extra-token-and-s1addr | holdout frame | Twin S1s with no address: 0 (every S1 has one). SX-name tokens outside the key found in a twin's address own the SX at 0.134 / 0.169, **below** the base rates of 0.149 / 0.201 | drop |
| p58-oracle-split | holdout, sub44r OOF | Oracle re-decision of the frame: India **+0.002359**, US **+0.002912**. Raw-identical twins: +0.002233 / +0.002537. Raw-differing twins: +0.000126 / +0.000375, a perfect-decision ceiling of **+0.00020 LB-eq**. Owners that are new pairs: +0.000247 / +0.000161, a ceiling of **+0.00018 LB-eq**. By K: ≤ 3 +0.0013 / +0.0018, 4–10 +0.0008 / +0.0008, 11–30 +0.0003 / +0.0003 | drop |

**Calibration (`calib58.py`, 2-fold SX-hash OOS).** The best-twin score is well calibrated: the rawp bucket 0.8–0.9 is 0.92 / 0.89
precise, 0.7–0.8 is 0.82 / 0.75. Only 45 India / 125 US holdout decisions score ≥ 0.8, and at ~0.1% of holdout S1 × ≤ +0.06 per
decision the lever cannot exceed ~+0.00005. The unique raw-closest twin now owns the SX at 0.742 India (n 469) / 0.610 US
(n 2,063) unconditionally, and 0.866 / 0.822 when the owner is in the group. P49 measured 0.727 / 0.479; the owner-in-set
fraction rose, but the slice stays small.

**Test candidate-set size (`testcnt58.py`, label-free, own key, upper bound).** The expansion would add ≤ 61,257 India, 58,619
US and 46,107 France candidate rows: +0.076 / +0.088 / +0.178 per S1 against 4.42 today. On train, 65–70% of such rows are already
candidates, so the net increase is about a third of these bounds.

## What failed      (approaches tried and dropped, with the number that killed them)
- **Candidate expansion (task 1).** It moves owner-in-set from 0.905 / 0.950 to 1.0 only for same-key owners. That adds 3,301
  India / 4,976 US owned SX with a new-pair owner. Their oracle ceiling is +0.00018 LB-eq, and their twins are mostly raw-identical.
- **Threshold decode (task 3, flat τ / τ_e).** It reached +0.00002–0.00003 LB-eq on one half each. The ≥ 0.8 slice is too small.
- **Expected-F0.5 decode.** It lost −0.00006. The per-twin gain formula over-accepts non-empty twins, and an empty twin is
  almost always a singleton (0.965).
- **Location tokens and S1 address status** carry no signal (see the table).
- **Joint component decode (P49 T2).** Not re-run: T2's gain over per-SX decisions was +0.000004, which cannot close a 0.0001 gap.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
All scripts run from the repo root via `probes/P58-twin-cands/py.sh` (1.3 GB cap). Work files are in `data/work/p58/`; logs in `logs/`.
- `frame58.py [c]` writes `ex_{c}.parquet`: one row per (address-less SX, same-key S1) with sx, s1, src, kfull, in_set, p, am,
  holdout, y, the raw names, own_in and old_own_in. 12–15 s, ~1 GB. `noaddr_sx(c)` streams a country's address-less S2/S3 records.
- `cov58.py`: owner coverage by group size (includes groups > 30). About 40 s.
- `d58.py [arm]`:
  - `load(c)` returns (base state, featured frame);
  - `feats(c, ex, cnt)`: the twin-row features;
  - `apply(d, ex, s, tau, tau_e)` and `apply_ef(d, ex, s, lam, ps)`: per-S1 F0.5 after re-deciding;
  - writes `d58.json`. About 60 s.
- `ef58.py [arm]`: the EF decode, writes `ef58_{arm}.json`. `calib58.py`: calibration. `split58.py`: oracle split. `a58.py`:
  token / S1-address check. `testcnt58.py`: test counts.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Nothing ships. rapidfuzz and LightGBM are MIT; no external data.
- **Base.** sub44r OOF stands in for sub54's India/US decision on the holdout. The difference is P50's zoo refit on the full
  holdout, measured at +0.00002–0.00004. sub44r accepts only 56 India / 135 US frame rows at holdout S1s, so the stand-in does not change the verdict.
- **Expansion key.** SX outside the P23 dump use their own key (their #8c argmax is not stored). A typo'd SX whose argmax lies
  outside the dump is therefore not expanded. The oracle split shows the decision, not the coverage, is binding.
- **In-sample #8c p on competitor twins.** It inflates the rawp arm for non-holdout twins; the raw arm, without p, is just as flat.
- **Counts for competitor twins.** The proxy is truth-addressed, as in P49 (label-free count matches it 98–99%). It favours the
  prior, and the prior still adds ≈ 0.
- **France.** Not evaluated: the lever fails on labelled countries, so no France LOCO was run.

## Open questions   (for the master; each with your best guess)
- Where does the leader's India/US edge come from, if not here? Best guess: addressed pairs (the combiner / CE stack), or France.
  The address-less same-name bucket gives at most +0.0002 LB-eq to *any* label-free decode, because the separable part is small.
- Is ~20% of owned twin SX belonging to a different-key S1 (a "stolen" name) a lever? Best guess: no. The SX's name exactly
  equals the twins', so every name signal points away from the owner.
