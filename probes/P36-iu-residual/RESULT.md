# P36-iu-residual — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **The gate fails, so there is no `output/sub36`.** Both briefed levers fail on the holdout. (a) Blocking-miss recovery
  for the top with-address pattern (synthetic-brand SX): LB-eq **−0.000038 ± 0.000011** (h0→h1) and −0.000063 on the
  swap. (b) Expected-F0.5 decoding on a10's p': **+0.000102 ± 0.000058 (1.8 SE)** h0→h1 and −0.000070 on the swap.
- **Post-a10 budget (LB-eq):** low-p FN 0.0021, stolen FN 0.0016, blocking miss 0.0011 with an address and 0.0012
  without, pruner 0.0011, FP 0.0012. What is left is ambiguous in the data itself: name-only SX shared by same-name S1
  (70–90% of low-p / stolen), synthetic-brand renames next to decoys at the same address, and same-name sibling
  decoys (the FPs). a10's p' is already calibrated on every one of these classes.
- **Keep #34's India/US as it is.** Hand-off for France: test France assigns only **54.8%** of its 31k synthetic-brand SX,
  against India 89.5% and US 73.3%. Non-brand SX are assigned at 56–58% in all three countries.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **India / US: adopt nothing.** `output/sub34v_rescue90` stays the India/US base. No sub36 was built and the validator
  was not run, because the gate needs ≥ +0.00015 at ≥ 3 SE in both directions and no arm came close.
- **Tool for a France probe (not for run.py yet):** `probes/P36-iu-residual/gib.py`.
  - `learn(split) -> list[str]` learns the generator's 44 brand syllables with no labels and no word list (the
    k-core of a syllable-glue graph over SX name tokens that no S1 name uses). Train and test give the identical 44.
  - `is_gib(names, M) -> bool array` flags a name with a token spelled by ≥ 2 syllables.
  - Cost: `learn` ~80 s and < 1 GB per split. `is_gib` ~10 s per million names.
  - `recover.cands(split, c, skip)` finds the address top-10 S1 of the unassigned brand SX, with 15 address / hn
    features. It takes 20–80 s per country and 1.5 GB at most.

## Evidence         (table: ledger name | world | key metrics | verdict)
All rows are on the #34 India/US decision (#8c + P34's guarded a10), made **out of fold** on the holdout. The S1 of half h
take p' from the P34 booster fit on half 1−h. `base.py` reproduces P34's half-1 F0.5 exactly (India 0.990200, US
0.989601). Whole-holdout F0.5: India 0.990615, US 0.989850. LB-eq = 0.468 · ΔIndia + 0.383 · ΔUS.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| error-budget-after-a10 | holdout (#34 oof) | table below | keep |
| gib-address-recovery-t_tuned | holdout (#34 oof) | h0→h1 **−0.000038 ± 0.000011**; h1→h0 −0.000063 ± 0.000013; t_g 0.95; US add precision 0.09 / 0.02, India 6 adds (1.00). At t 0.5: −0.0006. Unassigned in-dump gib SX: top-1 address match true **19%** (India and US) | drop |
| ef-decoding-on-a10-p | holdout (#34 oof) | ef_indep_iso +0.000102 ± 0.000058 / −0.000070 ± 0.000049; ef_gate_iso −0.000004 / −0.000080; ef_mix_raw +0.000094 / −0.000032; t retune −0.000060 / −0.000038. Band ECE of #8c p → a10 p': India 0.043 → 0.012, US 0.025 → 0.006 | drop |
| train-caps-5S2-6S3 | holdout (#34 oof) | +0.000001 / +0.000004 (the caps almost never bind) | drop |
| gib-aware-threshold-check | holdout (#34 oof) | gib am pairs, true rate by p' band. India 0.4–0.5: 0.55, 0.5–0.6: 0.58, 0.6–0.71: 0.57. US: 0.43 / 0.50 / 0.60. Calibrated, and below the 0.67–0.75 add bar | drop |
| fn-content-identical-twin | holdout (#34 oof) | FNs with an identical accepted twin at the same S1: India 1 of 7,844, US 0 of 12,330 | drop |
| gib-assigned-share-test | test (#34v_rescue90) | gib SX assigned: France 0.548 (31,035), India 0.895 (88,440), US 0.733 (111,360). Non-gib: 0.560 / 0.575 / 0.585 | inconclusive (France pointer) |

**Error budget after a10** (`budget.py`). A per-S1 Shapley value over 7 error types, so the rows add up to the F0.5
lost; the method is P23-research `budget2.py`'s. Pair counts are true pairs (FN rows) or predicted pairs (FP rows).

| bucket | pairs India / US | F0.5 lost India / US | LB-eq India / US | LB-eq sum | P23 on #8c (LB-eq sum) |
|---|---|---|---|---|---|
| blocking miss, SX with an address | 1,627 / 899 | 0.00179 / 0.00062 | 0.00084 / 0.00024 | 0.00108 | ≈ 0.0014 |
| blocking miss, SX address-less | 1,206 / 2,611 | 0.00111 / 0.00169 | 0.00052 / 0.00065 | 0.00116 | (in the above) |
| pruner miss | 1,415 / 1,866 | 0.00137 / 0.00125 | 0.00064 / 0.00048 | 0.00112 | ≈ 0.0011 oracle |
| **low-p FN** (argmax, p' < t) | 1,740 / 4,257 | 0.00192 / 0.00314 | 0.00090 / 0.00120 | **0.00210** | 0.0023 |
| **stolen FN** (SX argmax elsewhere) | 1,856 / 2,697 | 0.00186 / 0.00198 | 0.00087 / 0.00076 | **0.00163** | 0.0016 |
| FP, non-singleton S1 | 438 / 675 | 0.00104 / 0.00114 | 0.00049 / 0.00044 | 0.00092 | 0.0009 |
| FP, singleton S1 | 27 / 47 | 0.00031 / 0.00033 | 0.00014 / 0.00012 | 0.00027 | 0.0003 |
| **total** | | 0.00938 / 0.01015 | 0.00439 / 0.00389 | 0.00828 | 0.0085 |
| wrongly-empty S1 (a slice of the FN rows) | 170 / 286 S1 | 0.00193 / 0.00216 | 0.00090 / 0.00083 | 0.00173 | — |

**Pattern read.** The category mix comes from `a10.cat_frame` over every pair of the bucket. The raw examples (30 per
bucket, names + addresses) are in `data/work/p36/examples.txt` and `examples_fp.txt` (gitignored). The patterns:

| bucket | category mix India / US | named patterns (from the raw examples) |
|---|---|---|
| low-p FN | address-less SX 70% / 77%; name_replaced 14% / 10% (synthetic brand: 234 / 407 pairs); hn_diff 5.5% / 11.7%; addr_name_variant 4% / 1%; Indic 3% / 0 | **(1) name-only SX of a same-name S1 chain**: the SX has no address and its name (a legal-form or typo variant) fits two or more same-name S1 in different cities, so p' sits at 0.2–0.6. **(2) synthetic brand + kept address.** **(3) house-number noise** on an otherwise equal address (±1, a dropped last digit, a replaced number) |
| stolen FN | address-less 91% / 91%; name_replaced 5.5% / 8% | The same as (1): the "thief" is a same-name S1 in another city. It is a competing train S1 in 88% / 84% of cases, and its p ≥ t in only 3% / 4% (the SX is then assigned nowhere). The true pair's p is mostly 0.01–0.5 |
| blocking miss with an address | name_replaced 48% / 81% (gib: 701 / 664 pairs); addr_name_variant 20% / 7%; web 9% / 5%; Indic 9% / 0; hn_diff 8.5% / 5% | **(4) synthetic-brand rename**: a made-up name glued from generator syllables, with the S1's address copied and perturbed (hn digit drop or ±1, typos, a replaced city). Blocking's name view cannot reach it |
| FP (non-singleton + singleton) | the SX is a distractor 62% / 57%, or belongs to another (train) S1 34% / 39%; p' > 0.99 for 32% / 22% | **(5) same-name sibling decoy**: a legal form added or moved ("LLP X", "X Limited"), words shuffled, a nearby house number (India +2 to +21 units; US a 4-digit number losing its last digit). **(6) a synthetic brand at an address shared by two S1** |

## What failed      (approaches tried and dropped, with the number that killed them)
- **(a) Blocking-miss recovery for synthetic-brand ("gib") SX.** This was the top with-address pattern.
  - `gib.learn` finds the 44 syllables with no labels. It gives the same 44 on train and on test (France included).
    With them, gib SX are 3.2% / 3.8% of India / US holdout true pairs. In train, 99.2% (India) and 88.4% (US) of gib
    SX are true matches, but 89–92% of those pairs are already TP.
  - The residual cannot be found from the address. Take the gib SX that #34 left unassigned and whose status is known
    (in the P23 dump). Their top address match is the true S1 only **19%** of the time.
    - US: 45% of that pool are distractors placed at a real S1 address (decoys). They match the true ones on
      hn-equality (0.56 vs 0.58), cosine (0.73 vs 0.76) and uniqueness.
    - India: in 75% of the pool the true S1 is not the top address match.
    - The cleanest slice (cos > 0.95 and a unique match) is 6% true in the US and 20 queries in India.
  - The scorer was a LightGBM on the address / hn features, fit on non-holdout train S1, with t tuned on one half.
    It loses in both directions (Evidence).
- **(b) Expected-F0.5 subset decoding on the a10 p'.** The brief's retry condition holds: p' is about 4× better
  calibrated than #8c's p. P23's decision bench was rerun on p'. It covers EF-indep, EF + empty gate, and EF with a
  learned singleton prior, each raw and isotonic, with miss rates {0, 0.02, λ, 0.15}.
  - The best arm is ef_indep_iso: +0.000102 at 1.8 SE, and it flips sign on the swap (−0.000070).
  - Every other arm is within ±0.0001, and so is a global t retune.
  - This matches P23's result on #8c's p (V4 −0.00025 / +0.00000), so a better-calibrated p does not unlock it.
- **The train caps (≤ 5 S2, ≤ 6 S3 per S1).** They bind on a handful of S1: +0.000001 / +0.000004.
- **(c) A gib-aware threshold.** Inside the candidates, p' is calibrated for gib SX in every band, and those bands sit
  below the add bar.
- **(c) Copying an accepted content-identical twin.** 1 of 7,844 India FNs and 0 of 12,330 US FNs have one.
- **Not retried (dead in earlier probes, and the budget does not reopen them):**
  - the address-less / twin bucket (P35);
  - pruner τ (P10: flat; P23: ≈ 0.0002 real);
  - a margin rule (V3d, now inside a10).

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
Run everything from the repo root with `PYTHONPATH=probes/P36-iu-residual probes/P36-iu-residual/py.sh <script>`
(2.2 GB cap). Outputs go to `data/work/p36/` (1.9 MB).
- `base.py`: `state(c)` builds the #34 holdout decision out of fold. It returns pairs (+ `pa` = p', `acc`), per-S1
  n_true, halves, f8 and f34. `paired(d, h, rep)` gives the mean and SE on one half. Self-check: reproduces P34's
  half-1 F0.5. 1 s.
- `budget.py`: the Shapley budget → `err_{c}.parquet` (one row per error pair, with its type) and `budget.csv`. 2 s.
- `examples.py`: the category mix and 30 raw examples per bucket → `examples.txt`. 6 s.
- `gib.py`: `learn(split)` (cached to `morph_{split}.json`), `is_gib(names, M)`, `n_syll(token, M)`. Self-check
  included. ~80 s per split.
- `recover.py cands <split> <c>` → `gc_{split}_{c}.parquet`. It covers the unassigned gib SX with their top-10 S1 by
  address and FEATS. The holdout run skips SX that #34 accepted (holdout S1: p'; train S1: #8c's p). The test run needs
  `acc_test_{c}.parquet`. 20–80 s, ≤ 1.5 GB. The gc files were deleted for disk space; rerun to regenerate them.
- `ho_gib.py`: the (a) gate (fit on non-holdout train S1 → `grecover.txt`; tune t on a half, report the other half)
  → `ho_gib.json`. 50 s.
- `ef.py`: the (b) gate. P23's `s3_bench.ef_mix` and P5's `decide.ef_select` on p', plus the caps → `ef.json`. 25 s.
- `te_gib.py [sub]`: the label-free test profile of gib SX assignment per country → `te_gib.csv`. 2 min.
- `diag/*.py`: the one-off checks behind the numbers above. calib = ECE, gibho = gib status on the holdout,
  diag / diag2 = gib pool anatomy (needs the gc files), gibcand = gib p' bands, fp = FP anatomy, dup = twins.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Nothing ships, so there is no licence or rule exposure.** `gib.py` uses only the stdlib; the syllables come from
  the split's own names (test-side unsupervised use is allowed), with no hand list. `recover.py` reuses P3's
  vectorizer, sparse_dot_topn (Apache-2.0) and rapidfuzz (MIT).
- **Holdout-world blind spot.** A gib SX that no holdout S1 competes for (90% of them) has an unknown #34 status, and
  (a) treats it as unassigned. That is pessimistic on the adds to holdout S1. The drop verdict does not depend on it:
  the pool whose status is known is 19% precise at top-1.
- **US half 0 is P17-in-sample for the base p** (the P34 note), so the h1→h0 US figures are indicative. Neither lever
  passes even h0→h1.
- **Assumption:** "India+US ≥ +0.00015" was read as LB-eq (0.468 / 0.383), as P34 read it.
- **Disk:** during the run the disk fell below 1 GB and memguard paused every probe. I freed 70 MB of my own
  intermediates. Other probes' data was not touched.

## Open questions   (for the master; each with your best guess)
- **Is any India/US lever of ≥ +0.00015 left?** Best guess: no, not in the decision layer or in cheap recovery. The
  residual is dominated by cases the data leaves ambiguous: name-only SX of same-name chains, decoy brands at shared
  addresses, and same-name sibling decoys. p' is calibrated on each of them. Spend the remaining hours on France.
- **France synthetic-brand SX (pointer for a France probe).** Test France assigns 54.8% of its 31,035 gib SX.
  - Comparison: the gib/non-gib assignment ratio is 1.56 in India and 1.25 in the US, and 0.98 in France. At the US
    ratio France would assign ≈ 70%, so ≈ 15 points (≈ 4.7k SX) are missing. The likely cause is France's strict t
    (0.9868), which a name-mismatch pair rarely reaches.
  - But an address key will not recover them. France's unassigned gib SX have a top-1 address gap of 0.023, with ≈ 8
    S1 within 0.05 cosine. Only 1.3% have a clean unique match.
  - The US decoy rate (45% of the unassigned gib pool) warns against a blind tier.
  - Best guess: worth one France probe that scores gib pairs with the hn-aware matcher (hn-equal + street-equal, with
    the name features masked) and gates them on the US holdout's gib precision. Expect ≤ +0.0003 LB.
- **US test assigns fewer gib SX (73%) than the holdout implies (≈ 82%).** Best guess: test's extra distractors (5.75
  SX per S1 vs 4.67 in train) explain it. Non-gib assignment falls even further (58.5% on test vs ≈ 74% true in train:
  ratio 0.79, against 0.89 for gib), so no action is needed.
