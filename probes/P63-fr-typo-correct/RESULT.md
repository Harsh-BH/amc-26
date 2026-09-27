# P63-fr-typo-correct — RESULT (final, 27 Sep 14:05)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **The gate fails.** Correcting SX typos to the S1 vocabulary lowers France-analogue F0.5 in both LOCO directions, on every recipe and both analogues:
  - **US→India −0.00083 ± 0.00011; India→US −0.00204 ± 0.00011** (chain analogue, gc70 recipe = the sub_iu54_frgc70 France recipe).
  - The pair-wise variant, which only corrects toward the pair's own S1 tokens, is worse: −0.00110 / −0.00271.
- The corrector itself works: 13.7% of France SX records get a sensible fix (e.g. "avneue"→"avenue", "ranaud"→"renaud"). But the M2 already absorbs typos, and the one-letter difference it erases is informative: pairs the correction adds are only ~40% true.
- **Recommendation: do not adopt. `output/sub63` was not built** (the brief builds it only if the gate passes). India/US/France outputs are untouched.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
Adopt nothing. Keep `output/sub_iu54_frgc70` / `output/sub_iu54_best` as they are.
- There is no SUBMISSIONS row, because there is no candidate.
- `probes/P63-fr-typo-correct/typo.py` is kept only as a reusable, self-checked corrector, in case someone wants to try it as *extra* features (see Open questions). Do not use it to rewrite text.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| p63-rates | train India / US, test France | Corrected token share, name / addr: France 0.63% / 1.71%; India 1.35% / 0.27%; US 1.68% / 1.88%. SX records with ≥ 1 fix: **France 13.7%** (addr 11.9%, name 2.0%); India 7.3%; US 15.3% (addr 10.4%). US is the France-like direction | keep (corrector) |
| p63-loco-global | LOCO fold 0 (P42 frames) | dF vs the uncorrected analogue, fixed thresholds (table below). Every cell < 0, both halves | **drop** |
| p63-loco-pairwise | LOCO fold 0 | pair-wise variant: every cell < 0, larger than global | **drop** |

**LOCO dF (×1e-4, fixed thresholds, SE ≈ 1.0–1.5):**

| corrector | analogue | gc70 US→India | gc70 India→US | sub42_both US→India | sub42_both India→US | base only US→India | base only India→US |
|---|---|---|---|---|---|---|---|
| global map | chain | **−8.3** | **−20.4** | −7.3 | −19.8 | −8.8 | −17.8 |
| global map | m2 | −4.2 | −22.0 | −3.4 | −22.0 | −2.8 | −20.0 |
| pair-wise | chain | −11.0 | −27.1 | −9.9 | −26.4 | −10.8 | −23.1 |
| pair-wise | m2 | −5.9 | −28.8 | −5.0 | −29.3 | −4.2 | −25.7 |

- Re-mapping the thresholds by density on the corrected p changes nothing (global chain gc70: −8.7 / −20.1).
- Both report halves agree (h0 / h1, global chain gc70: −10.8 / −5.9 and −19.8 / −21.0).

**Protocol** (`loco.py`):
- **Model:** P6's source-only M2 (`models/<src>.txt`, from the S3 offload) scores every train pair of the target whose SX has a fold-0 candidate.
- **g0:** the pair's P4 X as is. It reproduces P6's `S_holdout` p exactly (max |diff| 0 on 1,240,443 / 1,885,813 fold-0 rows).
- **g1:** the same X with `features.strings` recomputed on corrected SX name / core / addr (sq and acro follow core).
  - A recompute on *uncorrected* text matches P4's X exactly (0 mismatching columns).
  - Context, block, hn / city / state and rule features stay as they are, so blocking is unchanged.
- **Frame:** re-argmax over every train S1, then fold-0 argmax rows in P42's npz format.
  - r = P30's source-only bge, unchanged; hn_rel = P39's.
  - My uncorrected frame (c0) equals P42's frame (rows, s1, y, hn_rel, g sum, r-NaN count), and evaluating c0 gives dF = 0.000000.
- **Decisions:** P42's `decide` (τ 0.1, cap 5/6) with its France→LOCO density map.
- The vocabulary is the target's own S1, as the brief specifies.

**Why it hurts** (chain gc70 decomposition, pairs added / removed vs the uncorrected analogue):

| corrector | direction | added (true) | removed (true) | added-false whose SX had its true pair accepted before |
|---|---|---|---|---|
| global | US→India | 764 (305 = 40%) | 248 (166) | 0 / 459 |
| global | India→US | 2,164 (989 = 46%) | 695 (615) | 0 / 1,175 |
| pair-wise | US→India | 894 (337 = 38%) | 205 (125) | 0 / 557 |
| pair-wise | India→US | 2,773 (1,093 = 39%) | 335 (268) | 0 / 1,680 |

- 99–100% of the added pairs have a corrected SX. Their precision is 38–46%, against ~98% for the accepted set, so F0.5 loses.
- A one-letter difference between an SX and a candidate S1 is either a typo (true pair) or a look-alike S1 (false pair).
  - The M2's fuzzy features (ratio, JW, partial, char-3gram TF-IDF) and its training already weigh those two cases.
  - Correction turns every such difference into an exact token match. SX that were correctly rejected then cross the threshold with a look-alike S1.
  - No added false pair takes an SX from its true pair.
- On score shifts, the global map moves 4.8% (US→India) / 3.9% (India→US) of the corrected *true* pairs down by more than 0.05. Some fixes point at the wrong word, e.g. "kornex"→"korney" and "paltors"→"pastors" among the US examples. The frequency tie-break favours common words over a rare brand token.

## What failed      (approaches tried and dropped, with the number that killed them)
- **Global corrector** (unique nearest S1-vocabulary token, DL ≤ 1 / ≤ 2 for len ≥ 8, frequency tie-break at ≥ 3×): −0.00083 / −0.00204.
- **Pair-wise corrector** (a globally-OOV SX token is rewritten only to the unique nearest token of the pair's own S1): −0.00110 / −0.00271. It removes fewer true pairs, but it adds more look-alike pairs.
- **Not tried** (no time before 15:00, and the gate result makes them unlikely):
  - retraining the source M2 on corrected text;
  - measuring the blocking-recall change.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- **`typo.py`** (rapidfuzz only; self-check: `.venv/bin/python probes/P63-fr-typo-correct/typo.py`):
  - `counts(texts) -> Series token -> freq`;
  - `build_map(vocab, query, chunk=20000) -> Series oov -> fix`. It uses a SymSpell deletes index and exact DL verification, with the query side chunked for RAM;
  - `apply(texts, map) -> (texts, changed)`.
- **`rates.py SPLIT C`** → `data/work/p63/map_<split>_<c>_{name,addr}.parquet` and `rates_<split>_<c>.json` (30 examples per family inside). Runtime: France 28 s / 1.0 GB, India 114 s / 1.8 GB, US 206 s / 1.8 GB.
- **`loco.py`:**
  - `score SRC TGT` (65 s / 290 s, 1.8 GB);
  - `score_pair SRC TGT` (200 s / 342 s, 1.9 GB);
  - `frame SRC TGT [tag]`;
  - `eval [tag]` → `loco.json` (global) and `loco_pair.json` (pair-wise). It also holds `pair_fix(sx, s1, oov)` and P39's `hn_rel`.
- **Work files:**
  - Kept: `data/work/p63/` (57 MB), holding the maps, rates and the c0 / c1 npz frames.
  - Deleted to save disk (≈ 1.3 GB free): the 370 MB `g_*.parquet` score tables. Rebuild them with `loco.py score` / `score_pair` (≈ 5 min each). P6's models were copied to `data/work/p63/models/`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Worktree:** I worked in the main checkout's `probes/P63-fr-typo-correct/`, where the brief was placed, because disk was 2.2 GB free and then 1.3 GB (P42 did the same). Nothing under `src/` was touched, and nothing is committed.
- **Scope of the correction:** only SX name / core / addr text was corrected, not city / places / state (rules features). The LOCO keeps the CE score r uncorrected. The brief's premise is that the CE already handles typos, and correcting r too could only add the same look-alike effect.
- **Model:** the LOCO uses P6's 65-feature M2, not France's c_hn (+5 hn features). The failure is large (7–20 SE) and holds in the France-like direction (US: 15% of SX corrected, like France's 14%), so a c_hn analogue is very unlikely to flip it.
- **Timing:** the probe started at 12:33, after the 12:30 LOCO deadline. The verdict was ready at ≈ 13:30.
- **Licence / data:** rapidfuzz (MIT) only. The vocabulary is the country's own S1 file; no dictionary and no external data.

## Open questions   (for the master; each with your best guess)
- **Would retraining M2 on corrected text rescue the idea?** Best guess: no. Correction erases the typo-vs-look-alike evidence, and retraining can at best learn to trust exact matches less. The only version with upside is to **add** corrected-text similarities as extra columns, next to the originals, plus a count of corrected tokens, then retrain c_hn. That needs a c_hn retrain, a France chain recalibration and a new LOCO: several hours, so not for today.
- **Does correction raise blocking recall?** Not measured. Best guess: negligibly. P3's name TF-IDF is char-3gram, which already tolerates one-edit typos. The P43 candidate set already carries 4.4 candidates per S1.
- **Is France's "under-confident matcher" (true pairs at p 0.3–0.9) a typo effect?** The LOCO says typos are not the lever: fixing them adds false pairs faster than true ones. Best guess: the low-p true pairs are look-alike-heavy groups, where the M2 is right to hesitate.
