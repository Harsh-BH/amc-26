# P66-fr-noaddr — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **The premise is wrong at the pair level.** Address-less SX are 97.7% owned by *some* S1, but in LOCO only **63–64%** of their
  argmax pairs are true, and only 27–42% of the ones the strict France chain leaves unaccepted. Adding all of them: India→US **−0.0024** (FR 0.5).
- The one class that holds is **`xfc`**: exact unique full name **plus a unique S1 name core**. It is 0.996 / 0.987 precise over
  all train argmax pairs and passes the precision gate in LOCO (0.978 / 0.889). France has 1,900 such pairs → **LB +0.00007 to +0.00010**.
- **Upload `output/sub66_final`** = sub64 + 1,884 France pairs (validator `--check-ids` PASS). This is small but safe. Do not push the rest of the 29k.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Upload `output/sub66_final`** (md5 c6246285…). It is `output/sub64` + 1,884 France adds on 1,810 S1.
  - 0 adds hit a P64 conflict (exact names carry the same legal form; checked with `lf.pair_class`).
  - P54 dup step: 0 adds.
  - India/US lines byte-identical to sub64; candidates hard-linked from sub64.
  - sub64 has no `code.zip`, so this folder has none.
- Same adds on the other bases: `output/sub66` (sub_iu54_frgc70 + 1,900) and `output/sub66_best` (sub_iu54_best + 1,884).
- **run.py (P43)**, France loop, after gc (same place as P40's rule):
```python
import rule66 as R66, rule40 as R40                    # probes/P66-fr-noaddr/rule66.py, probes/P40-generator/rule40.py
pool = am & ~acc & (p >= 0.3)
cls = np.zeros(len(acc), bool)
cls[pool] = R66.xfc(n1[pool], nx[pool], ax[pool], core1[pool], fr_full_count, fr_core_count)
acc |= R40.add(s1c, sxc, is2, p, am, acc, sib, cls, 0.3)          # caps 5 S2 / 6 S3, free SX, non-sibling
```
  - `fr_full_count` = `{rule40.fkey(name): #S1}` over the test S1 of the unseen country.
  - `fr_core_count` = `{rf core: #S1}` over the same S1.
  - Address-less = the raw address is empty after strip, or the literal `NULL` / `<NULL>`.

## Evidence         (table: ledger name | world | key metrics | verdict)
LOCO = P27 M2 trained on src, scoring tgt fold 0, on the P37 France-chain analogue: strict t_base + rescue90 + gc70, with
thresholds density-mapped to the fr31 France pairs. Cell format: dF France-F0.5 (added, prec). SE(h1) is given where it matters.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| p66-fr-profile | France test, sub_iu54_frgc70 | 43.1k address-less SX (3.0%), 31.2% matched. Of the 29,642 unmatched, 92% have a candidate; median argmax p 0.29. Of those with a candidate: exact name = argmax S1 51% (14,015), of which unique name 6,353 and twin names 7,662; core_eq & core-unique 4,769 | keep (descriptive) |
| p66-train-law | LOCO | address-less argmax pairs true: **0.637 (US→India) / 0.632 (India→US)**. True owner is the argmax: 0.71 / 0.73. Base recall 0.42 / 0.59. Unaccepted pool precision **0.42 / 0.27** | drop the premise |
| p66-loco-any | LOCO | FR 0.1: US→India +0.00133 (2,239, 0.86); India→US **−0.00677** (8,309, 0.32). FR 0.5: +0.00093 / **−0.00239** | drop |
| p66-loco-xf_twin | LOCO | FR 0: −0.00199 (1,119, 0.22) / −0.00266 (2,583, 0.23) | drop |
| p66-loco-xf (P40 xf_noaddr, no p floor) | LOCO | FR 0: +0.000151 (395, 0.86) / −0.000018 (585, 0.68). FR 0.7: +0.000114 / +0.000034 (0.73) | drop (India→US ≈ 0, prec < 0.85) |
| p66-loco-near90 / near80 / tsr100 | LOCO | near90 FR 0: +0.00055 (0.90) / −0.00009 (0.59). near80 −0.00052 India→US. tsr100 −0.00061 India→US | drop |
| p66-loco-near95 | LOCO | FR 0.5: +0.000051 (54, 1.0) / +0.000034 (63, 0.83) | inconclusive (France: only 292 pairs) |
| **p66-loco-xfc** | LOCO | cell precision over all argmax pairs **0.996 / 0.987** (n 1,805 / 4,096). Pool 47 / 9 pairs. FR 0.3: **+0.000038 ± 0.000011 (42, 0.976) / +0.000004 ± 0.000003 (9, 0.889)**. Per-pair gain 0.06–0.09 France-F | **keep** |
| p66-loco-xfc\|near95c | LOCO | FR 0.3: +0.000062 (71, 0.986) / +0.000024 (31, 0.839) | inconclusive (India→US prec < 0.85) |
| p66-stage-sub66_final | France test | +1,884 pairs / 1,810 S1 on sub64. P64 conflicts 0, dup 0. Validator `--check-ids` PASS. check.py OK | keep |

**Why the xfc gain is small in LOCO but real in France.** The source-trained LOCO models already accept 98%+ of the class, so
only 47 / 9 pairs are left to add. France's chain leaves **1,900** (0.0073 per S1, vs 0.0005 / 0.00007 in LOCO). 1,648 of
them sit at p 0.9–0.987, just under France's 0.9868 bar. Expected gain: 1,884 × 0.06–0.09 / 259,452 = +0.00044 to +0.00065
France-F0.5, ×0.1498 = **LB +0.00007 to +0.00010**.

## What failed      (approaches tried and dropped, with the number that killed them)
- **"Assign the address-less SX to its argmax"**, at any floor: India→US −0.0068 (FR 0.1) to −0.0004 (FR 0.9). Name-only
  evidence does not identify the owner. The 97.7% is ownership, not argmax correctness.
- **Exact names shared by 2+ S1 (twins, 7.7k of France's unmatched):** precision 0.22 in both directions. Coin flips, as
  P40/P49 found.
- **xf without the core guard (P40's xf_noaddr, all floors):** India→US −0.00002 to +0.00003 at precision 0.68–0.76.
  - diag.py: 97% of its false adds are owned by another S1, and in 86–89% of those the owner has the same core as the argmax S1.
  - In ~half the owner's key is the SX key plus extra words, typically a legal form the copy dropped (e.g. SX "X" belongs to
    "X LLC", not to the separate S1 "X").
  - Core count 1 lifts pool precision to 0.96 / 0.89; core count ≤ 2 (`xfc2`) is still 0.75 India→US.
- **Near-name rules** (ratio ≥ 80 / 90, token-set = 100) with a unique S1 key: precision 0.43–0.72 India→US, all negative there.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `rule66.py`: `xfc(n1, nx, ax, c1, full_count, core_count) -> bool[]` for run.py, plus a self-check.
  - Checked against fr66's class on France: 10,403 / 40,822 argmax address-less pairs, identical masks.
- `fr66.py count | build CLASS T OUT [BASE2]`: France adds per class / floor → `fr66_count.json`, and the staged folders.
  - Pairs are computed on `P37_BASE` (default sub_iu54_frgc70), then re-filtered on BASE2: owned SX dropped, caps re-checked,
    P64 class-4 conflicts skipped.
  - 1.5 min, ~1.2 GB.
- `loco66.py`: every class × FR floor, both LOCO directions → `loco66.json`. 40 s, 1.6 GB.
- `diag.py US|India`: why xf's false adds fail (aggregates only). 30 s.
- `prof.py [BASE]`: France address-less profile → `data/work/p66/fr_noaddr.parquet`, `prof.json`. 17 s.
- `check.py OUT BASE`: stand-in for `--check-ids` on the changed lines. Checks: only France lines change, additions only,
  ids exist in test S2/S3, adds ⊂ the S1's candidates, no SX owned twice.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- No new dependency (rapidfuzz, pyarrow). No external data. Name/core counts come from the test S1 (label-free, ruling 2).
- The LOCO evidence for xfc is thin: 9 pairs India→US (+0.000004 ± 0.000003). The class-wide train precision (0.987–0.996
  over 5.9k pairs) carries most of the case.
- **France-specific risk:** France's p for these pairs is lower than in LOCO, so they may be harder cases. Mitigation: 87%
  of them are at p ≥ 0.9.
  - If France's precision on the class is only ~0.75 (the xfc2 level), the gain falls to about 0.
  - It is negative only below ~0.7 precision, which nothing in train suggests for a core-unique exact name.
- The core comes from P29's rf `core` (legal forms and fillers stripped). If run.py computes the core differently, the class
  shifts slightly.
- Assumption: the NULL / `<NULL>` placeholders count as address-less. P40 used empty only. The difference does not matter here:
  all 1,884 staged adds have a truly empty address (checked).

## Open questions   (for the master; each with your best guess)
- **Is a bigger France address-less push possible?** Best guess: no, not by rules. The remaining unmatched are twin names
  (7.7k, 0.22 precision) or core twins (4.4k, 0.68–0.88). Only an owner-disambiguating signal would help, and address-less
  records have none (P40 found no tie-breaker).
- **Add `near95c` (+116 France pairs)?** Best guess: not worth it. India→US precision 0.75–0.85, and it is ~+0.00003 France-F.
- **Lower the floor to 0 (+24 pairs at p < 0.1)?** No. The model strongly disagrees there, and the gain is negligible.
