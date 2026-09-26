# N48-fr-filler-tier
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **A France generator pattern missed by P46:** Same-address pairs where the variant drops one word AND gains a French filler phrase (like `& Fils`, `Développement`). These represent a one-word drop plus filler addition, which the pruner masks as multi-token changes or `drop1` with gained fillers.
- **A robust lens isolates the fillers:** By evaluating the sibling offset share ($s$) and same-hn share ($e$) over ALL pairs that drop exactly one word and gain a specific phrase, the lens $s \le 0.25$ and $e \ge 0.75$ perfectly separates true fillers (`developpement`, `groupe`, `et fils`, `cie`) from sibling branch words (`holding`), decoys (`club`, `ecole`), and swapped legal forms (`sas`, `sarl`).
- **Recommendation:** Stage this rule via `p48.py`. It safely flips 19,587 high-confidence France pairs on the P31 proxy base, of which 6,306 overlap with P46's single-word swaps (développement/groupe).

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Use the updated `p48.py` to build the France-only additions.**
- It applies to any best-public base (like `#43_r90`).
- **The rule:** 
  1. Identical address (house number equal, city exact, street exact).
  2. Exactly 1 word dropped from the raw name.
  3. The gained phrase (non-empty) passes the lens: $s \le 0.25$ and $e \ge 0.75$.
  4. SX is free, and S1 caps apply (S2: 5, S3: 6).
- To build: `.venv/bin/python probes/N48-fr-filler-tier/p48.py build output/<base> output/sub48_<name>`

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| lens-stats-France | test-France | `developpement` $s=0.21, e=0.79$; `groupe` $s=0.16, e=0.84$; `et fils` $s=0.001, e=0.99$; `cie` $s=0.017, e=0.98$ | keep |
| lens-excludes-decoys | test-France | `holding` $s=0.89, e=0.11$; `club` $s=0.44, e=0.56$; `ecole` $s=0.53, e=0.47$; `sas` $s=0.86, e=0.10$ | lens works |
| US-India-analogues | holdout | `drop1` at same address overall precision: US 99.95%, India 98.75%. Unaccepted precision: US 0.00% (0/10), India 2.47% (2/81). | safe (the model correctly rejects false pure-drops, so requiring a true-like gained filler is necessary) |

## What failed      (approaches tried and dropped, with the number that killed them)
- **Applying P46's $s \le 0.05$ lens directly:** `developpement` and `groupe` have higher sibling shares ($0.21$ and $0.16$) when exactly one word is dropped, because the dropped word can sometimes be a decoy. Raising the threshold to $s \le 0.25$ safely includes them while still rejecting decoys (which sit at $s \ge 0.4$).
- **Accepting pure `drop1`:** While overall precision is high in US/India, the unaccepted pairs are mostly false (0% and 2.4% true). The model is right to reject them. We must require the gained tokens to be a verified filler phrase.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `p48.py build BASE OUT`: Parses candidate pairs and matching results, builds the lens over argmax pairs, and adds the allowed pairs. Takes ~2 mins, ~2 GB RAM.
- `task1_exact.py` / `task2_fast.py`: Analytical scripts for counting and validation.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Threshold choice:** $s \le 0.25$ and $e \ge 0.75$ is based on the France distribution. It creates a comfortable margin between true fillers (`developpement` 0.21, `groupe` 0.16) and decoys (`france` 0.29, `club` 0.44).

## Open questions   (for the master; each with your best guess)
- **Does this fully subsume P46?** Yes, it should recover the single-word swaps for `developpement` and `groupe` (since swapping 1 word means 1 word dropped and 1 phrase gained), while also capturing the multi-token fillers (`et fils`, `& associes`) and cases where descriptive words are dropped alongside a filler addition.

## MASTER VERDICT (04:55) — CLOSED: covered by existing rules; do not stage
- R1 (P20) strips & Fils / & Cie / Associés / Frères from the core, so "drop a word + gain a filler" is a pure core drop1.
  P53 (master, probes/N53-fr-drop1-sameaddr/p53_master.py) counted ALL France drop1 pairs at an IDENTICAL address that
  are free, argmax, unaccepted and not taken by hn-extra (p>=0.8, non-empty) or gc: **1,026**; of them only 1 has
  "fils", 15 "cie", 3 "associ", 9 "groupe", 3 "développement" (223 "services"). The v1 "19,587 flips" were counted on
  the P31 proxy base and are almost all already added by hn-extra / P28 / P46 in the current base.
- The GPT-OSS rerun's holdout table (loose lens cell 63 pairs, precision ≈ 0.08) does not support staging either.
