# P66-fr-noaddr — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- DRAFT (13:20). France profile done; LOCO running.
- France under sub_iu54_frgc70: 43.1k address-less SX (3.0%), 31.2% matched; 29,642 unmatched, 92% of them have a candidate.
  Of those, 51% carry exactly the argmax S1's full name, but only 6,353 (23%) of a UNIQUE France name; 7.7k are exact names of twin S1s.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- pending LOCO.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| p66-fr-profile | France test, sub_iu54_frgc70 | address-less 3.0%, matched 31.2%; unmatched 29,642: fk_eq 14,015, fk_eq&unique 6,353, core_eq&unique 4,769; median p 0.29 | keep (descriptive) |

## What failed      (approaches tried and dropped, with the number that killed them)
- pending

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `prof.py [BASE]`: France address-less profile → `data/work/p66/fr_noaddr.parquet`, `prof.json`. 17 s, < 1 GB.
- `loco66.py`: LOCO both directions on the P37 France-chain analogue → `loco66.json`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Address-less = address empty after strip, or the literal NULL / <NULL> placeholder.

## Open questions   (for the master; each with your best guess)
- pending
