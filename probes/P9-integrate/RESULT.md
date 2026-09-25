# P9-integrate — RESULT   (DRAFT, in progress)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- In progress: promoting P4 features / M2 training, P4b rerank and P6's France rule into `src/`, wired as `run.py --stage model`.
- Found: P6's rule reproduces sub4's France t = 0.82 only with the reference taken on **M2's** US+India test p at M2's t* 0.71
  (the model France is scored with). With the CE-blended US/India p as reference it gives 0.83.
- Recommendation: pending.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
pending

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| (check, no row) | test | P6 rule on cached probs: ref = M2 p @ 0.71 → t_France 0.82; M2 @ 0.70 → 0.82; blend @ 0.70 → 0.83 | reference = M2 p |

## What failed      (approaches tried and dropped, with the number that killed them)
pending

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
pending

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- France t sits on a flat, close boundary: France empty rate 5.682% at 0.82 vs reference 5.691% (5.737% at 0.83).

## Open questions   (for the master; each with your best guess)
pending
