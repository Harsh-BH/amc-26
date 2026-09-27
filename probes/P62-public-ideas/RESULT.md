# P62-public-ideas — RESULT (DRAFT 06:55, being updated)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- Idea 3 (clone09 density check): clone world built (every free holdout SX gets Poisson(0.9) exact clones; argmax, a10,
  #34's p', CTX2/CTX3 re-derived with P57's world()); evaluation running.
- Ideas 1 (acronym rule) and 2 (e5 dense-retrieval rescue): not started.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- Pending.

## Evidence         (table: ledger name | world | key metrics | verdict)
- Pending.

## What failed      (approaches tried and dropped, with the number that killed them)
- Pending.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `clone09.py build <c> [lam] [seed] [pois|bern]` -> `data/work/p62/A_clone<mode><lam>s<seed>_<c>.parquet`
  (India 64 s, US 92 s, peak 1.8 GB). `clone09.py eval [lam] [seed] [mode]` -> `data/work/p62/ev_<world>.json`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- clone09 recomputes the stacker-level context only; m2's own input context inside #8c's p is kept from the full world.

## Open questions   (for the master; each with your best guess)
- Pending.
