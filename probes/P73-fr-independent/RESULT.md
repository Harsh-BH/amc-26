# P73-fr-independent — RESULT (DRAFT 15:20, GPU jobs running)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- DRAFT. The France-independent CE recipe is built. It fine-tunes a fresh public checkpoint on P67's v2 synthetic pairs of the target country plus agreement pseudo-labels (P61 rule) on the target's band. No labelled India/US pair enters training.
- Three jobs are running. `p73-bge-loco-0927-1516` runs base bge-reranker-v2-m3 on India + US (LOCO). `p73-sm-0927-1516` runs ms-marco-electra-base (the English stand-in) on India + US, then camembert-mmarcoFR on France. `p73-bge-fr-0927-1516` runs base bge on France.
- Gate: the independent CE as a new P55 combiner column, at the shipped cell. US-as-target (India→US) must reach ≥ +0.0003 and India (US→India) ≥ 0.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
Pending.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|

## What failed      (approaches tried and dropped, with the number that killed them)
—

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `prep73.py` → `data/work/p73/ch/train_ind{India,US,France}.parquet`, also on S3 at `amc26-p73/ch/` (1.5 s, < 0.5 GB). Row counts:

  | train file | pairs | synthetic | pseudo-labels |
  |---|---|---|---|
  | India | 134k | 120k | 14.4k |
  | US | 149k | 120k | 29.2k |
  | France | 168k | 120k | 48.4k (France is subsampled to 20% of its band, to match LOCO coverage) |

- `launch73.py NAME INSTANCE k:m,...` with m in bge | cam | ele. `solution_p73.py` wraps P13's Job, unchanged.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Models (licences checked on the HF API cards, 27 Sep):
  - BAAI/bge-reranker-v2-m3: Apache-2.0, 568M.
  - antoinelouis/crossencoder-camembert-base-mmarcoFR: MIT, 111M.
  - cross-encoder/ms-marco-electra-base: Apache-2.0, 109M.
- Pseudo-labels come from India/US-trained models (the P55 combiner and Qwen). Only the pairs where both agree confidently are kept, so some India/US decision rules still enter through them.

## Open questions   (for the master; each with your best guess)
—
