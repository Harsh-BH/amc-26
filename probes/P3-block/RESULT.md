# P3-block — RESULT
## TL;DR
DRAFT (dev world only; the full-corpus holdout run is in progress). Blocking in the reverse direction wins: for each S2/S3, take its top-k S1 by TF-IDF cosine (name char-3grams + address words).
On dev, SX→S1 at k=1 gives pair recall 0.987 at 5.8 candidates/S1, and forward S1→SX at k=20 gives only 0.931 at 20/S1. A union of both directions over 3 views gives 0.9976 at 20.7/S1.
Recommendation pending the full-corpus numbers: dev has only 30k S1, so the reverse direction is flattered.

## Recommendation
Pending. Current best on dev: `probes/P3-block/block.py::candidates(s1_df, sx_df, CFG, vecs)` with views na/name/addr, forward k=10/5/5 and reverse k=2/1/1.

## Evidence
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| P3-block/dev-f:na20 | dev | pair_recall 0.931, 20/S1, India 0.861, cross_script 0.372 | inconclusive |
| P3-block/dev-r:na1 | dev | pair_recall 0.987, 5.8/S1, cross_script 0.964 | keep |
| P3-block/dev-f:na10+r:na2 | dev | pair_recall 0.9948, 16.4/S1 | inconclusive |
| P3-block/dev-f:na10,name5,addr5+r:na2,name1,addr1 | dev | pair_recall 0.9976, 20.7/S1, full-recall 0.992 | inconclusive |

## What failed
- Forward-only top-k (S1→SX) tops out at about 0.957 at k=100 on dev. SX with an empty address or an Indic name get low cosine against every S1, so they fall below the S1's top-k. Reverse ranking is per SX, so that handicap cancels out.

## Interfaces
`probes/P3-block/block.py`: `fit_views(train_sx, cfg)`, `candidates(s1_df, sx_df, cfg, vecs)`. Full-scale RAM and time: pending.

## Risks
- Normalisation is a stopgap: anyascii (ISC) for non-Latin, then lowercase and strip punctuation. P1's normalize.py and P2's translit.py are not in yet.
- France: IDF is fitted on train (US/India), so French-only tokens (rue, sarl) get train IDF. That can't be validated.

## Open questions
- May IDF be fitted on test S2/S3 text (PROBLEM.md open question 4)? Best guess: stay with train-fitted IDF.
