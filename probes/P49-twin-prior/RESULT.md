# P49-twin-prior — RESULT (DRAFT, 27 Sep ~04:55 IST; tests 1, 3, 4, 5 done; decode on sub44r running)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- Draft. Test table below. The group-size prior is real but weak. The raw-name channel is strong on a small slice. Script → state
  is deterministic but covers almost no rows. Identical address-less copies always share an owner (no tie-breaker).
- The decode on the sub44r out-of-fold decision is in progress.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- Pending the decode gate.

## Evidence         (table: ledger name | world | key metrics | verdict)
Frame: #8c dense holdout (P23 dump). Ambiguous = an address-less SX whose candidate list holds ≥ 2 S1 sharing its #8c argmax
S1's full-name key (P40 floor.key). The table counts only SX with the owner inside the set and ≥ 1 holdout twin:
India 8,045, US 12,707.

| test | India | US | reading |
|---|---|---|---|
| 1 group size, r(k)=f(k+1)(k+1)/f(k), nset=2: the smaller-k twin is the owner | 0.583 ± 0.014 (model 0.576) | 0.564 ± 0.010 (0.577) | calibrated, weak (+0.012 / +0.021 nats per SX) |
| 1 brief's h(k)=f(k+1)/f(k) | model says 0.54, observed 0.365 | 0.57 vs 0.393 | overconfident, worse than uniform |
| #8c argmax is the owner | 0.394 (1/nset 0.246) | 0.360 (0.288) | p already carries some signal |
| 3 twins' raw names differ | 4.8% of SX | 17% | |
| 3 raw-closest twin (Levenshtein, unique) is the owner | 0.889 ± 0.030 (n 108, uniform 0.38) | 0.820 ± 0.017 (n 484, 0.33) | strong, small slice |
| 4 P(script \| state), learned on non-holdout train | MI 1.520 of 1.522 nats | — | deterministic |
| 4 best-state twin (Indic SX, unique best) is the owner | 0.672 ± 0.057 (n 67, uniform 0.20) | — | tiny slice |
| 5 identical raw address-less member pairs | in a group 3.4%; across same-raw-name S1 0 of 1,005 | 3.4%; 0 of 3,635 | identical → SAME owner |
| 5 in frame: identical-raw address-less sibling's owner = x's owner | 28 / 28 | 98 / 98 | copies of one record: no tie-breaker |
| 5b identical-raw ADDRESSED sibling at exactly one twin is the owner | 0.243 ± 0.013 (uniform 0.224) | 0.275 ± 0.010 (0.273) | none |

Law (train truth, every S1): India and US identical. S2 f = .130/.357/.296/.152/.054/.011 (cap 5), S3 cap 6, var/mean 0.74 (under-dispersed
→ the smaller twin is favoured). Poisson sizes would make r constant (no information).

## What failed      (approaches tried and dropped, with the number that killed them)
- The brief's h(k) form (no (k+1) factor). It predicts 0.54–0.57 top-twin ownership where 0.37–0.39 is observed.
- Test 5's "identical → other twin" hypothesis. The data says the reverse (100% same owner), and the copy carries no new evidence.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `frame.py [c]` → data/work/p49/tw_{c}.parquet (twin rows), s1_{c}.parquet. 22–30 s, < 1.9 GB.
- `law.py` → data/work/p49/law.json (f per country × source), `ratios(f)` → (h, r).
- `test1.py`, `test35.py`, `test5b.py`, `test4.py`: the tests; logs in `logs/`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Competitor (non-holdout train) S1 have only 16% of their addressed truth in the frame, so a label-free count for them cannot be
  computed on holdout. Test 1's 'addr' count uses truth-addressed members as a proxy for a ~99%-precise decision.

## Open questions   (for the master; each with your best guess)
- Pending.
