# P1-taxonomy — RESULT  (DRAFT, being updated)
## TL;DR
normalize.py v1 lifts exact-name agreement on true pairs from 21.9% (s0) to 59.8% (`name_core`); random pairs stay at 0.00%.
After normalisation, recall-side F0.5 at stake: cross-script 3.5 pts > partial overlap 1.9 > website 1.6 > typo 1.1 > no-token 0.5.
Precision risk: 50% of S1 share their name_core with another S1 of the same country. Address separates them (house no. 66.6% vs 0.3%).

## Evidence (so far; train-world sample = 30k S1, 103,906 true pairs, 100k random same-country pairs)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| lift-name-addr-v1.1 | train-sample30k | name exact s0 0.219 → core 0.598 (rand 0); any-token s0 0.855/0.157 → core 0.866/0.009; house no. eq 0.867/0.004; city_freq eq 0.792/0.011 | keep |
| taxonomy-v1 | train-sample30k | see table below | keep |
| lookalike-chains-v1 | train S1 2.2M + sample | 49% of exact-core true pairs have a wrong same-core S1; address score picks the true S1 strictly in 89.9%, ties 10.0%, wrong 0.18% | keep |
