# N50-fr-moved-addr: True "moved" entities vs Decoys

## TL;DR
Pairs where the SX name perfectly matches the S1 name (`name_raw_eq = True`), the city matches, but the street and house number are completely different (not a generator sibling offset), are a rich source of true variants **if restricted to S3** (where businesses frequently move or have poorly synced old addresses) and **unique names (`k=1`)**. 

In the US and India, the matcher already scores almost all of these true pairs highly (`p > 0.9`). Thus, within the `pred==False` slice (rejected by the matcher), precision sits near 0.0 because it correctly rejected the rare decoys while capturing the true pairs. 

However, in France (due to out-of-domain shift), the base model scores these at `p = 0.3 - 0.6` and rejects them (`acc == False`). Using the revised validation bar (negatives-per-S1 bound), adding these pairs for S3 passes comfortably.

**Recommendation:** ACCEPT unaccepted France argmax pairs that have exact name match, `k=1`, and `sx_source == S3`, despite a completely different house number and street in the same city.

## Task 1: Full Train Truth
By measuring the overall precision of pairs with different street and hn (non-sibling offset) in the train truth, grouped by exact name equality (`name_raw_eq`) and twin count `k`, we find that unique matching names (`k=1`) are overwhelmingly true pairs:

| Country | `name_raw_eq` | `k` | Pairs | True Pairs | Precision |
|---------|---------------|-----|-------|------------|-----------|
| India   | True          | 1   | 9,181 | 9,008      | 0.981     |
| US      | True          | 1   | 21,384| 21,311     | 0.996     |

## Task 2 & Validation Bar
When restricting to the `holdout` set and `pred==False`, the number of false (decoy) pairs is incredibly small. 
For `name_raw_eq=True`, `k=1`, and `sx_source == S3`:

* **India:** 8 false pairs in the holdout (out of 86,635 holdout S1). 
  * `neg_per_s1` = 8 / 86,635 = 0.0000923.
  * Expected France negatives ($N_{neg}$) = 0.0000923 × 259,452 = 23.96
* **US:** 0 false pairs in the holdout (out of 129,822 holdout S1).
  * `neg_per_s1` = 0.
  * Expected France negatives ($N_{neg}$) = 0.

By applying this rule on the France P31 proxy base (`argmax == True`, `acc == False`, `core_1 == core_x`, `city_1 == city_x`, different street/hn, non-sibling, `name_raw_eq == True`, `k == 1`, `sx_source == S3`), we **ADD 150 pairs**.

* **India-rate bound:** $1 - 23.96 / 150 = 0.8403$ (passes the $\geq 0.75$ bar)
* **US-rate bound:** $1 - 0 / 150 = 1.0$ (passes the $\geq 0.90$ bar)

Label-free checks confirm France does not have a disproportionate number of these pairs relative to US/India:
* Share of such pairs among total S3 argmax pairs: France (0.0011) vs India (0.0032) vs US (0.0040).

## Deliverable
`p50.py` has been written and implements the `build` script (with `selftest`) per the `AGY_COMMON.md` specification. It cleanly adds the 150 safe S3 variants to the specified output base.
