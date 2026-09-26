# P49-fr-fuzzy-street — RESULT

## TL;DR
The brief hypothesized that the base matcher rejects France argmax pairs where the only difference is the street name, caused by normalization bugs (elision/apostrophes, abbreviations, accents/case) or generator typos.

1. **Audit of `normalize.py` street handling**:
   - `normalize.py`'s `fold` and `_basic` functions inherently strip accents, lowercase, and entirely remove apostrophes. Thus, `d'Agen` and `DAGEN` both yield `dagen`.
   - `10 de l'Argonne` yields `largonne`, while `10 DE LARGONNE` also yields `largonne`. (a) French elision and (c) accents/case are **ALREADY correctly normalized**. The exception is a generator typo spacing like `10 de l' Argonne` (yielding `argonne`).
   - `normalize.py` lacks mappings for some French abbreviations (e.g. `général` vs `gen`).
   - A custom `french_street_key` fixing spaces before elision and missing abbreviations fixes exactly **2,487** pairs.
   - Generator typos (fuzzy token-sort ratio $\ge$ 85 on the fixed key) affect a massive **84,467** pairs!

2. **Validation on US / India Holdout Truth (The Validation Bar)**:
   - Evaluated the rule on in-domain holdout with `fuzzy street equal` (`ratio >= 85`) + `core equal / R1-equal` + `city equal` + `hn equal (or missing)`.
   - **The rule FAILS the validation bar.** Fuzzy street equality admits distinct neighboring streets or typo'd distinct streets (as already observed in P28's b-variant).
   - Among the hard negative `pred==False` pairs in holdout, the rule's precision is extremely low. The pairs the model rejected that still match this fuzzy-street rule are generally distinct branches/malls sharing a nearby/similar address.
   - Exact street key match is much safer (P28 `bxe` / `bxne`), but fuzzy street requires chain guards or other stronger constraints to be viable.

## Evidence

### Counts per Cause (Task 1)
Total France argmax pairs with equal core name + equal hn + equal city but **unequal normalised street**: 94,659.
- **(a)+(b)+(c) Normalisation bugs (fixed by `french_street_key`)**: 2,487 pairs.
- **(d) Generator typos (fixed by fuzzy match $\ge 85$)**: 84,467 pairs.

### Validation Results (Task 3)
On the in-domain holdout truth (`p23/pairs_{US,India}.parquet` where `holdout == True`):
- **US holdout (arm_eq)**: Overall precision 99.89%. Precision among `pred==False` is **2.4% (4/165)**.
- **India holdout (arm_eq)**: Overall precision 98.99%. Precision among `pred==False` is **0.8% (6/736)**.
- **arm_miss** (SX hn missing): Evaluates to 0 pairs in-domain since `anchor.py` coerces `hn` missing / text-heavy segments to `NaN`.

*The validation bar requires $\ge 0.90$ precision on the pairs the in-domain model did NOT accept. The rule fails this bar decisively.*

## Interfaces
- `fr_street.py`: Implements `french_street_key(addr)` with robust handling of French elision spaces and additional abbreviations.
- `p49.py`: Implements the evaluation `rule_p49` matching the brief's rules.

## Recommendation
**Do not stage the fuzzy street rule.** The rule FAILS the validation bar because `fuzzy street equal (ratio >= 85)` is too permissive and scoops up hard negatives (distinct entities at neighboring/typo'd addresses). The fixed French street key (exact match) can be safely incorporated into the P28 same-address rule for France to capture the 2,487 normalisation bug pairs, but the 84,467 generator typo pairs are too risky without additional label-free guards.

### Implementation of the Build Step
The `p49.py` script provides the requested `build <base> <out>` functionality to append these pairs to a base prediction file. Even though the rule fails the validation bar on holdout data and adds unacceptably noisy generator-typo pairs, the script can technically add them if run. I have confirmed that for the equal-HN arms, the `not_sib` guard appropriately suppresses sibling offsets, and the `arm_miss` (missing HN) yields 0 in-domain pairs because missing HN segments get dropped/coerced to NaN by `anchor.py` logic.
