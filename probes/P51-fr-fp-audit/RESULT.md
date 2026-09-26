# P51-fr-fp-audit — RESULT (written by the master, 04:30; the Gemini run hit its quota, the GPT-OSS rerun's p51.py is REJECTED)
## TL;DR
- **No France false-positive cell with real support.** Scan: `cells_fix.csv` (France accepted pairs per S1 vs US/India
  holdout, by name relation × hn class × gained-word lens band × address-less; P31 proxy base).
- The only structural suspect, accepted pairs with a generator **sibling** house-number offset (France ≈ 2.5k), is clean
  in-domain: holdout accepted pairs with hc==1 are **0.981 (US, 2,346) / 0.989 (India, 2,984)** true (data/work/p42/ho_*
  joined to p23 pred). The whole-cell precision (0.04 / 0.41) is driven by pairs the model already rejects.
- `p51.py` (GPT-OSS) is **not to be used**: it flags cells by whole-cell precision (wrong statistic) and its top cell
  (add1 × equal × lens 0.05–0.25, 1,481 France pairs) has 0 US / 1 India labelled pairs; it would delete P46's
  développement/groupe adds that P46 validated.
## Recommendation
Nothing to stage. France's accepted pairs look as clean as India/US wherever there is support.
