# N57-loco-cellscan — RESULT

## TL;DR
- Evaluated both gc70-style chain (current final France, no hn-extra) and the chain with hn-extra on LOCO frames.
- Cell enumeration covered combinations of `name_rel` × `band` (gained-word lens band) × `hrel` (hn relation) × `p-band` (matcher probability).
- **No cells survived the strict LOCO validation.** (No cells had positive ΔF0.5 in BOTH halves and BOTH directions with |Δ| > 2 SE each, for either addition of free argmax pairs or removal of accepted pairs).

## Recommendation
- A clean "no survivors". There are no additional simple rule-based cell additions or removals that unambiguously pass the two-country LOCO hurdle with statistical significance. 

## Evidence
- Conducted LOCO scanning using base (argmax & p ≥ t) with and without P42 hn-extra logic.
- Analyzed cross-country removals (for cells with ≥ 50 accepted pairs in both directions) and additions (for cells with ≥ 50 free argmax pairs, capped at 5 S2 / 6 S3 per S1).
- While some cells may have had positive impacts in one direction, none systematically passed the > 2 SE threshold for both halves across both US→India and India→US.
- The highest scoring near-miss removals were strong in only one direction and negative in the other, confirming they are country-specific quirks rather than universally valid patterns. For example, in the gc70 chain:
  - `('eq', 'none', 'hn_diff', '0.7-0.9')`: US→IN was negative (-0.0010 dF), but IN→US was strongly positive (+0.0075 dF).
  - `('add1', 'rare', 'hn_missing', '>=t')`: US→IN was positive (+0.0033 dF), but IN→US was slightly negative (-0.0001 dF).
- The lack of new universal robust patterns supports that the previous rule layers (like N54) have already captured the predictable low-hanging fruit.
