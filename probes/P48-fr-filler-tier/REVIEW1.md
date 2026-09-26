# Master review of RESULT.md v1 — NOT accepted yet. Fix these, then update RESULT.md (keep v1 numbers in "What failed").

1. The loose lens (s <= 0.25, e >= 0.75) contradicts P46's holdout evidence: on the US holdout, P46's lens cells were
   s <= 0.05: 94.6% true; 0.05–0.09: 87%; 0.09–0.14: 72%; 0.14–0.2: **21%**. You never validated YOUR rule on labels.
   REQUIRED: run the EXACT p48 rule (drop exactly one S1 word + gained phrase, identical hn+street+city, lens learned on
   that country's own argmax pairs exactly as p48 learns it on France) on the US and India holdout dumps
   (data/work/p23/pairs_*.parquet, holdout==True; join raw names/addresses from the train sources and rf features),
   and report precision of the pairs it would ADD (pred==False, SX free), by lens band s <= 0.05 / 0.05–0.15 /
   0.15–0.25, overall and per gained phrase for the top phrases. The analogue fillers in US/India will be different
   words (e.g. "& sons", "group", "services", "inc"): that is fine — the rule is word-agnostic, the lens is learned.
2. Two variants: STRICT (s <= 0.05, e >= 0.8, like P46) and your LOOSE one. Stage STRICT unless LOOSE's extra band is
   >= 0.90 precise on BOTH US and India holdout adds.
3. Overlap accounting. The current best base already contains (a) P46 (développement / groupe one-word swaps),
   (b) P42/P37 hn-extra: free argmax non-sibling SX with hn equal to the S1's and p >= 0.80 in non-empty groups,
   (c) P28 same-address rule, (d) P41 gib, (e) P40 xf_noaddr. On the P31 proxy base, report how many of your adds
   survive after removing pairs that (a)/(b) would take (for (b): hn-equal & p >= 0.8 & S1 already has an accepted
   member), i.e. the truly NEW pairs, per variant. Expected LB gain ≈ new_pairs × (2·prec − 1) × ~0.08 / 1.73M is a
   rough guide; report it.
4. Build output only if a local base exists; otherwise the script + counts are the deliverable. Selftest must pass.
Memory: another two probes run in parallel — keep one python job at a time, ≤ 1.7 GB.
