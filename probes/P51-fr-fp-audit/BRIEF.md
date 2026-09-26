# P51-fr-fp-audit — the precision side of France: accepted France pairs that are likely FALSE
Read probes/NIGHT/AGY_COMMON.md first (incl. the REVISED VALIDATION BAR and MEMORY GOTCHAS), then P46/P48 RESULT.md
(the lens: per gained word, sibling-offset share s and same-hn share e; decoy-marker words like "holding" have s≈0.9).

Every France probe so far ADDED pairs (recall). Nobody audited France's ACCEPTED pairs for false positives. The generator's
decoys carry decoy-marker words (holding / participations / distribution / club / ecole / france ...: high s in the lens)
usually WITH a sibling house-number offset — but a decoy that keeps the S1's house number would be accepted today.

Task (label-free on France, labelled analogues on US/India holdout p23 dumps, holdout==True):
1. Shape-excess scan. Define cells over ACCEPTED argmax pairs (France: acc==True in data/work/p31/pairs_France.parquet;
   US/India: pred==True on holdout): name_rel × hn class (equal / sibling / other / missing) × gained-word lens band
   (s <= 0.05, 0.05–0.25, 0.25–0.5, > 0.5; lens learned per country on its own argmax pairs as P46/P48 do) × address-less.
   Report accepted pairs PER S1 in each cell for France vs US vs India, and the holdout precision of the cell (US, India).
   A France cell with markedly MORE accepted pairs per S1 than US/India AND low US/India precision is a FP suspect.
2. For the top suspects: estimate France FPs = France per-S1 excess × 259,452 (assume US/India per-S1 is the truth rate),
   and what REMOVING the cell's France pairs would do: expected LB ≈ (#removed_false × ~0.2 − #removed_true × ~0.08)/1.73M
   (removing a false pair from a group mostly restores precision; removing a true pair costs recall; be explicit).
   Removal only makes sense if the cell's France precision is estimated < ~0.7.
3. Also check group-size excess: France S1 with >= 5 accepted S2 or >= 6 S3, vs US/India predicted/truth shares
   (P47 brief: truth S2 count shares 0/1/2/3/4/5 = .130/.358/.296/.151/.054/.011) — which pairs in over-full groups
   are the weakest.
4. If a removal rule passes (France est. precision of removed pairs < 0.6, both analogues agree), write p51.py
   `build <base> <out>` that REMOVES those France pairs (India/US byte-identical, candidates unchanged) + selftest.
   A clean negative ("France accepted pairs look as clean as US/India in every cell") is a valuable result too.
Deadline: RESULT.md final by 08:00 IST.
