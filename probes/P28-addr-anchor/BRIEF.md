# P28-addr-anchor: "one name word swapped + same full address ⇒ same business" rule (user's idea), holdout-validated
User's idea: if the names differ by one word but the address is the same, rely on the address — two different companies
rarely share the exact address. Evidence (probes/P20-france-patterns/RESULT.md): France has ~75k "one core word swapped,
house number EQUAL" argmax pairs; #8c accepts only 35%, while the same cell in the US holdout is 89% true. Generator
siblings keep the hn only 2–6% (France) / 2.7% (US) / 26% (India). For F0.5 the accept break-even is P(true) ≈ F*/(1+β²)
≈ 0.79, so an 89%-true cell should be accepted. Reuse P20's code (probes/P20-france-patterns/fr20.py: the name-relation ×
hn-relation taxonomy, noise_core, sib_zero) and P15's hn helpers. Own worktree; code under probes/P28-addr-anchor/.
Task:
1. Define the rule precisely (label-free, general code): pair is the SX's argmax (or would be after the boost), name cores
   differ by exactly one token (swap/insert/delete; also the R1 filler case), house number present on both and EQUAL,
   street tokens equal (after normalisation) and city equal, NOT a sibling offset, SX not already matched elsewhere.
   Variants: (a) hn+city only, (b) hn+street+city, (c) (b) + chain guard (another S1 in the same city shares the hn → skip).
2. **Validate on the US/India full-corpus holdout (labels) on the #8c yardstick** (run.py --split holdout pipeline p;
   report half 1, paired SE): apply "p := max(p, 0.99)" (or accept regardless of t) for rule pairs; report precision of the
   rule pairs, how many there are, and ΔF0.5 per country and per variant. India behaves differently (hn noise) — report
   separately; the rule may be applied per country.
3. Apply the best variant to France on top of #9ab's France file (data/work/p19/prob_test_France_idf144_bge.parquet, t 0.985):
   count pairs affected, check P6's profile bands, build with
   `probes/P14-france-shift/combine.py output/sub8c <file> 0.985 output/sub28a` (+ validator). If the rule also helps US
   (holdout), build a variant applying it to US too.
RESULT.md with the table + recommendation. Memory 3 GB. Never end your turn to wait. Deadline 26 Sep 20:00.
