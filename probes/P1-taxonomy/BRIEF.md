# P1-taxonomy: data analysis for modelling
Goal: find where the F0.5 is, and build normalize.py v1 that recovers the easy part.
1. Difficulty taxonomy. On 100k true pairs from the train world, classify each pair (first match wins):
   exact-after-normalization · legal-suffix/abbreviation-only · typo (edit distance <= 2 on core name) ·
   word reorder · website-as-name · '|'-joined name/site · cross-script · no shared name token but address
   agrees · genuinely different surface form (DBA). Give shares per country and per source (S2 vs S3).
   Estimate F0.5-at-stake per class (share × how often a naive matcher misses it) and rank the classes.
2. Look-alikes (hard negatives): how many S1 share a normalized name within a country (chains)? For true pairs,
   how often does a wrong S1 with the same name exist? What address evidence separates them?
3. Address structure per country: segment order, city/state position, state abbreviations, PIN/ZIP presence,
   landmark phrases (Near/Opp/Behind — beware "Opp, AL" is a town), NULL placeholders. Define a city_key
   extraction rule and measure city_key agreement on true vs random pairs.
4. Mine substitution dictionaries automatically from aligned true pairs (token pairs that co-occur: pvt↔private,
   st↔street, co↔company …) and a legal-suffix inventory per country. French legal forms, street types and
   articles come from domain knowledge (see research_techniques.md §6); they can't be validated — label them.
Deliverable: probes/P1-taxonomy/normalize.py with name_norm, name_core (suffix-stripped), acronym, is_website,
website_core, addr_norm, city_key, state, postcode, house_number — vectorized over pandas Series, and a
self-check. Report the exact-match and token-overlap lift on true vs random pairs, before vs after.
