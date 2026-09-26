# N49-fr-fuzzy-street — France pairs with an exact name and same house number whose STREET string differs
Read probes/NIGHT/AGY_COMMON.md first, then probes/P28-addr-anchor/RESULT.md (same-address rule) and
code/business_entity_resolution/src/normalize.py (how streets / house numbers are normalised).

Observation (master, unaccepted France argmax pairs p 0.3–0.99 on the P31 base): pairs with the same (core) name and the
same house number and city, but the street tokens differ only by (a) French elision / apostrophes ("de l'Argonne" vs
"DE LARGONNE", "d’Agen" with a curly apostrophe), (b) abbreviation (R./Rue, AV/Avenue, BD, ALL./Allée, IMP/Impasse,
CRS/Cours, PL/Place, RTE/Route, CHE/Chemin, ST/Saint, GÉN/Général), (c) accents/case, (d) a generator typo (1–2 char
edit, swapped letters: "Naujac" vs "Jeacne"?) — are rejected at p 0.4–0.9. Also SX with the street but NO house number
while the S1 has one ("Rue Saint Bernard, Lille" vs "6 RUE Saint Bernard, Lille") with an identical name.

Task:
1. Audit normalize.py's street handling on France: how often do France argmax pairs with equal core name + equal hn +
   equal city have unequal normalised street, and which of (a)–(d) causes it (counts per cause). If (a)/(b)/(c) are plain
   normalisation bugs, write a fixed French street key (in your probe folder, not src/).
2. Define the rule: free argmax SX, core name equal (or P20 R1-equal), hn equal (or SX hn missing — separate arm),
   city equal, fuzzy street equal (your key, or token-sort ratio >= X after the key), not a sibling (hn_cls != 1).
3. Validate on US / India holdout truth (p23 dumps, holdout==True) with the analogous US/India street key: precision
   overall and among pred==False pairs, by p band and by arm; and on full train truth. Apply the VALIDATION BAR.
   The generator's decoys change the house number (sibling offsets) — confirm the equal-hn arms are clean and report the
   missing-hn arm separately (it is riskier: a decoy could drop the number).
4. p49.py build <base> <out> + selftest; counts on the P31 proxy base, overlap with what P28 (same-address), P37 hn-extra
   (p >= 0.8 hn-equal extras) and P40 xf_noaddr already cover — report only the NEW pairs as the gain.
Deadline: RESULT.md final by 07:30 IST.
