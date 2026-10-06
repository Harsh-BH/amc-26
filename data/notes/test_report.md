# AMC-26 test-split profile

## 1. Parse integrity & row counts

| file | wc -l (data rows) | parsed rows | match | columns |
|---|---:|---:|---|---|
| test_source1.tsv | 1,732,544 | 1,732,544 | OK | entity_id, business_name, business_address, country |
| test_source2.tsv | 4,887,273 | 4,887,273 | OK | entity_id, business_name, business_address, country |
| test_source3.tsv | 5,082,316 | 5,082,316 | OK | entity_id, business_name, business_address, country |

Parse used `sep='\t', quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False, engine='c'` (pyarrow engine raises `ValueError: quoting option not supported`, confirmed empirically before this run). All 3 files parse to exactly wc-1 rows -- no line-merging from embedded quote chars.

S1:S2:S3 row-count ratio = 1 : 2.821 : 2.933  (1,732,544 : 4,887,273 : 5,082,316)

Total entity count across all 3 test files: **11,702,133**. S1 alone (the entities requiring a submission row, per the PDF: "Generate matches for every entity in this file"): **1,732,544** -- matches the validator's docstring figure of "~1.7M-entity test set" almost exactly.

vs train: S1:S2:S3 ratio was 1 : 2.281 : 2.395 (2,206,821 : 5,034,616 : 5,285,603), train total = 12,527,040. Test ratio is different (2.821 vs 2.281, 2.933 vs 2.395) -- test S1 (1,732,544) is smaller than train S1 (2,206,821) by 21.5%.

## 2. Country distribution per source (exact label strings)

| source | country (repr) | rows | % of source |
|---|---|---:|---:|
| S1 | 'India' | 809,986 | 46.75% |
| S1 | 'US' | 663,106 | 38.27% |
| S1 | 'France' | 259,452 | 14.98% |
| S2 | 'India' | 2,312,565 | 47.32% |
| S2 | 'US' | 1,871,330 | 38.29% |
| S2 | 'France' | 703,378 | 14.39% |
| S3 | 'India' | 2,405,000 | 47.32% |
| S3 | 'US' | 1,945,701 | 38.28% |
| S3 | 'France' | 731,615 | 14.40% |

Exact set of country label strings seen across all 3 test files: `['France', 'India', 'US']`
vs train: `['India', 'US']` exactly, no variants. Test adds exactly one new label, ['France'] on top of the same US/India labels (same exact strings, no case/spelling drift observed).

## 3. Empty rate of name/address per source x country (vs train)

| source | country | rows | empty business_name | empty business_address |
|---|---|---:|---:|---:|
| S1 | France | 259,452 | 0.00% | 0.00% |
| S1 | India | 809,986 | 0.00% | 0.00% |
| S1 | US | 663,106 | 0.00% | 0.00% |
| S2 | France | 703,378 | 0.00% | 3.06% |
| S2 | India | 2,312,565 | 0.00% | 2.28% |
| S2 | US | 1,871,330 | 0.00% | 2.94% |
| S3 | France | 731,615 | 0.00% | 2.94% |
| S3 | India | 2,405,000 | 0.00% | 2.46% |
| S3 | US | 1,945,701 | 0.00% | 2.84% |

Comparison vs train (train empty business_address %; business_name was 0.00% everywhere in train):

| source | country | train empty addr% | test empty addr% | delta |
|---|---|---:|---:|---:|
| S1 | US | 0.00% | 0.00% | +0.00pp |
| S1 | India | 0.00% | 0.00% | +0.00pp |
| S2 | US | 3.68% | 2.94% | -0.74pp |
| S2 | India | 2.87% | 2.28% | -0.59pp |
| S3 | US | 3.50% | 2.84% | -0.66pp |
| S3 | India | 3.07% | 2.46% | -0.61pp |

## 4. France deep dive

`France` present as exact label: **True**

### S1 France sample (n=259,452)

| entity_id | business_name | business_address |
|---|---|---|
| S1-545867313 | Bien & Frères SAS | 23 Boulevard de la Renaissance, Saint-Nazaire, Pays de la Loire |
| S1-919814052 | Collège Pierre | 19 Rue Margaux, Bordeaux, Nouvelle-Aquitaine |
| S1-29566510 | Films & Fils SARL | 160 Rue de l'Hommelet, Roubaix, Hauts-de-France |
| S1-791938329 | Dunkerque Comite SAS | 140 Avenue Adolphe Geeraert, Dunkerque, Hauts-de-France |
| S1-839371455 | UF Lycée SARL | 22 Rue du Honduras, Saint-Nazaire, Pays de la Loire |
| S1-162325484 | Deleves Club SARL | 467 Rue du Congo, Tourcoing, Hauts-de-France |
| S1-405615910 | Organisme (France) Amis SARL | 2 Impasse Docteur Guérin, Saint-Herblain, Pays de la Loire |
| S1-189128768 | Jeunesse Compagnie EURL | 1 Chemin de la Pelleterie, Nantes, Pays de la Loire |
| S1-712906518 | Lycée Saint Scolaire | 63 RUE de la Perverie, BP42203, Nantes, Pays de la Loire |
| S1-59386384 | Tour Collège | Nouvelle-Aquitaine, Bordeaux, 66 Rue du Commandant Arnould |
| S1-365915221 | Syndicat Sante SARL | 57 Rue Monseigneur Piedfort, Calais, Hauts-de-France |
| S1-605090290 | La Teste-de-Buch Groupe SARL | 16 d Rue de Maugis, La Teste-de-Buch, Nouvelle-Aquitaine |
| S1-698903008 | Mérignac Ecole SARL | 15 Rue Bertheau, Mérignac, Nouvelle-Aquitaine |
| S1-555652172 | Trame Pharmacie SAS | 254 Cours de l'Yser, Bordeaux, Nouvelle-Aquitaine |
| S1-861502065 | Club des Lussat | 999 Allée des Colchiques, La Teste-de-Buch, Nouvelle-Aquitaine |

**S1 France patterns** (n=259,452): 5-digit postcode in address 0.4%; rue/avenue/bd/etc street-type token 94.5%; CEDEX 0.0%; SARL/SAS/SASU/SA/EURL/SCI/EI legal-form token in name 68.9%; accented Latin char (é/à/ç/...) 38.6%; apostrophe 5.6%; CJK script 0.00%; Arabic script 0.00%; Devanagari/Indic script 0.00%; empty name 0.00%; empty address 0.00%.

### S2 France sample (n=703,378)

| entity_id | business_name | business_address |
|---|---|---|
| S2-456838151 | Pharmacie Dëfi | 4 IMP. DE BORDEAUX, LA BAULE-ESCOUBLAC, Loire-Atlantique |
| S2-900915904 | Dorganisation (France) Culturel SARL | 3 R. OSISP ZADKINE, CALAIS, Hauts-de-France |
| S2-780824528 | fdration.com | 17 RUE DES 3 OIMEAUX, Nantes, Loire-Atlantique |
| S2-927141655 | Fédération de Communal Distribution SCI | (30) R. PEIRRE BAUMANN, LILLE |
| S2-765579974 | Salon Amis Sa | DUNKERQUE, 4 RUE MARENGO |
| S2-172379997 | UMR Élémëntaire | Hauts-de-France, 265 RUE DE ZUYDCOOTE, DUNKERQUE |
| S2-973309639 | telectric.com | Loire-Atlantique, Nantes, 18 R DE LA GOURMETTE |
| S2-187121795 | Corporative Club EI | 1 RUE BELLUS MAREILHAC, BORDEAUX |
| S2-240225044 | Club Passion de | 826 QUAI GUSTAVE LAMARLE, CALAIS, Hauts-de-France |
| S2-40094195 | Modelistes Societe Sa | LA BAULE-ESCOUBLAC, 6 ALLÉE DES IRIS |
| S2-408346169 | Pharmacie de la Notre SA | 36 COURS JOURNU AUBER, BORDEAUX |
| S2-805538118 | culte deducation sportive participations sa | 53 R DE TURENNE, CALAIS |
| S2-607876408 | FERME FETES | Nord, 14 R HIPPOLYTE LEFEBVRE, LILLE |
| S2-485403045 | FRATERNITE AMICALE DÉVELOPPEMENT  SARL | 170 AVENUE GUSTAVE DELORY, ROUBAIX |
| S2-669556034 | La Teste-de-Buch Section SA | 7 IMPASSE DES FRÈRES CAUDRON, LA TESTE-DE-BUCH |

**S2 France patterns** (n=703,378): 5-digit postcode in address 0.5%; rue/avenue/bd/etc street-type token 59.0%; CEDEX 0.0%; SARL/SAS/SASU/SA/EURL/SCI/EI legal-form token in name 49.0%; accented Latin char (é/à/ç/...) 37.9%; apostrophe 5.4%; CJK script 0.00%; Arabic script 0.00%; Devanagari/Indic script 0.00%; empty name 0.00%; empty address 3.06%.

### S3 France sample (n=731,615)

| entity_id | business_name | business_address |
|---|---|---|
| S3-412791493 | Maison Domicile Distribution SARL | 41 Boulevard Trystram, Dunkerque, Hauts-de-France |
| S3-594124696 | Amicale Dû Benevoles | 118 Rue De Douai, Lille, Nord |
| S3-308271973 | Francaise Amicale SARL | Lille, 44 Rue Des Ecoles, Hauts-de-France |
| S3-429286940 | Ets Çommercants EURL | 13 Rue Jean Dollfus, Bordeaux, Nouvelle-Aquitaine |
| S3-986823532 | Altern Societe | 15 R Des Bordes, La Teste-de-buch, Gironde |
| S3-698209621 | Centre Hiatus Médical | N°16 Allee Du Corail, Bordeaux, Gironde |
| S3-122389486 | Collège Smr SA + Associés | #22 Avenue Du Semaphore Pyla, La Teste-de-buch, Gironde |
| S3-826716811 | Amical Sport | 14 Avenue Du General Leclerc, Pessac, Nouvelle-Aquitaine |
| S3-318348462 | Lège-Cap-Ferret Anciens | 1 Aelee Du Riou, Lege-cap-ferret |
| S3-963767083 | SARL Maison Cercle | 9 Rue Gustave Courbet, Dunkerque |
| S3-15079560 | ASSOCIATION DES MECANOS HOLDING | 0059 R Alrard Dugauquier, Lille, Nord |
| S3-580543711 | Locale Sportive SARL | 6 Avenue Du Lonig, Nantes |
| S3-844133211 | Collège du Jour EURL | 3 Rue Marengo, Dunkerque |
| S3-913171994 | Mini Primaire Du école | 47 Rue Jean Saint-marc, La Teste-de-Buch |
| S3-136667913 | Vie Mâintien Club SA | 56 Boulevard D’acrachon, La Teste-de-buch |

**S3 France patterns** (n=731,615): 5-digit postcode in address 0.5%; rue/avenue/bd/etc street-type token 60.4%; CEDEX 0.0%; SARL/SAS/SASU/SA/EURL/SCI/EI legal-form token in name 46.5%; accented Latin char (é/à/ç/...) 37.8%; apostrophe 5.4%; CJK script 0.00%; Arabic script 0.00%; Devanagari/Indic script 0.00%; empty name 0.00%; empty address 2.94%.

France entity counts: S1=259,452, S2=703,378, S3=731,615, total=1,694,445 (14.48% of all test rows).

## 5. Script mix per source x country (vs train)

| source | country | % name w/ Indic | % addr w/ Indic | % name w/ CJK | % name w/ Arabic | % name non-ASCII (any) |
|---|---|---:|---:|---:|---:|---:|
| S1 | France | 0.00% | 0.00% | 0.00% | 0.00% | 15.72% |
| S1 | India | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| S1 | US | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| S2 | France | 0.00% | 0.00% | 0.00% | 0.00% | 24.54% |
| S2 | India | 23.64% | 23.81% | 0.00% | 0.00% | 27.61% |
| S2 | US | 0.00% | 0.00% | 0.00% | 0.00% | 6.25% |
| S3 | France | 0.00% | 0.00% | 0.00% | 0.00% | 23.94% |
| S3 | India | 13.33% | 22.89% | 0.00% | 0.00% | 18.21% |
| S3 | US | 0.00% | 0.00% | 0.00% | 0.00% | 6.39% |

vs train (India Indic-script rates): S1/India 0.00%, S2/India name 23.51% / addr 23.67%, S3/India name 13.17% / addr 22.49%; US and S1 always 0.00% in train.

## 6. Anything that would break a pipeline

- **S1**: duplicate entity_id rows: 0; empty entity_id: 0; empty country: 0; both name+address empty: 0; entity_id NOT prefixed `S1-`: 0; control chars in name/address: 0/72; literal tab inside name/address field: 0/0; embedded CR/LF inside name/address field: 0/0; BOM at start of name/entity_id: 0/0; rows with a literal `"` in name/address: 132/2.
- **S2**: duplicate entity_id rows: 0; empty entity_id: 0; empty country: 0; both name+address empty: 0; entity_id NOT prefixed `S2-`: 0; control chars in name/address: 0/183; literal tab inside name/address field: 0/0; embedded CR/LF inside name/address field: 0/0; BOM at start of name/entity_id: 0/0; rows with a literal `"` in name/address: 346/3.
- **S3**: duplicate entity_id rows: 0; empty entity_id: 0; empty country: 0; both name+address empty: 0; entity_id NOT prefixed `S3-`: 0; control chars in name/address: 0/162; literal tab inside name/address field: 0/0; embedded CR/LF inside name/address field: 0/0; BOM at start of name/entity_id: 0/0; rows with a literal `"` in name/address: 324/6.
- Second missingness encoding (train also had this): literal `NULL`/`<NULL>` placeholder token in business_address on top of genuinely-empty strings -- S1 18 rows (0.00%), S2 105,296 (2.15%), S3 105,751 (2.08%). Matches train's ~2.5-2.9% rate closely.
- `|`-joined trade-name/website in business_name (train also had this, ~0.3%): S1 0, S2 11,856 (0.24%), S3 12,260 (0.24%).
- URL/website-as-name (`.com`/`www`/etc.): S1 0, S2 156,040 (3.19%), S3 163,994 (3.23%) -- consistent with train's ~4%.
- No empty rows, no header/column drift, no zero-byte or truncated files (parse integrity in Section 1 already confirms every row is well-formed and none were dropped or merged).

## 7. Rough compute sizing (blocking by country)

| country | S1 | S2 | S3 | S2+S3 | S1 x (S2+S3) pairs | @k=10/S1 candidates | @k=50/S1 candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| France | 259,452 | 703,378 | 731,615 | 1,434,993 | 372,311,803,836 | 2,594,520 | 12,972,600 |
| India | 809,986 | 2,312,565 | 2,405,000 | 4,717,565 | 3,821,161,604,090 | 8,099,860 | 40,499,300 |
| US | 663,106 | 1,871,330 | 1,945,701 | 3,817,031 | 2,531,096,158,286 | 6,631,060 | 33,155,300 |
| **total** | 1,732,544 | 4,887,273 | 5,082,316 | 9,969,589 | 6,724,569,566,212 | 17,325,440 | 86,627,200 |

Full country-scoped cross join = 6,724,569,566,212 candidate pairs -- infeasible to score exhaustively (a naive all-pairs matcher over just the largest country would dominate). A k=10-neighbours-per-S1-entity blocker caps candidates at 17,325,440 pairs (0.0003% of the full cross join); k=50 caps at 86,627,200 (0.00%). Both are a tiny, tractable fraction of the full join, so the blocking stage (not the pairwise scorer) is what determines feasibility.


_(memory checkpoint after test frames loaded: RSS = 2760560 kB)_

## 8. Test/train ID overlap and exact-duplicate records

_(memory checkpoint after dropping test frames, before loading train: RSS = 5541444 kB)_

| source | train id count | test-id ∩ train-id (same source) |
|---|---:|---:|
| S1 | 2,206,821 | 0 |
| S2 | 5,034,616 | 0 |
| S3 | 5,285,603 | 0 |

Entity-id overlap between train and test (same-source id string appearing in both files): NONE -- disjoint id spaces.

Exact-duplicate records (identical business_name + business_address + country, hashed) found in a train row that also match a test row (any source): **0** train rows hash-collide with a test row (hash collisions are possible but astronomically unlikely at this scale with a 64-bit hash over ~12M x ~11.7M comparisons -- treat this as an exact-duplicate count, not just a hash-collision artifact).

---
## Implications (max 6)

- **France ships with zero labeled examples anywhere** (train has none, and the leakage check confirms zero ID/record overlap with train) -- any France-specific threshold, weight, or blocking parameter must come from structural priors (name-token overlap, legal-form normalization, address-parsing), not from calibration against known true/false pairs the way US/India can be tuned against train's ground truth. Hold out a France-shaped synthetic slice from US/India to sanity-check generalization before trusting France scores.
- **Country mix shifted, not just added a 3rd label**: within non-France rows, test is ~45% US / 55% India (663,106 : 809,986) vs train's ~60% US / 40% India -- a blocking/matching pipeline tuned on train's US-heavy mix will see proportionally far more India traffic (and India's higher Indic-script-crossing rate) at test time than it saw examples of.
- **France needs its own address-normalization dictionary**, same role as US "Rd/Road" or India "Pvt/Private": legal-form tokens SARL/SAS/SASU/SA/EURL/SCI/EI appear in 47-69% of France names (vs 0% in train's US/India), 5-digit postal codes are rare (0.4-0.5%, unlike the higher-signal postal-match feature train relied on) so the train report's "postal-code-equality = 94% precision" early-accept key will barely fire for France and needs a French-address-specific replacement (arrondissement/commune name, rue/avenue/bd token match).
- **Same noise generator, no new script risk**: France samples show the identical corruption patterns as US/India (typos like "Dëfi"/"fdration.com"/"PEIRRE", missing address parts, ~3% empty S2/S3 addresses) and 0% CJK/Arabic/Devanagari -- non-ASCII in France is entirely accented Latin (é/à/ç, 38%) plus apostrophes (5.5%), so existing Unicode-normalization code (NFKD strip-accents) handles it; no new script-detection branch is needed.
- **Compute is fine at any reasonable k**: full country-scoped cross join is ~6.7T pairs (infeasible), but k=10 candidates/S1-entity caps at 17.3M pairs and k=50 at 86.6M -- both trivially scoreable by a lightweight pairwise model; blocking quality (recall ceiling), not pairwise compute, remains the bottleneck, same conclusion as train.
- **Pipeline hygiene is clean and matches train**: no dup entity_ids, no empty ids/rows, no embedded tabs/CR/LF, no BOM, quoting never merges lines (Section 1 parse-integrity match is exact). The same two pre-existing data quirks from train carry over unchanged and must still be cleaned before feature extraction: literal `NULL`/`<NULL>` placeholders in ~2.1-2.2% of S2/S3 addresses, and `|`-joined name/website strings in ~0.24% of S2/S3 names.

