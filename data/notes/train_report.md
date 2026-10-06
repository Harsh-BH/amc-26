# AMC-26 train-split profile

## 1. Parse integrity & row counts

| file | wc -l (data rows) | parsed rows | match | columns |
|---|---:|---:|---|---|
| train_source1.tsv | 2,206,821 | 2,206,821 | OK | entity_id, business_name, business_address, country |
| train_source2.tsv | 5,034,616 | 5,034,616 | OK | entity_id, business_name, business_address, country |
| train_source3.tsv | 5,285,603 | 5,285,603 | OK | entity_id, business_name, business_address, country |
| train_ground_truth.tsv | 2,206,821 | 2,206,821 | OK | source1_entity_id, matched_entity_ids |

Parse used `sep='\t', quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False, engine='c'` (pyarrow engine rejects `quoting=`). All 4 files parse to exactly wc-1 rows -- no line-merging from the embedded `"` chars confirmed present in source1/source2 (they occur inside addresses, never break a line because the field is never comma/tab-quoted).

S1:S2:S3 row-count ratio = 1 : 2.281 : 2.395  (2,206,821 : 5,034,616 : 5,285,603)

## 2. Country distribution & empty-field rates per source

| source | country | rows | % of source | empty business_name | empty business_address |
|---|---|---:|---:|---:|---:|
| S1 | 'US' | 1,323,633 | 59.98% | 0.00% | 0.00% |
| S1 | 'India' | 883,188 | 40.02% | 0.00% | 0.00% |
| S2 | 'US' | 3,016,817 | 59.92% | 0.00% | 3.68% |
| S2 | 'India' | 2,017,799 | 40.08% | 0.00% | 2.87% |
| S3 | 'US' | 3,170,056 | 59.98% | 0.00% | 3.50% |
| S3 | 'India' | 2,115,547 | 40.02% | 0.00% | 3.07% |

## 3. Ground truth structure

- Exactly one GT row per S1 entity, and GT id-set == S1 id-set: **True** (2,206,821 unique S1 ids in GT vs 2,206,821 in source1).

Match-list size distribution (0 = singleton):

| size | rows | share |
|---|---:|---:|
| 0 | 123,247 | 5.58% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.94% |
| 5+ | 574,249 | 26.02% |

**Singleton fraction (overall): 5.5848%**

| country | rows | singleton fraction |
|---|---:|---:|
| India | 883,188 | 5.5878% |
| US | 1,323,633 | 5.5828% |

Of 7,638,365 total match references: S2 = 3,693,619 (48.36%), S3 = 3,944,746 (51.64%).

| non-singleton clusters | count | share |
|---|---:|---:|
| both | 1,776,047 | 85.24% |
| S3-only | 164,498 | 7.89% |
| S2-only | 143,029 | 6.86% |

## 4. Referential structure of matches

- Every S2/S3 id referenced by at most one S1 (strict 1-to-many): **True** (max references to a single S2/S3 id = 1; 0 ids referenced >1x).
- S2 distractor rate (never matched): 26.64%  (1,340,997 / 5,034,616)
- S3 distractor rate (never matched): 25.37%  (1,340,857 / 5,285,603)
- S1 clusters with >1 S2 id in the same cluster: 1,129,968 (54.23% of non-singleton clusters), max 5.
- S1 clusters with >1 S3 id in the same cluster: 1,224,128 (58.75% of non-singleton clusters), max 6.

## 5. Script mix (Indic vs Latin)

| source | country | % business_name w/ Indic chars | % business_address w/ Indic chars |
|---|---|---:|---:|
| S1 | India | 0.00% | 0.00% |
| S1 | US | 0.00% | 0.00% |
| S2 | India | 23.51% | 23.67% |
| S2 | US | 0.00% | 0.00% |
| S3 | India | 13.17% | 22.49% |
| S3 | US | 0.00% | 0.00% |

India matched pairs (3,059,843): S1 side has Indic chars in 0.00%, matched side in 18.02%. Pairs that **cross scripts** (one side Indic, other Latin): 18.02%. Both sides Indic: 0.00%.

## 6. Signal strength: TRUE pairs vs RANDOM pairs (same country)

Sampled 50,000 TRUE pairs and 50,000 RANDOM same-country pairs (country mix matched to the true-pair mix: {'US': np.float64(0.597), 'India': np.float64(0.403)}).

| signal | TRUE pairs | RANDOM pairs |
|---|---:|---:|
| exact normalized-name match | 21.81% | 0.00% |
| >=1 shared name token | 85.67% | 15.84% |
| both sides have a 5-6 digit postal-like number | 5.20% | 0.78% |
| postal numbers match (of those where both have one) | 94.11% | 0.00% |
| shared comma-segment ("city-ish") token | 87.25% | 2.59% |
| country label equal | 100.00% | 100.00% |

## 7. ID / row-order leakage check

- Correlation(S1 numeric id, matched numeric id) over 50,000 pairs: **-0.0041**
- Correlation(S1 row position, matched-record row position) over 50,000 pairs: **0.0041**
- S1 numeric id range: [15,163, 999,990,872]; matched-record numeric id range: [65,472, 999,997,748]
  (near-zero correlation => ids/row order carry no shortcut signal for matching; both look ~randomly assigned)

## 8. Trivial baseline scores (train, macro F_0.5)

- All-empty submission: **5.5848%** (= singleton fraction; every non-singleton scores 0, every singleton scores 1).
- Perfect-precision, exactly-one-match-per-S1 oracle (singletons predicted empty; non-singletons get 1 correct match out of k true matches, so precision=1, recall=1/k, F_0.5=1.25/(0.25k+1)): **69.6388%** overall (non-singleton subset alone: 67.8429%).
  This isolates what recall is worth once precision is perfect: going from the empty baseline to "one right guess per entity" only buys 64.0540% because most non-singletons need 2-5+ matches for full credit.

## 9. Illustrative matched clusters (candidates, by noise type)

### US

**abbreviation**
- `S1-11454997` Metro Holding Company Corporation | NC, Raleigh, 3860 Pin Oak Road
    - `S2-962646715` Metro Htdning Company Corporation | NC, 3860 PIN OAK RD, RALEIGH
    - `S2-66871487` Metro Holding-Company | 3860 PIN OAK RD, RALEIGH, NC
    - `S3-259782554` holdingmetro.com | 3860 Pin Oak Road, Raleigh, North Carolina
    - `S3-797182676` Metro Holding Co Corporation | 3860 Pin Oak Rd, Raleigh, North Carolina
- `S1-13731184` Hogue, Marquez & Ramirez Inc | CA, 12405 Ojai Santa Paula Road, Ojai
    - `S3-791040979` Hogue, Marquez & Ramirez (Incorporated) | 12405 Ojai Santa Paula Rd, NULL, OAJI, California
    - `S3-972605814` Hogue, Marquez & Ramirez Incorporated | 12405 Ojai Santa Paula Road, NULL, OAJI, California
    - `S3-24804832` Hogue, Marquez Ramirez Inc Services | 12405 Ojai Santa Paula Rd, NULL, OAJI, California
    - `S3-710307063` Hogue, Marquez Ramirez & Inc | 12405 Ojai Santa Palua Rd, NULL, OAJI, California
    - `S3-280401489` Deltahalo Co f/k/a Hogue, Marquez & Ramirez Inc | 12405 Ojai Santa Paula Road, Ojai, CA

**landmark**
- `S1-613197691` Tober's Marketing Inc | 17794 Substation Road, Opp, AL
    - `S2-19138714` Inc Tober'S Marketing | 17794C SUBSTATION RD, OPP, AL
    - `S3-678227220` Tober'S Marketing Inc | 17794 Substation Road, Opp, Alabama
    - `S3-610237151` Tober's Marketing | 17794 Substation Road, Opp, Alabama

**missing_parts**
- `S1-101169573` H/Q Chiron | 1151 Cottonwood Lane, Marengo, IL
    - `S2-955814120` H/Q Chiron | 1151 COTTONWOBD LN, MARENGO, PO BOX 9441, IL
    - `S2-106515820` H/Q Chiron Corporation | 1151 COTTONWOOD LN, PO BOX 9441, MARENGO, IL
    - `S3-990705378` H/Q Chir0n | 1151- Cottonwood Ln, Marengo, Illinois
    - `S3-390129833` H/Q Chiron [LP] | 
- `S1-101218094` Cherianne Belliveau Air Inc | 1927 Roberta Drive, Decatur, IL
    - `S2-552298266` cherianne belliveau air inc | 1927 ROBERTA DR, LONG CREEK TOWNSHIP, IL
    - `S2-835524235` Cherianne Air Belliveau Inc | 1927 ROBERTA DRIVE, DECATUR, IL
    - `S3-180852869` Cherianne Bllivaeu Air-Inc | 1927 Roberta Drive, Dcatur CITY, Illinois
    - `S3-18300673` Cherianne Belliveau Air [Incorporated] | 

**dba_website**
- `S1-100038073` Global Clean Ares | 1577 Powell Circle, Elba, AL
    - `S2-96321333` Global Ares Clean | #1577 POWELL CIR, AL, ELBA
    - `S2-994499450` GLOBALCLEANARES.COM | #1577 Powell Cir, ELBA, AL
    - `S3-654849095` Global Clean-Arfehs | 1577 Powell Cir, Elba, Alabama
    - `S3-204470450` Global Clean  Ares | 1577 Powell Circle, Elba, Alabama
    - `S3-885321887` G1obal Clean Center | 1577 Powell Cir, Elba, Alabama
    - `S3-997776625` Fluxfluxiri doing business as Global Clean Ares | 1577 Powell Circle, Elba, AL
- `S1-100215616` Weiss Vance Center | 274 Campers Lane, Canton, NC
    - `S2-965377728` Weiss Vance Center | 274 Campers Lane, CANTON, NC
    - `S2-987219974` weissvancecenter.com | 274 CAMPERS LN, CANTON, NC
    - `S3-26866348` Weiss Vance Centfr | 274 Campers Ln, Canton, North Carolina
    - `S3-193479704` Synirisol DBA: Weiss Vance Center | 274 Campers Lane, Canton, NC

**no_token_overlap**
- `S1-10010172` Twisted Deli | 4 Dix Road, Woburn, MA
    - `S2-238973954` TWISTED DELI CENTER | 4 DIX ROAD, WOBURN, MA
    - `S3-221049389` @twisteddeli | ##4 Dix Road, <NULL>, Woburn, Massachusetts
    - `S3-837584068` Twisteddeli.Com | ##4 Dix Rd, <NULL>, Woburn, Massachusetts
    - `S3-61561752` Brixkelo f/k/a Twisted Deli | 4 Dix Road, MA, Woburn
- `S1-100652395` Karrah Erwin, P.A. | 3240 Lynn Ridge Drive, Unit 2C, Raleigh, NC
    - `S2-413903395` #karraherwin | 3240 LYNN RIDGE DRIVE, RLAEIGH CDP, NC
    - `S2-709523128` KARRAH ERWIN, P.A. | 3240 LYNN RIDGE DR, RLAEIGH CDP, NC
    - `S3-820080251` Karrah Érwin, P.Á. | 240 Lynn Ridge Dr, Unit 2C, Raleigh, North Carolina

### India

**abbreviation**
- `S1-100543222` Lifewood Traders Pvt. Ltd. | 3Rd Floor Shankar Vihar, Vikas Marg Laxmi Nagar, East Delhi, Delhi
    - `S2-414828710` LIFEWOOD TRADERS PVT. LTD. | 3RD FLOOR SHANKAR VIHAR, VIKAS MARG LAXMI NAGAR, EAST DELHI, Delhi
    - `S2-419360730` M/s Lifewood Tidaemrs Pvt. Ltd. | 3RD FLOOR SHANKAR VIHAR, VIKAS MARG LAXMI NAGAR, EAST DELHI, दिल्ली
    - `S3-922350996` Lifewood Pvt. Ltd. [Center] | 3Rd Floor Shankar Vihar, East Delhi, DL
    - `S3-46431466` Lifewood Traders Pvt. | 3Rd Floor Shankar Vihar, East Delhi, DL, East Delhi, Vikas Marg Laxmi Nagar
- `S1-100788312` Nashik Event Private Limited | Nashik, Maharashtra, Pl 13, Dnyaneshwar Soc., Adgaon, Nashik
    - `S3-158038307` Nashik Event (Private) | Nashik, MH, Dnyaneshwar Soc., Adgaon, Nashik, Pl 13, Nashik

**landmark**
- `S1-102369978` Sunrise Products Limited | Siddhivinayak, Siddharth Nagar, Opp. Vivek Cl. Goregaon, (W), 20/307, Mumbai, Maharashtra, Mumbai City
    - `S2-318911490` सनराइज प्रोडक्ट्स लिमिटेड | 20/307/8, MUMBAI CITY, MUMBAI, Maharashtra
    - `S3-483089119` Sunrise-Products Limited | Siddhivinayak, Siddharth Nagar, Opp. Vivek Cl. Goregaon, (W), MH, Thane, Mumbai City, 20/30
- `S1-102463632` Silver Infotech Private Limited | Mittal Complex, Opp Anandaloke Nursing Home Near Ddts Petrol Pump, Sevoke Road, Sil, Iguri, Siliguri, Darjeeling, West Bengal
    - `S2-152799064` সিলভার ইনফোটেক প্রাইভেট লিমিটেড | MITTAL COMPLEX, OPP ANANDALOKE NURSING HOME NEAR DDTS PETROL PUMP, SEVOKE ROAD, SIL, IGURI, SILIGURI, KOLKATA, পশ্চিমবঙ্গ
    - `S3-56843070` Silver Infotech Private Ltd Enterprises | 

**missing_parts**
- `S1-100104589` Krsna Vidyalaya | D/101 Panchvati 1Raheja Township Malad East, Mumbai, Maharashtra
    - `S2-391790274` Dr Krsna Vidyalaya | D/101 PANCHVATI 1RAHEJA TOWNSHIP MALAD EAST, MUMBAI, Maharashtra
    - `S3-965842029` Krsna Vidyalaya | D/101 Panchvati 1Raheja Township Malad East, Mumbai, महाराष्ट्र
    - `S3-960759055` Krsna Vidyaalya | 
    - `S3-392887768` Krsna Vidcoalay | D/101 Panchvati 1Raheja Township Malad East, Mumbai, MH
    - `S3-179490859` Yumaumbra t/a Krsna Vidyalaya | D/101 Panchvati 1Raheja Township Malad East, Mumbai, Maharashtra
- `S1-100467453` Pear Traders Private Limited | Plot No.175-B, Vasanthanarasapura Industrial Area 1St Phase Negenapalli, Kora Hobli, Tumkur, Karnataka
    - `S3-716137607` Mr Pear Traders  Limited Private | 
    - `S3-443392436` Pear Traders Private Ltd | Vasanthanarasapura Industrial Area 1St Phase Negenapalli, KA, Kora Hobli, Plt No.175-b, Tumkur

**dba_website**
- `S1-100510264` De Health Clinic | 101 Shyam Rang Aptsjawahar Nagar Sundarvan Society Vasna, Ahmedabad, Gujarat
    - `S2-361840404` de health clinic | DOOR NO 101 SHYAM RANG APTSJAWAHAR NAGAR SUNDARVAN SOCIETY VASNA, AHMEDABAD, Gujarat
    - `S2-934070568` De Ceiic Health | NO 101 SHYAM RANG APTSJAWAHAR NAGAR SUNDARVAN SOCIETY VASNA, AHMEDABAD, Gujarat
    - `S3-91389305` Sri dehealthclinic.com | Door No 101 Shyam Rang Aptsjawahar Nagar Sundarvan Society Vasna, Ahmedabad, ગુજરાત
    - `S3-638374328` De Health Center | H.no 101 Shyam Rang Aptsjawahar Nagar Sundarvan Society Vasna, Ahmedabad, ગુજરાત
    - `S3-96937464` De Health Clinic Ltd | No 101 Shyam Rang Aptsjawahar Nagar Sundarvan Society Vasna, Ahmedabad, ગુજરાત
- `S1-100597662` Palm Group of Companies | D.No.38-39-7, 104 Area, Bapuji Nagar Industrial Estate Post, Visakhapatnam, Vishakhapatnam, Andhra Pradesh
    - `S2-993056533` Palm Companies of  Group | D.NO.38-39-7, 104 AREA, BAPUJI NAGAR INDUSTRIAL ESTATE POST, VISAKHAPATNAM, Andhra Pradesh
    - `S2-785853219` Smt Palm Group of Companies Corporation | DOOR NO 38-39-7 , 104 AREA, BAPUJI NAGAR INDUSTRIAL ESTATE POST, VISAKHAPATNAM, Andhra Pradesh
    - `S2-920006165` Palm Group of-Companies | #38-39-7 , 104 AREA, BAPUJI NAGAR INDUSTRIAL ESTATE POST, VISAKHAPATNAM, Andhra Pradesh
    - `S2-308844222` Palm Group of Companies (Company) | 
    - `S2-261675431` PALM GROUP OF | D.NO.38-39-7, 104 AREA, BAPUJI NAGAR INDUSTRIAL ESTATE PBST, VISAKHAPATNAM, ఆంధ్రప్రదేశ్
    - `S3-438238548` @palmcompanies | No #38-39-7, 104 Area, Bapuji Nagar Industrial Estate Post, Visakhapatnam, Vishakhapatnam, AP
    - `S3-800736541` palmgroupcompanies.com | Door No #38-39-7, 104 Area, Bapuji Nagar Industrial Estate Post, Vishakhapatnam, ఆంధ్రప్రదేశ్
    - `S3-408790700` Smt Palm Group of Companies Company | D.no.#38-39-7, Visakhapatnam, Vishakhapatnam, AP
    - `S3-817609410` companiespalm.com | D.no.#38-39-7, 104 Area, Bapuji Nagar Industrial Estate Post, Visakhapatnam, Vishakhapatnam, ఆంధ్రప్రదేశ్
    - `S3-808707342` Palm | D.No.38-39-7, 104 Area, Bapuji Nagar Industrial Estate Post, Visakhapatnam, Vishakhapatnam, Andhra Pradesh

**transliteration**
- `S1-100557121` High Anand Infra Pvt Ltd | Gobindpura, New Grain Market, Sangrur, Punjab
    - `S2-223306631` ਹਾਈ ਅਨੰਦ ਇਨਫ੍ਰਾ ਪ੍ਰਾ. ਲਿ. | GOBINDPURA, SANGRUR, Punjab
    - `S2-85310664` HIGH AANND INFRA PVT LTD | ਪੰਜਾਬ, GOBINDPURA, NEW GRAIN MARKET, SANGRUR
    - `S2-627975593` highanandinfra.com | GOBINDPURA, NEW GRAIN MARKET, SANGRUR, Punjab
- `S1-100890437` First Innovative Infrastructure Private Limited | C-43, Third Floor, Jhilmil Colony Shahdra, Delhi, East Delhi, Delhi
    - `S2-98633355` Limited First Innovative Infrastructure Service | DELHI, EAST DELHI, THIRD FLOOR, JHILMIL COLONY SHAHDRA, C-43, दिल्ली
    - `S3-979727144` First Innovative Ínfrastructure Private Limited | #948 C-43, Delhi, East Delhi, DL
    - `S3-957962434` First इनोवेटिव Infrastructure Private Limited | H.no 948 C-43, Third Floor, Jhilmil Colony Shahdra, Delhi, East Delhi, DL
    - `S3-213980283` First Innovative Infrastructure Private | 948 C-43, Third Floor, Jhilmil Colony Shahdra, Delhi, East Delhi, दिल्ली
    - `S3-405595629` फर्स्ट इनोवेटिव इंफ्रास्ट्रक्चर प्राइवेट लिमिटेड | Door No 948 C-43, Third Floor, Jhilmil Colony Shahdra, Delhi, East Delhi, दिल्ली

## 10. Surprising / notable findings

- **S1**: 0 rows with both fields empty; 1/69 rows with control chars in name/address; name len max=105 (p99=42), address len max=256 (p99=124); 0 (0.0%) names look like a URL/website (`.com`/`www`); 0 names are all digits/punct (phone-like); 14 (0.0%) addresses contain a literal `NULL`/`<NULL>` placeholder token (on top of the genuinely-empty ones in Section 2); 0 names contain a literal `|` (trade-name and website concatenated in one field); country labels seen: ['US', 'India'].
    longest name: `164/B-43 Vinayaka.Plaza 2Nd Floor 24Th Cross Ro Ad 6Th Block Jayanagar Bangalore 560082 Breweries Pvt Ltd`
- **S2**: 0 rows with both fields empty; 3/118 rows with control chars in name/address; name len max=104 (p99=48), address len max=249 (p99=118); 201,266 (4.0%) names look like a URL/website (`.com`/`www`); 9 names are all digits/punct (phone-like); 131,729 (2.6%) addresses contain a literal `NULL`/`<NULL>` placeholder token (on top of the genuinely-empty ones in Section 2); 15,592 names contain a literal `|` (trade-name and website concatenated in one field); country labels seen: ['India', 'US'].
    example website-as-name: `SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com`
    example `|`-joined name: `SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com`
    longest name: `LTD. 303&304 DABOUR BUILDING C-I LOCAL SHOPPING CENTRE IIND FLOOR JANAK ELECTRONICS OVERSEAS PVT. CENTER`
- **S3**: 0 rows with both fields empty; 1/124 rows with control chars in name/address; name len max=123 (p99=50), address len max=240 (p99=115); 210,922 (4.0%) names look like a URL/website (`.com`/`www`); 15 names are all digits/punct (phone-like); 130,786 (2.5%) addresses contain a literal `NULL`/`<NULL>` placeholder token (on top of the genuinely-empty ones in Section 2); 16,526 names contain a literal `|` (trade-name and website concatenated in one field); country labels seen: ['US', 'India'].
    example website-as-name: `wilfordhancock.com`
    example `|`-joined name: `M/s Basera Advisors  Group | www.baseraadv.com`
    longest name: `164/B-43 Vinayaka.Plaza 2Nd Floor 24Th Cross Ad Ro 6Th Block Jayanagar Bangalore 560082 560082 Breweries Industries Pvt Ltd`

- Country labels across all 3 sources are exactly `{India, US}` -- no variants (no "USA"/"United States"/"IN" seen in train). Test adds `France` per the problem PDF.
- GT `matched_entity_ids` reference ids only from S2/S3 that exist in their source files, and only from S1 ids that exist in source1 (checked in sections 1-4 above); no orphan ids, no self-references to S1 observed.

---
## Implications for blocking / matching

- Blocking must be **country-scoped**: cross-country pairs share almost no signal (Section 6), and 27%/25% of S2/S3 are pure distractors even within-country, so the matcher needs a real accept/reject threshold, not just top-k nearest neighbor.
- A cheap **postal-code-equality** blocking key is very high precision when present (94% of TRUE pairs share a postal number vs 0% of RANDOM pairs) but only fires for ~5% of true pairs -- use it as a strong feature/early-accept, not the sole blocking key.
- **Name-token overlap** is the workhorse blocking signal (86% TRUE vs 16% RANDOM) -- token/n-gram inverted-index blocking on normalized name tokens should get most of the way to a good candidate set.
- Exact normalized-name match only covers 22% of true pairs -- fuzzy matching (edit distance / Jaccard / TF-IDF cosine) is required on top of exact-match blocking, matching the PDF's own tip.
- India needs a **transliteration-aware** pipeline: 18% of India true pairs cross Devanagari/Latin scripts (S2 in particular is 23-24% Devanagari) -- normalize/transliterate before string comparison rather than relying on raw string similarity.
- F_0.5's 2x precision weighting plus a 5.6% singleton rate means **abstaining is cheap, false merges are expensive**: default to "no match" under low confidence (empty-submission floor is already 5.6%), and get precision right before chasing recall -- perfect precision with just ONE correct match per non-singleton already reaches 70%.
- IDs/row order carry no leakage (Section 7) -- no shortcut features from id numerics or file ordering; features must come from name/address/country text.
- Clean literal `NULL`/`<NULL>` placeholder tokens (~2.5% of S2/S3 addresses) and `|`-joined name/website fields (~0.3%) to genuine empty/split strings before feature extraction, or they will pollute token- and length-based features.
