# P1-taxonomy — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
normalize.py v1 alone, as exact keys (country, name_core) + an address check, scores **F0.5 0.7831** on dev at test-density
S1 chains (0.7849 on holdout with the v1.1 keys; s0 scores 0.1775), with 99.9% pair precision. Exact-name agreement on true pairs goes 21.9% → 61.4%; random pairs stay at 0.00%.
What's left, by recall-side F0.5 at stake: cross-script 3.5 pts > website 1.6 > typo 1.1 > partial overlap 0.9 > no-shared-token 0.5.
Adopt `normalize.py` as the shared normaliser; use `address_fields` (house no. + city + state) for every accept decision, because 50% of S1 are name look-alikes.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
1. Promote `probes/P1-taxonomy/normalize.py` to `src/normalize.py` unchanged (self-check: `python normalize.py`).
   - Names: `name_core(names, country)` is the matching key. `name_norm` is the display/TF-IDF form (legal words kept).
     Use `acronym(name_core(...))`, plus `is_website` / `website_core` for the 5.4% of S2/S3 names that are sites or handles.
   - Addresses: call `address_fields(addr, country, freq)` once. It returns
     `addr_norm, state, places, city_key, postcode, house_number`. The single-field wrappers each re-parse the address.
   - `freq = place_freq(S1 addresses)`. Fit it on the S1 file being matched (train S1 for worlds, test S1 for test —
     France cities are not in train). Without `freq`, city_key falls back to the last place segment (0.729 vs 0.792 agreement).
2. **Baseline to beat / first-stage rule** (`exact_match.py`, variant D): same (country, name_core), or S2/S3 website_core ==
   squashed S1 core. Accept the S1 with the unique best `4·[house_no eq] + 2·[city eq] + [place overlap] + [state eq]`, provided it is ≥ 2;
   skip ties; each S2/S3 goes to at most one S1. This is high precision (0.999 pair precision at test density), so later
   stages only have to add recall.
3. Evaluate every name-only rule with all train S1 loaded as distractors (`exact_match.py dev dense`). The dev/holdout worlds fill
   their S2/S3 with records whose own S1 is missing, so they hide chain collisions. Name-only unique-key matching scores
   0.684 on dev but **0.431** at test density.
4. Features the matcher (P4) should take from here: name_core exact / Jaccard / any-token, house_number eq, city_key eq,
   place overlap exact and fuzzy (rapidfuzz ratio ≥ 85), state eq, website_core-in-squashed-core. **Drop postcode**:
   it is present in 0.07% of true pairs, because US 5-digit numbers are house numbers, not ZIPs.
5. name_core already strips the generator's noise words `center/centre/services/service` (v1.2). That moved dense F0.5
   0.7748 → 0.7831 at unchanged precision.
6. Cheap next wins, in order of stake: (a) P2 transliteration before name_core (3.5 pts). (b) Websites: match website_core
   by substring/prefix of the squashed S1 core, not just equality. It is a substring or 8-char prefix in 85.6% of website pairs
   (1.6 pts; D uses equality only). (c) Token-level fuzzy name similarity for typos and heavy scrambles (1.1 + 0.9 pts).

## Evidence         (table: ledger name | world | key metrics | verdict)
Sample = 30k train-world S1 with full clusters (103,906 true pairs) + 100k random same-country pairs. "Dense" = dev world + all
2.2M train S1 as distractors (≈ test's 1.73M S1).

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| lift-name-addr-v1.2 (final; v1 / v1.1 rows are earlier) | train-sample30k | true vs random. Name exact: s0 0.219/0 → norm 0.315/0 → core **0.614/0**. Any shared token: s0 0.855/0.157 → core 0.866/**0.007**. Jaccard ≥ .5: s0 0.770 → core 0.794 (random 0.0002). addr seg0 exact (s0) 0.322 → addr_norm exact 0.430 (random 0). Place overlap 0.856/0.012, fuzzy85 0.895/0.014. city_key (freq) **0.792/0.011**, city_key (last) 0.729/0.008. State eq (both present) 0.9995/0.071, present in 95.5% (v1: 90.4%). House no. eq 0.867/0.004, present in 76%. Postcode present 0.07% | keep |
| taxonomy-v1 (second row = v1.2 keys) | train-sample30k | class shares and stakes below | keep |
| lookalike-chains-v1 | train S1 2.2M + sample | 50.6% of train S1 share their core with another S1 of the same country (test 48.8%, **France 54.5%**, India 54%, US 40%). 49% of exact-core true pairs have a wrong same-core S1. Wrong vs true S1: house no. eq 0.3% vs 66.6%, city 3.1% vs 78.5%, place 3.2% vs 84.9%, state 8.7% vs 94.6%. Address score picks the true S1 strictly in 89.9%, ties 10.0%, wrong 0.18% | keep |
| address-structure-v1 | train sample + test 1/40 | table below | keep |
| exact-keys-A_s0 | holdout | F0.5 0.1775 (reproduces s0) | — |
| exact-keys-B_core_unique | dev / holdout / dev-dense | 0.684 / 0.576 / **0.431** (name-only collapses at chain density) | drop as a rule |
| exact-keys-C_core+addr | dev / holdout / dev-dense | 0.772 / 0.761 / **0.751**. Pair precision 0.996 / 0.983 / 0.999 | keep |
| exact-keys-D_core+addr+web | dev / holdout / dev-dense | 0.797 / 0.785 / 0.775. Pair recall 0.575 at dense | keep |
| exact-keys-D_core+addr+web-dense-generic | dev-dense | **0.7831** (US 0.825, India 0.720). Pair precision 0.9992, recall 0.589, singleton acc 0.998 (C: 0.7600) | keep → adopted in v1.2 |
| mine-subs-city-alias-v1 | train 60k S1 (disjoint from sample) | 553 US city aliases (India 1). Place agreement true 0.855 → 0.867, random 0.0115 → 0.0117 | inconclusive (small, clean gain) |
| core-minus-generic-words | sample + train S1 | exact core 0.598 → 0.614, S1 chain share 0.506 → 0.514 | "drop" by a proxy rule; overruled by the dense F0.5 row above |

**Difficulty taxonomy** (first match wins, % of true pairs). Stakes are recall-side F0.5 points lost if the naive matcher
misses the class. M1 = Jaccard ≥ .5 on s0-normalised names; M2 = the same on name_core.

| class | all | India | US | S2 | S3 | India/S2 | India/S3 | US/S2 | US/S3 | miss M1 | miss M2 | ceiling | stake M2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_script | 6.72 | 16.66 | 0 | 8.85 | 4.73 | 21.96 | 11.71 | 0 | 0 | 0.99 | 0.99 | 3.51 | **3.45** |
| website / handle | 5.35 | 4.58 | 5.87 | 5.47 | 5.24 | 4.61 | 4.55 | 6.05 | 5.70 | 0.99 | 1.00 | 1.57 | **1.57** |
| typo (Levenshtein ≤ 2 on core) | 5.14 | 4.98 | 5.24 | 5.35 | 4.94 | 5.22 | 4.75 | 5.44 | 5.07 | 0.36 | 0.62 | 1.67 | **1.08** |
| partial overlap (≥1 shared core token: heavy typo/scramble, word swap) | 5.76 | 4.50 | 6.61 | 5.80 | 5.72 | 4.40 | 4.59 | 6.74 | 6.49 | 0.20 | 0.55 | 1.72 | **0.94** |
| no shared token, address agrees | 1.81 | 1.97 | 1.70 | 1.45 | 2.14 | 1.11 | 2.77 | 1.69 | 1.71 | 0.99 | 1.00 | 0.49 | 0.49 |
| token subset (word dropped / generic word added) | 11.18 | 7.81 | 13.46 | 10.48 | 11.84 | 6.72 | 8.82 | 13.02 | 13.88 | 0.18 | 0.02 | 3.94 | 0.06 |
| acronym (one side = initials of the other) | 0.18 | 0.28 | 0.11 | 0.05 | 0.30 | 0.01 | 0.53 | 0.07 | 0.14 | 0.95 | 0.95 | 0.05 | 0.05 |
| different surface form (DBA/fake name, no address help) | 0.04 | 0.01 | 0.05 | 0.02 | 0.05 | 0 | 0.02 | 0.04 | 0.07 | 0.95 | 1.00 | 0.01 | 0.01 |
| pipe-joined residual | 0.08 | 0.06 | 0.09 | 0.09 | 0.07 | 0.07 | 0.04 | 0.10 | 0.08 | 0.95 | 0.44 | 0.03 | 0.01 |
| word reorder (sorted tokens, ± typo) | 2.35 | 0.86 | 3.35 | 2.44 | 2.26 | 0.90 | 0.83 | 3.48 | 3.23 | 0.01 | 0.00 | 0.74 | 0.00 |
| exact after normalisation (case/accents/punct/alias/pipe/ID) | 31.54 | 22.72 | 37.50 | 28.79 | 34.10 | 19.26 | 25.95 | 35.22 | 39.63 | 0.07 | 0 | 13.55 | 0 |
| legal-suffix / honorific / generic-word only | 29.87 | 35.58 | 26.00 | 31.21 | 28.62 | 35.73 | 35.45 | 28.15 | 23.99 | 0.06 | 0 | 12.51 | 0 |

"ceiling" = F0.5 points lost if the whole class were missed. "stake M2" = points lost to the name_core matcher's misses
(stake = share × miss rate, turned into points by the exact metric on the sample's 30k S1, singletons included). The "partial overlap" and
"acronym" classes are my additions to the brief's list; nothing fits between them and "no shared token". Recall-only F0.5 of the
matchers as the whole system: s0 exact 0.432, M1 0.899, M2 0.911 (US 0.956, India 0.845). M1 also shares a token with
15.7% of random pairs; M2 with 0.7%.

Raw noise prevalence on the S2/S3 side of true pairs: brackets 8.1%, accents injected 6.7%, website/handle 5.8%,
Indic 7.4% (India S2 24%, S3 13%), empty address 4.5%, NULL address 2.5%, digit-for-letter 3.0%, alias constructs 2.0%
(**S3 only**: 2.9% India, 4.4% US; never S2), `|` join 0.3%, `(ID: n)` tail 0.3%, junk prefix 0.9%.
website_core is a substring / 8-char prefix of the squashed S1 core in 85.6% of website pairs.

**Address structure** (% of rows; train sample, test every 40th row):
- Segments per address: India 5.7, US 3.2, France 3.0. Order is shuffled in **every** source, S1 included.
  The state segment is last in ~82–87% of rows, first in 4–8%, middle in the rest.
- State form: US S1 codes (99.9%), US S2 codes (95%), US S3 full names (90%). India S1 full names (100%);
  India S2 full 73% + Indic script 24%; India S3 codes 65% + Indic 22%. Indic in addresses is **only** the state name
  (16 strings, all mapped). France: région/département in 64–66% of S2/S3; **none at all in ~35% of France S2/S3**.
- State codes India (mined): MH DL UP KA TN WB GJ TG HR RJ KL BR MP AP PB OD. Telangana ↔ Andhra Pradesh shows up 370× in
  true pairs, so both map to `ap`.
- Postcodes are near-absent everywhere: PIN 0.2–0.6% (India), France 0.16%, US 0.
- House number in the first segment: US 86% (S1), India 50%. Landmarks (near/opp/behind…): India 13% (S1) / 11% / 8%; US and France 0.
  "Opp" as a whole segment is a town (US/India 0.01–0.05%); the landmark rule requires words after the marker.
- Noise only in S2/S3: NULL/`<NULL>`/N/A 2.7–3.8% (0% in France S2/S3, 3% France empty), PO BOX/PMB 3% US S2/S3 only (0 in S1), CITY/CDP/TWP/TOWNSHIP
  suffixes 9.5% US S2/S3 vs 6.4% US S1 (real "Kansas City"), `DIVREPORTINGCIRCLE`/`REGION`, alternate municipality
  names (township, CDP, neighbourhood for city), typos in city names (hence fuzzy place overlap: +4 pts).
- city_key rule: address segments → drop state (looked up **before** abbreviation, so "West Bengal" and "Kansas City" parse right),
  digit-bearing, street-typed (last token st/rd/ave/marg/rue…), landmark and C/O segments. Among the rest, pick the most frequent
  place in S1 (`freq`). Agreement: true 0.792 vs random 0.011 (US 0.793/0.001, India 0.791/0.026).

**Mined dictionaries** (`mined.json`, token aggregates only). Name swaps: inc↔incorporated, limited↔ltd (5.2k),
corp↔corporation, co↔company, pc/lp/group↔llc/inc. Honorifics added by the generator (sri, shri, smt, mr, dr, ms, the).
Generic words added far more often than dropped (center, services, service, partners, trading, labs). Digit-for-letter
(1→l, 0→o, 5→s, 8→b, 6→g), I→l (lnc/lndia). Legal-suffix inventory by S1 last token: US llc 28%, inc 18%, pc 3.6%, corp, pllc,
lp, co, corporation, company. India limited 56%, ltd 17%, llp 4.5%, co, company, corporation, corp; legal tokens are
dropped 30–44% of the time vs ~21–26% for content words. Address swaps: rd/dr/st/ave/ln/ct/cir/pl/blvd/trl/ter/hwy/pkwy/cv,
state code↔name, ordinal suffix noise (4th↔4rd), Atlanta↔ATL, Bombay/Calcutta/Madras/Poona/Bengaluru/Gurugram.
**French forms, street types and régions are domain knowledge (`# domain:` in the code), unvalidated.**

## What failed       (approaches tried and dropped, with the number that killed them)
- Name-only unique-key matching (B): 0.684 on dev, but 0.431 with test-density S1. Chains make the name alone unusable as a decision.
- Jaccard ≥ .5 on name_core as the only name test: stripping legal words shrinks the sets, so one typo fails it (partial-overlap
  misses 20% → 55%). It still wins overall (recall-only 0.911 vs 0.899) because random-pair collisions drop 20×. Pair it with fuzzy token scores.
- city_key = last place segment (0.729) loses to frequency-picked (0.792), because segments are shuffled.
- Postcode as a key or feature: present in 0.07% of true pairs.
- Mined city aliases: +1.2 pts place agreement. Real but small; kept in `mined.json`, not wired into normalize.py.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
`normalize.py` (pandas Series in → Series out; deps anyascii ISC, rapidfuzz MIT, both installed):
- `fold(s)`: lower ASCII (anyascii). `name_norm(names)`, `name_core(names, country=None)`, `acronym(core)`,
  `is_website(names) -> bool`, `website_core(names)`.
- `address_fields(addr, country=None, freq=None) -> DataFrame[addr_norm, state, places(list), city_key, postcode, house_number]`.
  Thin wrappers: `addr_norm`, `state`, `city_key`, `place_segments`, `postcode`, `house_number`. `place_freq(addr, country) -> dict`.
- Speed: ~130–150 µs/row single-threaded, all fields (name_core ~38, address_fields ~60–80). About 25–30 min for all 11.7M test rows
  on one core; the work is independent per row, so chunk it over processes (~2–3 min on 12). RAM ~0.5 GB per 200k-row chunk; keep
  `places` as a `|`-joined string when storing. `exact_match.py dev dense` peaks at 1.9 GB (it holds all 2.2M train S1 features).
Scripts (analysis only, not pipeline): `sample.py`, `featurize.py`, `analyze.py lift|taxonomy|placefreq|s1cores|lookalike|address`,
`exact_match.py [dev|holdout] [dense] [generic]`, `mine.py`. They cache in /tmp/p1.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Licences: anyascii ISC and rapidfuzz MIT. No external data; every dictionary is mined from train pairs or written from domain knowledge.
- Indic names get only anyascii's crude romanisation. P2's `to_latin` should run first (normalize is idempotent on Latin).
  Cross-script is still the #1 class at stake (3.5 pts).
- France: legal forms, street types, articles (de/la/du… in STOP) and the région/département map are unvalidated domain knowledge.
  France chains are the densest (54.5% of S1 share a core), and 35% of France S2/S3 carry no région. So France accepts must
  lean on house number + city, which the D rule already requires.
- `freq` fitted on test S1 text is an unsupervised statistic on test data (open question 4). Fallback without `freq` costs ~6 pts of city agreement.
- Heuristics that can misfire: "nee" as an alias marker (17 of 2.2M S1 contain it as a word); the digit→letter fold on mixed tokens; Washington/DC
  and Delhi state-vs-city ambiguity (the code-form segment wins); the "ln"→"in" rule (names only); AP/TG merged into one code.
- The eval worlds understate chain collisions (see Recommendation 3). All numbers here are on sampled worlds, not full train.
- `mined.json` holds dataset-derived tokens (city-alias pairs, token swaps) — aggregates, no rows. Treat it like a model artifact
  when deciding what to commit.

## Open questions   (for the master; each with your best guess)
1. Fit `place_freq` on test S1? Best guess: yes. It is S1 text only, needed for France; ask via the form if it can be spared.
2. Wire the 553 mined US city aliases (`mined.json`) into city matching? Best guess: yes, as a feature. +1.2 pts place agreement,
   random unchanged, but it is only valid for US/India cities seen in train.
3. Rebuild dev/holdout with distractor S1s for all fill-in S2/S3? Best guess: yes, or always score with `dense`.
4. Tie cases (10% of look-alike situations: no address evidence on either side). Best guess: abstain. F0.5 punishes a wrong pick more than a miss.
