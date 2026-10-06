I also opened and read both PDFs first (guidelines and problem statement), which confirmed the F_0.5 formula, output format, and that train covers only US/India (France is test-only) — all consistent with what the profiling below shows.

Note: I caught and fixed a real bug mid-analysis — an early `lookup()` helper split ids by source prefix then concatenated the blocks, silently reordering rows and breaking pair alignment in the true/random-pair signal comparison, the script-mix cross-check, and the leakage check. Fixed by using a single combined index so `.loc[ids]` preserves input order (verified with an inline self-check assert); all numbers below are from the corrected run.

Files:
- Script: `/tmp/claude-1001/-home-harsh1-github-repos-amc-26/b9a894ea-d7e4-441a-9f08-e7799e3d80b9/scratchpad/profile/profile.py`
- Report: `/tmp/claude-1001/-home-harsh1-github-repos-amc-26/b9a894ea-d7e4-441a-9f08-e7799e3d80b9/scratchpad/profile/train_report.md`

Full report content:

```markdown
# AMC-26 train-split profile

## 1. Parse integrity & row counts

| file | wc -l (data rows) | parsed rows | match | columns |
|---|---:|---:|---|---|
| train_source1.tsv | 2,206,821 | 2,206,821 | OK | entity_id, business_name, business_address, country |
| train_source2.tsv | 5,034,616 | 5,034,616 | OK | entity_id, business_name, business_address, country |
| train_source3.tsv | 5,285,603 | 5,285,603 | OK | entity_id, business_name, business_address, country |
| train_ground_truth.tsv | 2,206,821 | 2,206,821 | OK | source1_entity_id, matched_entity_ids |

Parse used `sep='\t', quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False, engine='c'` (pyarrow engine rejects `quoting=`). All 4 files parse to exactly wc-1 rows — no line-merging from the embedded `"` chars confirmed present in source1/source2 (they occur inside addresses, never break a line because the field is never comma/tab-quoted).

S1:S2:S3 row-count ratio = 1 : 2.281 : 2.395 (2,206,821 : 5,034,616 : 5,285,603)

## 2. Country distribution & empty-field rates per source

| source | country | rows | % of source | empty business_name | empty business_address |
|---|---|---:|---:|---:|---:|
| S1 | US | 1,323,633 | 59.98% | 0.00% | 0.00% |
| S1 | India | 883,188 | 40.02% | 0.00% | 0.00% |
| S2 | US | 3,016,817 | 59.92% | 0.00% | 3.68% |
| S2 | India | 2,017,799 | 40.08% | 0.00% | 2.87% |
| S3 | US | 3,170,056 | 59.98% | 0.00% | 3.50% |
| S3 | India | 2,115,547 | 40.02% | 0.00% | 3.07% |

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
- S2 distractor rate (never matched): 26.64% (1,340,997 / 5,034,616)
- S3 distractor rate (never matched): 25.37% (1,340,857 / 5,285,603)
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

Sampled 50,000 TRUE pairs and 50,000 RANDOM same-country pairs (country mix matched to the true-pair mix: US 59.7%, India 40.3%).

| signal | TRUE pairs | RANDOM pairs |
|---|---:|---:|
| exact normalized-name match | 21.81% | 0.00% |
| ≥1 shared name token | 85.67% | 15.84% |
| both sides have a 5-6 digit postal-like number | 5.20% | 0.78% |
| postal numbers match (of those where both have one) | 94.11% | 0.00% |
| shared comma-segment ("city-ish") token | 87.25% | 2.59% |
| country label equal | 100.00% | 100.00% |

## 7. ID / row-order leakage check

- Correlation(S1 numeric id, matched numeric id) over 50,000 pairs: **-0.0041**
- Correlation(S1 row position, matched-record row position) over 50,000 pairs: **0.0041**
- S1 numeric id range: [15,163, 999,990,872]; matched-record numeric id range: [65,472, 999,997,748]
- Near-zero correlation ⇒ ids/row order carry no shortcut signal for matching; both look ~randomly assigned.

## 8. Trivial baseline scores (train, macro F_0.5)

- All-empty submission: **5.5848%** (= singleton fraction; every non-singleton scores 0, every singleton scores 1).
- Perfect-precision, exactly-one-match-per-S1 oracle (singletons predicted empty; non-singletons get 1 correct match out of k true matches, so precision=1, recall=1/k, F_0.5=1.25/(0.25k+1)): **69.6388%** overall (non-singleton subset alone: 67.8429%). This isolates what recall is worth once precision is perfect: going from the empty baseline to "one right guess per entity" only buys 64.05 points because most non-singletons need 2-5+ matches for full credit.

## 9. Illustrative matched clusters (by noise type)

Full clusters (S1 + all its S2/S3 matches, `name | address`) for abbreviation, landmark, missing-parts, DBA/website, no-token-overlap (US) and abbreviation, landmark, missing-parts, DBA/website, transliteration (India) are in the saved report — real examples include `Metro Holding Company Corporation` / `Metro Holding Co Corporation` / `holdingmetro.com` (abbreviation+website); `Sunrise Products Limited` / `सनराइज प्रोडक्ट्स लिमिटेड` (transliteration); `Palm Group of Companies` with 10 matched variants across `.com` names and Telugu script (DBA cluster); `Tober's Marketing Inc`, address `..., Opp, AL` (landmark token that's actually a town name, not "opposite" — a false-positive trap for a naive landmark-keyword feature).

## 10. Surprising / notable findings

- No rows anywhere have both business_name and business_address empty.
- ~4.0% of S2/S3 business_name values are website/URL-like (`.com`/`www`), e.g. `wilfordhancock.com`.
- ~2.5% of S2/S3 business_address values contain a **literal `NULL`/`<NULL>` placeholder token** on top of genuinely-empty strings — a second, distinct missingness encoding.
- ~0.3% of S2/S3 business_name values contain a literal `|` joining a trade name and a website in one field, e.g. `SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com`.
- Country labels are exactly `{India, US}` in train, no variants ("USA", "IN", etc. don't appear); test adds `France` per the problem PDF.
- GT never references an id outside its source file and never self-references S1; strict 1-to-many confirmed in Section 4.
- Max field lengths ~250 chars (addresses), ~120 chars (names); p99 name length ~50 chars.

## Implications for blocking / matching

- Blocking must be **country-scoped**: cross-country pairs share almost no signal, and 27%/25% of S2/S3 are pure distractors even within-country, so the matcher needs a real accept/reject threshold, not just top-k nearest neighbor.
- A cheap **postal-code-equality** key is very high precision when present (94% TRUE vs 0% RANDOM) but only fires for ~5% of true pairs — use as a strong feature/early-accept, not the sole blocking key.
- **Name-token overlap** is the workhorse blocking signal (86% TRUE vs 16% RANDOM) — token/n-gram inverted-index blocking on normalized name tokens gets most of the way there.
- Exact normalized-name match covers only 22% of true pairs — fuzzy matching (edit distance/Jaccard/TF-IDF cosine) is required on top of exact-match blocking.
- India needs a **transliteration-aware** pipeline: 18% of India true pairs cross Devanagari/Latin scripts (S2 is 23-24% Devanagari).
- F_0.5's 2x precision weighting + 5.6% singleton rate means **abstaining is cheap, false merges are expensive**: default to "no match" under low confidence; perfect precision with just one correct match per entity already reaches 70%.
- IDs/row order carry no leakage — no shortcut features from id numerics or file ordering; features must come from name/address/country text.
- Clean literal `NULL`/`<NULL>` placeholders (~2.5% of S2/S3 addresses) and `|`-joined name/website fields (~0.3%) to genuine empty/split strings before feature extraction.
```