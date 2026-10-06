# AMC-26 test-split profile

**Files:**
- Script: `/tmp/claude-1001/-home-harsh1-github-repos-amc-26/b9a894ea-d7e4-441a-9f08-e7799e3d80b9/scratchpad/profile/profile_test.py`
- Report: `/tmp/claude-1001/-home-harsh1-github-repos-amc-26/b9a894ea-d7e4-441a-9f08-e7799e3d80b9/scratchpad/profile/test_report.md`

Both PDFs opened first: `.../6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf` (confirms F_0.5 formula, output format `matching_results.tsv`/`candidate_pairs.tsv`, and "test additionally contains a third country, France") and the guidelines PDF. Data was already extracted at `data/student_resource/dataset/test/`. Ran `.venv/bin/python`. Peak RSS ~5.5GB mid-run (transient, during hash-building before test frames were freed), settled to ~3.2GB by completion — under the 15GB box, exit 0, no swap thrashing observed.

One real bug hit and fixed mid-run: `.str.contains(regex=True)` on pandas 3's pyarrow-backed string columns runs on Arrow's RE2 engine, not Python's `re` — RE2 rejects `\uXXXX` escape syntax (`ArrowInvalid: invalid escape sequence: \u`), even inside a raw Python string where Python's own `re` would have accepted it. Fixed by building those patterns (apostrophe, CJK, Arabic ranges) as **non-raw** Python string literals so Python's lexer expands `\uXXXX` into the real character before either regex engine ever sees it.

## 1. Row counts (parsed vs wc -l)
| file | rows | match |
|---|---:|---|
| test_source1 | 1,732,544 | OK |
| test_source2 | 4,887,273 | OK |
| test_source3 | 5,082,316 | OK |

S1:S2:S3 = 1 : 2.821 : 2.933 (train was 1 : 2.281 : 2.395 — more S2/S3 candidates per S1 entity in test). Total = 11,702,133. **S1 test entity count = 1,732,544**, matching the validator docstring's "~1.7M-entity test set." S1 test is 21.5% smaller than S1 train (2,206,821).

## 2. Country distribution (exact labels: `['France', 'India', 'US']`)
| source | India | US | France |
|---|---:|---:|---:|
| S1 | 809,986 (46.75%) | 663,106 (38.27%) | 259,452 (14.98%) |
| S2 | 2,312,565 (47.32%) | 1,871,330 (38.29%) | 703,378 (14.39%) |
| S3 | 2,405,000 (47.32%) | 1,945,701 (38.28%) | 731,615 (14.40%) |

France ≈ 14.5-15% of every source, exactly one new label, no spelling variants. **Non-France mix shifted**: excluding France, test is ~45% US / 55% India vs train's ~60/40.

## 3. Empty rates (name always 0%; address, vs train)
| source/country | train | test |
|---|---:|---:|
| S2/US | 3.68% | 2.94% |
| S2/India | 2.87% | 2.28% |
| S3/US | 3.50% | 2.84% |
| S3/India | 3.07% | 2.46% |

France address-empty: S1 0%, S2 3.06%, S3 2.94% — in line with US/India rates.

## 4. France deep dive (15 samples/source in the saved report)
S1 patterns (n=259,452): rue/avenue/bd-type street token 94.5%; SARL/SAS/SASU/SA/EURL/SCI/EI legal form in name 68.9%; accented Latin char 38.6%; apostrophe 5.6%; 5-digit postcode only 0.4%; CEDEX 0.0%; **CJK/Arabic/Devanagari all 0.00%** — no non-Latin script anywhere in France. S2/S3 similar, legal-form ~47-49%. Visible typo/noise patterns mirror US/India: `Dëfi` (accent corruption), `fdration.com` (dropped letters), `PEIRRE`/`Aelee` (transpositions), `Ets Çommercants` (odd cedilla) — same noise generator, not a new distribution.

## 5. Script mix vs train
India Indic-script rates match train closely (S2 name 23.64% vs train 23.51%, S3 name 13.33% vs train 13.17%). France: 0% Indic/CJK/Arabic everywhere; non-ASCII (15-25%) is entirely French accents.

## 6. Pipeline-breaking checks — clean
Zero duplicate entity_ids, zero empty ids/rows, zero bad-prefix ids, zero embedded tabs/CR-LF, zero BOM, across all 3 files. Control chars: 72-183 rows/file (address only). Literal `"` in 132-346 rows/file (harmless — `QUOTE_NONE` parsing already confirmed exact row-count match, so it never corrupts parsing). Same two quirks as train carry over: literal `NULL`/`<NULL>` placeholder in ~2.1-2.2% of S2/S3 addresses; `|`-joined name/website in ~0.24% of S2/S3 names; ~3.2% URL-like names.

## 7. Compute sizing (country-scoped blocking)
| country | S1×(S2+S3) full join | @k=10 | @k=50 |
|---|---:|---:|---:|
| France | 372.3B | 2.59M | 12.97M |
| India | 3,821.2B | 8.10M | 40.50M |
| US | 2,531.1B | 6.63M | 33.16M |
| **total** | **6.72T** | **17.33M** | **86.63M** |

Full cross join is infeasible; k=10/50 blocking caps at 0.0003%/0.0013% of it — trivially scoreable once blocking is done.

## 8. Train/test leakage check
Zero entity_id overlap in any source (disjoint id spaces). Zero exact-duplicate records (hash of name+address+country) between train and test. **Test — France included — is genuinely unseen data.**

## Implications (6)
- **France has zero labeled examples anywhere** and zero overlap with train — any France threshold/weight must come from structural priors (token overlap, legal-form/address normalization), not train-calibrated precision, since there's nothing to calibrate against.
- **Country mix shifted, not just +1 label**: non-France test is ~45% US/55% India vs train's ~60/40 — a pipeline tuned on train's US-heavy mix will see proportionally more India (and its script-crossing pairs) than it trained against.
- **Postal-code blocking won't help France**: only 0.4-0.5% of France addresses carry a 5-digit code (vs the high-precision postal-match key train relied on) — needs a French-specific key (commune name + rue/avenue token match) instead.
- **Same noise generator, no new script-handling needed**: France's non-Latin-script rate is 0% everywhere; non-ASCII is pure accented-Latin, so existing NFKD/strip-accents normalization covers it — no new script-detection branch required.
- **Blocking quality remains the bottleneck, not pairwise compute**: k=10/50 caps candidates at 17.3M/86.6M out of a 6.72T full join — same conclusion as train, now confirmed at test scale.
- **Two pre-existing data quirks carry over unchanged**: literal `NULL`/`<NULL>` placeholders (~2.1-2.2% of S2/S3 addresses) and `|`-joined name/website strings (~0.24%) still need cleaning before feature extraction, at the same rates seen in train.