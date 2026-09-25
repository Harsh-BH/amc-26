# The 2026 problem — Business Entity Resolution

Released 25 Sep 2026 00:00 IST. The official statement is `data/student_resource/README.md`
(same text as the problem PDF). PDFs, the video transcript, the raw profiling reports and scripts
are in `data/notes/` — gitignored, because they quote Amazon's rows. This file is the digest:
every rule below was checked against the sources, every number against the data.

## Timeline (IST)

| | |
|---|---|
| Challenge window | **25 Sep 00:00:50 → 27 Sep 23:59:49** (Unstop API; guidelines PDF says 12:00 AM → 11:59 PM) |
| Submissions | **5 per day** for the 3 days, then the submit button is disabled (guidelines PDF) |
| Top 100 announced | they then submit methodology details (guidelines PDF) |
| Top 50 results | 2 Oct — Top 50 get PPIs for Applied Scientist Intern (Unstop round + blog) |
| Grand Finale | 7 Oct 10:00–15:00, top 10 present to Amazon scientists (virtual) |
| Code review | 8–9 Oct, hidden Unstop round: Amazon evaluates top teams' approach + code |

## The task

Three sources describe the same real-world businesses with no shared identifier.
**Source 1 (S1) is deduplicated** — each business appears once. For every S1 record, list the
S2/S3 records that are the same business: zero, one or many.

Every source file has 4 columns: `entity_id` (prefix `S1-`/`S2-`/`S3-`), `business_name`,
`business_address`, `country`. No source column — the prefix says it.
`train_ground_truth.tsv`: `source1_entity_id`, `matched_entity_ids` (comma list, empty = no match).

**Everything is TAB-separated.** Read with `sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
keep_default_na=False` — names contain literal `"`, and pandas' default NA parsing would eat
real values. With `QUOTE_NONE` every file parses to exactly `wc -l − 1` rows.

## Output — two files, both TSV

| File | Header | Scored? |
|---|---|---|
| `matching_results.tsv` | `source1_entity_id⇥matched_entity_ids` | **Yes** — the only leaderboard file |
| `candidate_pairs.tsv` | `source1_entity_id⇥candidate_entity_ids` | No — audited for blocking recall + reduction ratio |

Rules for both: exactly one row per test S1 (all **1,732,544**, France included); empty second
column = no match; IDs comma-separated, **no spaces, no quotes, no duplicates in a list, no
duplicate S1 rows**; only `S2-`/`S3-` IDs that exist in the test set. `candidate_pairs.tsv` must
be the **last** candidate set the model scores, and matches must be a subset of it.

Validate before every upload (from `data/student_resource/`):

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test   # add --check-ids: ~10M IDs in RAM
```

What the validator does **not** catch: whitespace around the S1 id (reported as missing +
unknown IDs), a UTF-8 BOM (cryptic header error), silently skips blank lines, **passes without
`candidate_pairs.tsv`** (still required in the zip), checks nothing in the zip, computes no score.
Header match is case-insensitive; CRLF and a missing final newline are fine.

## Metric — macro F0.5 per S1 entity

`F0.5 = 1.25·P·R / (0.25·P + R)` per S1, then the mean over **all** S1 — singletons included.

- True singleton: empty prediction = **1.0**, any prediction = **0.0**.
- Has matches, you predict empty: **0.0** (recall 0) — no partial credit.
- PDF example: predict 3, 2 correct, 2 true → P=2/3, R=1 → 0.714.
- Precision weighs 2× recall: 1 right + 1 wrong on 2 true = 0.50; 1 right, 0 wrong on 5 true = 0.56.

Reference points on train: all-empty = **0.056**; perfect precision with exactly one correct
match per S1 = **0.696**. The last 30 points are recall on multi-match clusters.

Leaderboards: public = a subset of test, private = the rest, you always submit the full test set.
PDF says final rank = private; guidelines say shortlisting uses both (open question below).

## Submitting

**During the challenge** the Unstop play page has two upload slots — matching results (TSV)
and code (zip) — under one *Submit & Evaluate* button. A valid file shows status `SCORED` with
the F0.5; *"submissions that fail validation will not be evaluated."* Keep a version history of
every submission; final code may be requested later.

**Final package**, every team:

```
<team_name>_submission.zip
├── output/matching_results.tsv, candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/                 all source
│   ├── README.md            exact end-to-end run instructions (data → blocking → matching → output)
│   └── requirements.txt     pinned versions
└── Documentation_template.md   filled in (.md or .pdf), not renamed
```

The doc must cover methodology, blocking strategy, model architecture + features, anything
else relevant. **No page limit** (problem PDF + play page — more specific than the guidelines' "1–2 pages",
and it is what the submission page shows). Guidelines also ask for experiments, conclusions, and code comments.

## Rules that disqualify

1. **Final model: MIT or Apache-2.0 licence, ≤ 8B parameters.** Qwen / BGE-M3 / gte /
   MiniLM qualify; Llama and Gemma licences do not. Record each model's licence + size in the doc.
2. **No external lookup:** no ER APIs, business registries, **geocoding APIs**, or internet data
   augmentation. Hosted LLM/embedding APIs (Bedrock, OpenAI, Claude) are out — they fail rule 1
   and put an external service in the resolution path. Pretrained weights are fine (rule 1 only
   makes sense if they are); download them before relying on them.
3. **One Unstop ID per person, one login at a time.** Multiple IDs = instant disqualification;
   simultaneous logins "may terminate the ML Challenge altogether". Desktop/laptop only; the live
   play page also showed "use a screen bigger than 1024px" and a zoom warning when the window was narrow.
4. **Country is an open label set.** Never filter or one-hot on `{US, India}`; a missing France
   row gets the file rejected.
5. No negative marking, but eligibility, authenticity and final judgement rest solely with Unstop and
   the organizer (play page guidelines) — there is no appeal.

## What the data looks like

| | Train | Test |
|---|---|---|
| S1 / S2 / S3 | 2,206,821 / 5,034,616 / 5,285,603 | 1,732,544 / 4,887,273 / 5,082,316 |
| S2+S3 per S1 | 4.7 (US 4.67, India 4.68) | 5.8 (US 5.75, India 5.82, France 5.53) |
| Countries (S1) | US 60%, India 40% | India 47%, US 38%, **France 15%** |

Match structure (train):

- Match-list size: 0 → 5.6%, 1 → 5.4%, 2 → 17.0%, 3 → 24.1%, 4 → 21.9%, 5+ → 26.0%.
  Singleton share is the same in US and India. 7.64M match links, 48% S2 / 52% S3.
- **Strict 1-to-many:** every S2/S3 ID belongs to at most one S1. If two S1s claim the same
  record in your output, one of them is wrong.
- ~26% of S2 and of S3 records match nothing — distractors. Same-source duplicates are normal:
  54% of clusters hold >1 S2 record, 59% hold >1 S3 record.
- True pairs **always** share the country label.

Signals, 50k true vs 50k random same-country pairs:

| | True | Random |
|---|---|---|
| ≥1 shared name token | 85.7% | 15.8% |
| shared comma-segment of the address (city-ish) | 87.3% | 2.6% |
| exact normalized name | 21.8% | 0.0% |
| both sides carry a 5–6 digit postcode | 5.2% | 0.8% |
| postcodes equal, when both have one | 94.1% | 0% |

Noise and quirks:

- **India crosses scripts:** 18% of India true pairs have a Devanagari/other Indic name on the S2/S3
  side against a Latin S1. S1 is always Latin; ~23.5% of India S2 names and ~13% of S3 names are Indic.
- Literal `NULL` / `<NULL>` placeholder in ~2.5% of S2/S3 addresses, besides truly empty ones (~3%).
- ~4% of S2/S3 names are websites; ~0.3% are `Name | www.site.com`.
- Landmark words are unreliable: "Opp" also appears as a real US town name.
- **No leakage:** ID numbers and row order are random w.r.t. matches; train and test share no IDs
  and no identical records.

Test-only:

- **France**: Latin script with accents only (no Indic/CJK). Legal forms common (SARL/SAS/SASU/SA/EURL/SCI/EI
  in 69% of S1 names), street words (rue/avenue/bd) in 95% of S1 addresses, postcodes almost
  never (0.4%). Same noise generator as US/India (typos, dropped letters, website names).
- Non-France mix shifts to 55% India / 45% US (train 40/60), and there are more S2/S3 per S1.
- Candidate budget with country-scoped blocking, all 3 countries combined: k=10 → 17M pairs,
  k=50 → 87M; the full cross join is 6.7T (France alone: k=10 → 2.6M, full join 372B).

## Traps, ranked

1. A missing S1 row (usually France) rejects the whole file — assert the row count before upload.
2. A model or API that breaks the licence / no-external rule is disqualification, not a penalty.
3. Tuning a threshold for F1 or accuracy — sweep it for F0.5 on a train holdout.
4. Hill-climbing the public leaderboard with 5 submissions/day — trust the train holdout; spend
   uploads on catching pipeline bugs. The private split decides.
5. Zero candidates for a non-singleton scores 0 silently — track "% S1 with no candidates".
6. Unknown IDs: the PDF says "rejected", the validator says "lowers score" — treat as rejection.
7. France has no labels — thresholds validated on US/India are unvalidated for 15% of test.
8. `NULL` placeholders and `|`-joined names poison string features unless cleaned first.

## Open questions — ask via the Google Form

1. When does the 5/day cap reset? (Assume midnight IST.)
2. Does every leaderboard upload need the code zip, or only the final package?
3. Does "final model" cover a blocking embedder too? Is 8B per model or total?
4. May we fit unsupervised statistics (TF-IDF, IDF weights) on the **test** S2/S3 text?
   The fair-play line says "using only the provided training data".
5. Is the final zip due at 23:59:49 on the 27th or later? (`grace_period: 15` appears on every Unstop
   round, so it says nothing specific.)
6. Final ranking: private leaderboard only, or both?
7. What exact string is `<team_name>` in the zip name?

## Links

- Queries: Google Form https://docs.google.com/forms/d/1teTPPxo06EsNRYDFHvhYqVrVDoyFh38w0LNKPYssIQk/edit
- Support: support@unstop.com with a screenshot + registered email (they won't make decisions for you)
- Play page: https://unstop.com/competitions/1743604/round/1593683/play/code
- Explainer video (6 slides, nothing beyond the PDF): https://d8it4huxumps7.cloudfront.net/files/6ab509c5b7036_ml_challenge_2026_video.mp4
- Prep blog (credits + SageMaker notes): https://builder.aws.com/content/3HiM6zDmFrF98fRzOUETnGFDoqz/amazon-ml-challenge-2026-your-complete-prep-guide-with-live-demo
- Dataset zip: https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip (1.1 GB → 2.4 GB)

## Update (25 Sep, organisers) — candidate set size counts toward the final ranking
"Candidate generation counts toward the final ranking... The approach that generates a smaller candidate set per
Source 1 entity will be ranked higher in the final evaluation beyond the public/private leaderboard." They review
`candidate_pairs.tsv` and the code that produces it. By definition that file is the exact set the final model scores,
so a learned pruning stage (meta-blocking) before the matcher legitimately shrinks it. Full text:
`data/notes/unstop_play_page_v2.txt`.
