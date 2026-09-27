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
Final rank = the private score **of the best-public submission** (organiser ruling below).

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
   uploads on catching pipeline bugs. The private split decides, but **for the best-public upload
   only**, so the final pick must also be the top public score.
5. Zero candidates for a non-singleton scores 0 silently — track "% S1 with no candidates".
6. Unknown IDs: the PDF says "rejected", the validator says "lowers score" — treat as rejection.
7. France has no labels — thresholds validated on US/India are unvalidated for 15% of test.
8. `NULL` placeholders and `|`-joined names poison string features unless cleaned first.

## Organiser rulings (Query Form responses, 24–26 Sep)

From "Response from Amazon Leaders" in the form's response sheet (PDF export, 19 pages). These settle
the old open questions:

- **Final rank = the private score of your best public submission.** Not your last upload, and you
  don't pick one. A diagnostic upload that tops the public board becomes the final submission.
- **Test-side unsupervised use is allowed:** TF-IDF/IDF, token frequencies, blocking indexes on the test
  files, and **self-training / pseudo-labelling on test** (asked with France in mind), plus synthetic pairs
  from the provided records.
- **8B + MIT/Apache-2.0 applies to each model separately:** embedder, reranker, matcher, and any
  preprocessing or augmentation model (its licence is checked too). Classic models you train (LightGBM,
  LR) are fine. A local vLLM/Ollama server is fine if it makes no third-party calls.
- **External data:** no packages that bundle geo/postal/business data. That rules out libpostal,
  geocoders, postal-code or gazetteer data, "large static country/state/city tables", **cleanco** (it
  bundles a legal-term DB) and **AWS Entity Resolution**. Allowed: small hand-written dictionaries
  (legal suffixes, street types), **a hand-written list of state names and abbreviations ("City-level or
  postal tables are not")**, and pure-algorithm libraries (RapidFuzz, jellyfish, sklearn, LightGBM,
  fastText, wordfreq). Allowed text tools: anyascii, uroman, indic-transliteration, ftfy, unicodedata2,
  textdistance, your own transliteration code, and IndicXlit within the model limits. No online translation.
- **Content-identical S2/S3 duplicates: list every one of their IDs**, not only the top one.
- **candidate_pairs.tsv** = the final candidate set fed to the matcher. In a cascade that is the input to
  the first scoring model, and a pruned later stage is fine. It goes in the final zip only; the leaderboard
  takes `matching_results.tsv` alone. Smaller sets per S1 rank higher "beyond the public/private
  leaderboard". No formula was given.
- **Private split** = the same test files, so the same 3 countries. Public and private both contain France.
- **Zip layout:** `requirements.txt` + `README.md` go in `code/business_entity_resolution/` next to `src/`.
  The top level holds only `output/`, `code/` and the doc. No page limit on the doc.
- Row order in the submission doesn't matter. You don't have to submit every day. AI coding assistants are
  allowed. SageMaker isn't mandatory.

Still unanswered: when the daily cap resets, whether weights go in the zip, the hardware used to
reproduce runs, tie-breaks, and where and when the final zip is uploaded.

**Portal behaviour other teams reported (not confirmed by organisers):**
- The config shows `allowed_submission_per_round: 5`. Several teams that used 5 uploads on 25 Sep found the
  button greyed out ("Limit to submit in this round has exceeded") on 26 Sep. A double click counts twice.
- The code-zip slot disappears after the first upload, which may lock in the first zip.
- "Failed" status with no reason, often on files of about 60–100 MB (a 24–26 MB file scored). On Windows
  `.tsv` has an empty MIME type, so the presigned-URL call returns 400 and "Please Wait 0" hangs.
- Scores teams reported on day 1: 0.953, and 0.869 at rank 346.
- `<team_name>` in the zip name is **Yeti** (Unstop team name), so the file is `Yeti_submission.zip`.

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

## Organiser rulings (received 26 Sep, relayed by the team)
1. **Final entry = the upload with the best PUBLIC score**; the final rank is that upload's private score. The entry we
   want counted must beat every earlier public score (0.981142 = D4 as of 26 Sep noon). Lower-scoring uploads cost nothing.
2. **Unlabelled test data may be used**: IDF, blocking, and pseudo-labelling on test (asked for France specifically).
3. Candidate-set size per S1 ranks teams on top of the LB (smaller is better; a pruned later stage counts; no formula).
4. Content-identical S2/S3 duplicates: list every ID; the scorer credits each correct ID.
5. 8B / MIT–Apache limits apply per model (bge-reranker-v2-m3 Apache-2.0 and e5-small MIT both pass).
6. External data — allowed: hand-written state names/abbreviations, small legal-suffix and street-type dictionaries,
   anyascii / indic-transliteration / uroman / own transliteration. **Banned:** city-level or postal tables, "large static
   country/state/city tables", libpostal, cleanco, AWS Entity Resolution.
   → Gray area in our code: `normalize.FR_REGIONS` (13 régions → ~96 départements, one level below state). Replace with a
   mapping learned from the data (allowed by ruling 2).
7. The private split uses the same test files: same three countries, France in both splits.
8. Final zip: README.md + requirements.txt inside code/business_entity_resolution/; top level = output/, code/, the doc.

## Organiser rulings (Query Form export of 27 Sep 00:02, new since 26 Sep 12:35)
1. **No model > 8B anywhere, not even as a teacher** for distillation, synthetic data or labels, with or without APIs/MCP
   ("Using any model larger than 8B parameters is prohibited even for teacher model"). We comply: the pipeline uses
   only bge-reranker-v2-m3, e5-small, LightGBM, and Qwen3-Reranker-4B (P38, 4.02B). Claude is used as an AI coding
   assistant (allowed) to read data and write rules; it produces no labels, pairs or training data.
2. **A stricter threshold for countries "unseen in training" is allowed** (asked as a rule defined by "unseen in
   training", not by naming a country: answer "Yes"). → run.py still branches on `c == "France"` in ~15 places; the
   final zip should branch on `c in UNSEEN` (test S1 labels absent from train S1). Same behaviour, compliant wording.
3. Repeated: test-side unsupervised stats, self-training/pseudo-labels, synthetic pairs from provided records, and
   HF pretrained weights (MIT/Apache, ≤ 8B) are allowed; best public score decides the final entry.
Still unanswered: daily-cap reset time, where/when the final zip goes, and the final deadline (a team asked whether it
is 27 Sep 21:00 IST; we plan the last upload by 21:00 to be safe, not 22:00).

## Organiser rulings (Query Form export of 27 Sep ~20:10)
- Submission cap raised from 15 to **17 in total** for every team, "usable until midnight today, irrespective of any other
  limits" (no per-day cap on the last day). Deadline: midnight 27 Sep IST.
- **Only matching_results.tsv is uploaded in this round. The ZIP (code, candidate_pairs.tsv, methodology) is required for
  Round 2**, via instructions to come.
- "Final ranking (beyond public/private leaderboard) will be decided based on complete evaluation of the approach,
  **computationally efficient candidate generation process and code**." So candidate-set size / efficiency and code quality
  count in Round 2. Our shipped change-set parquets (final_build / final_patch) should be ported to real pipeline steps before then.
- Coding assistants are allowed. Models: MIT/Apache-2.0, ≤ 8B, offline, fine-tuned only on the provided data (per model).
