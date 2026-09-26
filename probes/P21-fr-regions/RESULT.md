# P21-fr-regions — RESULT
## TL;DR
- `normalize.FR_REGIONS` is now just the 13 région names, which the organisers allow as hand-written state names. The
  département map is learned at run time from unlabelled test France, with no geo table (`normalize.fr_departements`,
  cached in data/work/fr_regions.json). It learns exactly the table's entry for every département that occurs ≥ 20
  times in test France: nord, pas de calais, gironde, loire atlantique. It misses `charente` (3 S2 rows).
- Rebuilt from raw with the learned map, France's rows of **sub8c, sub9a, sub9b and sub9s are identical** to the
  staged files (0 / 259,452 rows differ in matching_results and in candidate_pairs). India/US rows are untouched by
  construction. In #8c the P3 candidates are byte-identical, and 3 M2 pairs move (max |Δp| 0.027) with no decision
  flip. In #9a's idf144 world, 3 P3 pairs swap and 22 M2 pairs move (max |Δp| 0.053) with no flip either.
  US/India parsing is unchanged: 0 of 1,493,588 holdout rows differ.
- Recommendation: merge branch `wave/p21-fr-regions` (commit b2f1d47: normalize.py, block.py, README, doc). The
  staged #8c / #9a / #9b / #9s files stay valid as they are; nothing needs re-uploading. Only the spare #9d moves
  (1,039 France rows, by design).
## Recommendation
Adopt the branch's `normalize.py` + `block.py` as they are:
- `FR_REGIONS = (13 région names)`.
- `learn_fr_departements(s1, sx, support=20, purity=0.95)`, with the slot test (≤ 5% of rows also name a région) and
  the S1-never-names test (≤ 5%). Anchors: unique exact name_core in S1 plus a shared address segment.
- `fr_departements()`, cached at `$ER_WORK/fr_regions.json`. It is hooked into `address_fields` on the first France
  row, so no caller changes.
- `block.idf_domain` takes its words from région names + learned départements.

The shared `data/work/fr_regions.json` already exists: 4 entries, built in 55 s at 1.03 GB. It is deterministic:
rebuilding it gives the same bytes. The existing France caches in data/work do not need rebuilding, because a rebuild
reproduces the same outputs. Keep the thresholds of the brief (support 20, purity 0.95): the 4 real départements
have 6.9k–37.6k anchors at purity 1.000. Only 13 segments pass the slot test at support ≥ 20. The other 9 (support 21–81:
streets and landmarks such as 'rue lyderic' and 'edhec') all fail the S1-naming test, so they never reach the anchor vote.
## Evidence
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| learned-fr-map-vs-table | test-France | learned 4 entries, all equal to the table; table has 5 entries for tokens in test France; the 1 missing is `charente` (3 S2 rows, one cluster); 0 extra, 0 wrong; anchor purity 1.000 on all 4; address_fields old vs new over 1,694,445 France rows: **3 rows differ** (those 3; state / addr_norm / places, city_key on 2); learn 55 s, 1.03 GB | keep |
| learned-fr-map-us-india-parse | holdout | address_fields old vs new over the whole holdout world (US/India S1 + S2/S3): 1,493,588 rows, **0 differ** | keep |
| (analysis) | test | the 3 rows sit in one row of each staged file (all three are candidates of one S1). None of them is in sub9a's idf_country sample (positions vs the seeded 168,224-row draw), and the rebuilt idf144 IDF arrays equal the shared ones | — |
| learned-fr-map-rebuild-sub8c | test-France | every France cache rebuilt from raw (isolated ER_WORK). S1 texts, rf_s1 and **P3 candidates byte-identical** (4,326,955 pairs). The texts differ only in `st` for the 3 rows, and rf_sx only for them. X files 000/001 differ, and **3 M2 pairs change p**: the 3 rows vs their S1, 0.99985→0.99988, 0.2897→0.3167, 0.99990→0.99962. **matching_results and candidate_pairs: 0 of 259,452 France rows differ** | keep |
| learned-fr-map-rebuild-sub9s | test-France | same rebuilt M2 p + sibling rule (170,752 zeroed), t 0.95: **0 / 259,452 rows differ** in both files | keep |
| learned-fr-map-rebuild-sub9b | test-France | same rebuilt M2 p + bge CE band: every band pair has a cached r (0 missing, no GPU), t 0.98: **0 / 259,452 rows differ** in both files | keep |
| learned-fr-map-rebuild-sub9a | test-France | the idf144 variant world rebuilt on the rebuilt base: IDF arrays identical. P3 (4,041,182 pairs) loses 3 pairs, gains 3 and shifts 3 ranks, in 3 S1s (6 of the 9 rows involve the 3 changed S2). M2 p moves on 22 pairs in 4 S1s (max |Δp| 0.053). t 0.975: **0 / 259,452 rows differ** in both files | keep |
| learned-fr-map-rebuild-sub9d | test-France | idfdom world rebuilt: 83 hashed address-IDF entries change (fewer domain words lowered). The reference is France's rows regenerated from the shared table-built caches, because output/sub9d was pruned from output/ by 15:30. t 0.98: **1,039 / 259,452 matching rows (0.40%) and 5,168 candidate rows (2.0%) differ**. Label-free profile: empty S1 6.89% → 6.88%, matches / S1 2.976 → 2.978 | inconclusive (by design) |
## What failed
- Anchor = unique exact name_core alone: région purity 0.94–0.955 on the true départements (below 0.95): ~5% wrong anchors.
  Adding "the two addresses share a segment" gives purity 1.000.
- Slot exclusivity alone lets streets/landmarks through ('rue lyderic', 'secretariat de chirurgie', …, support 20–81);
  the "S1 never names it" test (canonical, house number stripped) removes all of them.
## Interfaces
- `normalize.learn_fr_departements(s1, sx, support=20, purity=0.95) -> {segment: région}`. s1 / sx are DataFrames with
  business_name, business_address (one country's S1 and S2/S3). It is pure, and the self-check covers it.
- `normalize.fr_departements() -> dict`: the above on test France. It reads data/work/test_source{1,2,3}.parquet in
  250k batches, keeps only the France rows, and caches the result at `$ER_WORK/fr_regions.json` (atomic write).
  It costs 55 s and 1.03 GB peak, once; afterwards it is a JSON read.
- `normalize.address_fields` adds the learned entries to `STATES["france"]` the first time a France row comes in, so
  every caller gets them without changes: block.texts, rules.feats and its workers, features.rec, test_pipeline.
  `block.idf_domain` (#9d) takes its words from `FR_REGIONS` + `fr_departements()`.
- `FR_REGIONS` is now a tuple of the 13 région names.
- Probe scripts: `probes/P21-fr-regions/check_equiv.py` (old vs new parse of every row), `fr_world.py` (isolated
  ER_WORK with France caches left out), `fr_only.py` (France-only model stage + row diff vs a staged submission),
  `diff_caches.py` (per-stage cache diff), `explore.py` (the learner's design data).
## Risks
- **Compliance.** The only hand-written French geography left is the 13 région names: state level, which the rulings
  allow. The département knowledge now comes from unlabelled test rows (ruling 2), and nothing outside the data.
- **Coverage vs the table.** The learner misses any département with < 20 S2/S3 rows (charente here) and any département
  that S1 also names (e.g. "paris", which is both a city and a département — it does not occur in this test set).
  Both fall back to "no state" for those rows. That is what an unmapped segment already did.
- **Typos of région names** ("hauts de frnace") pass the slot and S1-naming tests. With ≥ 20 anchors they would be
  learned (correctly) as that région. None reaches support 20 in test, so none is learned.
- **Verification scope.** France was compared as the set of France S1 lines of each file. India/US lines come from
  untouched caches, and write order is id1 order, so the full files are byte-identical. I did not re-run the India/US
  model stage, because the 3 GB cgroup cap covers my whole process tree. The rebuilt caches were deleted afterwards
  (disk 98% full). `fr_world.py` + `fr_only.py` redo it in ~25 min.
- **Run-time dependency.** France parsing now needs data/work/test_source*.parquet (io_tsv.convert makes them in
  `run.ensure`) or a cached fr_regions.json. A process that parses France rows before the parquet exists would crash.
  The pipeline always converts first.
- **Concurrency.** rules.feats workers that all miss the cache would each learn it (~1 GB each). The pipeline builds
  it in the parent first (block.prep before run.feats); the case is marked `ponytail:` in fr_departements.
- **sub9d (idfdom) changes by design.** Its IDF-lowered word set shrinks from the words of ~96 départements to
  those of the 4 learned ones. Words such as "saint", "val", "paris" or "marne" that sit in French street/city names no
  longer get the lower IDF. #9d is also no longer train-only, because its word list is learned from test France. It was the
  "spare" option.
## Open questions
- **sub9d (idfdom).** Its word list changes by design (4 learned départements instead of ~96 listed): 1,039 France
  matching rows change (0.40%) and the profile is flat. Keep #9d with the learned words, or drop the recipe? Best guess:
  drop it or leave it as is. It was the spare "train-only" option, test statistics are now explicitly allowed, and #9a
  supersedes it. Restoring the old ~96-département word list would bring the table back.
- **Should `charente`-like rare départements be kept?** Best guess: no. A 3-row cluster cannot be learned without a
  table, it costs 0 output rows here, and the unmapped segment falls back to "no state" (the same as any unknown segment).
- **Is the doc's mention enough?** I added the learned map to the doc's "no external data" bullet and to the README
  step table. Best guess: yes.
