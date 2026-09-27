# P62-public-ideas — RESULT (final, 27 Sep 07:55 IST)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Idea 3 (clone09): keep sub_clean's India/US combiner.** Under test-like distractor density its lead over #34 grows:
  +0.000730 ± 0.000067 / +0.000604 ± 0.000062 LB-eq, against +0.000681 / +0.000528 on the plain frame. #34's own gain
  over #8c shrinks 35–60%. The arm ordering doesn't change in any replicate (Poisson, Bernoulli, λ=1.9 stress).
- **Idea 1 (acronym rule): ship the France change set (3,590 adds, expected +0.00014–0.00017 LB).** With an address
  check the rule is 0.997 / 0.9997 precise on India / US. Of the change-set pairs Qwen has scored, 98% have r ≥ 0.5.
  India/US itself is worth only +0.00001, since our decision already takes 96–97% of the rule's pairs.
- **Idea 2 (e5 dense rescue): drop.** In-domain it is borderline (+0.000100 / +0.000075, h1 under the gate), all of it
  from India. The learned verifier doesn't transfer across countries (LOCO ≤ +0.00003), so France is out. k=20 doesn't help.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Idea 3:** no change. India/US stays on sub_clean's recipe (= sub54's: P44 combiner + r_all + zoo, full refit). Do not
  revert to #34 / sub_clean_iu34 on density grounds.
- **Idea 1 (sent to P50 at ~07:05, CE addendum ~08:10):** apply `data/work/p62/acro_cs_france.tsv` on top of sub_clean.
  - Columns: s1, sx, tier, in_cand, p8c, comp_p. Append each sx to its s1's matching line.
  - Where `in_cand` is False (2,643 rows), also append the sx to the s1's candidate line, or the subset check fails.
  - Caps (≤ 5 S2 / ≤ 6 S3 after all adds) are already enforced against sub_clean; recheck if another change set touches
    the same S1.
  - The rule: `normalize.acronym(core1) == core_x.replace(" ", "")` (≥ 2-token core), with rf hn and city equal (both
    non-empty) and P3-normalised address token Jaccard ≥ 0.6. Every S1 street token must also be in the SX address
    (street tokens = address minus city/state/numbers/hand-written street-type and article words).
  - Filters: exactly one rule S1 per SX; SX not assigned in sub_clean; no other S1 on the SX at #8c p ≥ 0.5.
  - Builder: `acro.py pairs test France` → `acro_fr.py output/sub_clean` → `acro_cs.py`. Env defaults: ACRO_JAC 0.6,
    ACRO_ST 1.0.
  - Smaller first step, if wanted: tier T2 alone (236 rows, where #8c's p already agrees, median 0.998).
- **Idea 2:** nothing to adopt.

## Evidence         (table: ledger name | world | key metrics | verdict)
**Idea 3.** clone09 world = P57's full holdout frame, where every free SX (in no train truth list) gets exact clones: the
same text and the same per-pair p / r / hn / zoo p on every S1 row it has. Argmax, a10 columns, #34's p', CTX2 / CTX3
and n_comp_rows are then re-derived with `p57.world()`. The shipped boosters (P57's cache, fit on half h of the full
world) score half 1-h. Poisson(0.9) adds +112k India rows (+9.9%) and +141k US rows (+8.6%); #8c accepts 291 / 367
clones. LB-eq vs #34, paired per S1, h0→h1 / h1→h0:

| arm | plain frame | clone09 Poisson(0.9) s0 | clone09 Bernoulli(0.9) s1 | stress Poisson(1.9) s0 |
|---|---|---|---|---|
| **sub54 (= sub_clean India/US)** | +0.000681 / +0.000528 | **+0.000730 ± 67 / +0.000604 ± 62** | +0.000752 ± 68 / +0.000629 ± 62 | +0.000784 ± 76 / +0.000755 ± 72 |
| sub44r | +0.000656 / +0.000505 | +0.000732 / +0.000567 | +0.000756 / +0.000570 | +0.000810 / +0.000698 |
| rel54 (density arm) | +0.000638 / +0.000532 | +0.000666 / +0.000614 | +0.000711 / +0.000640 | +0.000712 / +0.000765 |
| rel44r | +0.000625 / +0.000485 | +0.000655 / +0.000538 | +0.000707 / +0.000574 | +0.000702 / +0.000662 |
| sub44r_psumoff | +0.000526 / +0.000447 | +0.000602 / +0.000494 | +0.000607 / +0.000506 | +0.000675 / +0.000645 |
| rall_only | +0.000331 / +0.000224 | +0.000410 / +0.000321 | +0.000438 / +0.000320 | +0.000497 / +0.000424 |
| sub44 | +0.000306 / +0.000210 | +0.000245 / +0.000195 | +0.000246 / +0.000205 | +0.000212 / +0.000249 |
| sub44_psumoff | +0.000178 / +0.000147 | +0.000161 / +0.000126 | +0.000156 / +0.000138 | +0.000100 / +0.000182 |
| yardstick #34 vs #8c (rep h0 / h1) | +0.000176 / +0.000210 | +0.000113 / +0.000129 | +0.000065 / +0.000085 | +0.000007 / +0.000104 |

Every r_all arm gains under density; sub44 (no r_all) is the only one that loses. At λ=1.9 (+21% India / +18% US
rows), sub44r, sub54 and rel54 are within one SE of each other, and #34's own gain over #8c is nearly gone. Ledger:
`clone09-pois0.9-s0`, `clone09-clonebern0.9s1`, `clone09-clonepois1.9s0`.

**Idea 1 (acronym rule).** Key = acronym(S1 core) == squashed SX core, same hn + city:

| variant | India train prec (unamb) | US train prec | France test pairs (unamb) |
|---|---|---|---|
| hn + city only | 0.25 (0.78) | 0.996 | 72,245 (10,851) |
| + address Jaccard ≥ 0.6 | 0.9935 (0.9976) | 0.9997 | 17,338 (10,760) |
| **+ every S1 street token in the SX address (shipped)** | **0.9973 (0.9991)** | **0.9997** | **11,495 (11,371)** |

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| clone09-pois0.9-s0 | holdout clone09 | the table above | keep (India/US unchanged) |
| acro-holdout-jac.6-st1 | holdout fold 0 | India: 587 rule pairs, sub54 accepts 566, 19 free misses at precision 0.89. US: 319 / 311 / 8 at 1.00. LB-eq +0.000007 / +0.000012 | drop (India/US) |
| acro-france-changeset-vs-sub_clean | test France | 11,371 unamb rule pairs: sub_clean matches 7,608 (67%), gc70 6,595 (58%). 3,693 free misses → 3,590 after the competitor and cap filters. Expected LB +0.00014 at q=0.9, +0.00017 at q=0.95 | keep (to P50) |
| acro-france-changeset-ce-audit | test France | Qwen on 113 scored pairs: 98.2% ≥ .5, 68% ≥ .97 (sub_clean's matched acronym pairs: 99.4% ≥ .5). bge on 383: 53% ≥ .5, median 0.60. India/US true acronym pairs: bge p_r_all ≥ .5 in 96% / 90% | keep |
| e5-rescue-k5-cheapverifier | holdout world, test-sized index | ceiling +0.0033. Blocked-miss owner at top-1: 14% India / 12% US; top-5: 23% / 22%. τ .85: India 110 adds at prec 0.96, US 25 at 1.00. LB-eq +0.000107 ± 26 / +0.000076 ± 20. LOCO best +0.00003 / +0.00002 | inconclusive → drop |
| e5-rescue-k20-cheapverifier | same | top-20 lifts blocked hits to 35% but US precision falls to 0.51. LB-eq +0.000097 / +0.000075. The transparent rule (token-set 100 + hn + city on top-1) has precision 0.02 | drop |

- Expected-gain model (idea 1): each touched S1 is assumed complete after its adds and its current matches correct;
  q = precision of the adds. Break-even q ≈ 0.73.
- Nothing in the change set was matched by any earlier upload: gc70 has 0 of T1, 6 of T2 and 26 of T3.
- France blocking gap check: an exact name_core key with the same address checks finds only 55 free misses in France,
  where sub_clean matches 99.97% of 311k such pairs. France's gap is specific to acronym SX, which share no name token.

## What failed      (approaches tried and dropped, with the number that killed them)
- Acronym rule on hn + city alone: India precision 0.25, which costs −0.00015 LB-eq if applied. The address and street
  checks fix it.
- Acronym rule on India/US: correct but redundant. m2 already has an `acro` feature and blocking catches these pairs,
  so the gain is +0.00001.
- Exact-core same-address rule on India/US: its holdout misses are look-alikes that the decision rightly rejects
  (precision 0.02), costing −0.0008 LB-eq.
- e5 dense rescue: recall-limited. Even at k=20 only 35% of blocked misses have their owner anywhere in the list. The
  verifier that makes it precise doesn't transfer across countries: LOCO precision 0.37 / 0.21 at τ .85.
- First dense-rescue accounting (+0.00008) was too optimistic. Distractor FPs landing on confuser S1 were not charged,
  and world distractors whose owners sat in the index retrieved those owners. Fixed by dropping the owners of world
  distractors from the index and charging confuser FPs to random holdout S1.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `clone09.py build <c> [lam] [seed] [pois|bern]` -> `data/work/p62/A_clone<mode><lam>s<seed>_<c>.parquet`
  (India 64 s, US 92 s, peak 1.8 GB; frames deleted after eval, for disk). `clone09.py eval [lam] [seed] [mode]`
  -> `data/work/p62/ev_<world>.json` (~4 min; uses P57's cached boosters `data/work/p57/b_*`).
- `acro.py pairs <split> <c> [acro|core]` -> `data/work/p62/{acro,core}_<split>_<c>.parquet` (s1, sx, n_s1, jac, st;
  ~1 min / country). `rule(j)` applies the Jaccard / street / ambiguity filter. `acro.py holdout`: precision + misses vs
  sub54 (~3 min, 1.2 GB).
- `acro_fr.py [sub_dir]` -> `data/work/p62/acro_fr.json` + `acro_fr_free.parquet`. `acro_cs.py` -> the change set
  `data/work/p62/acro_cs_france.tsv` + `acro_cs_france.json` (expected LB per tier and q).
- `dense.py ceiling` (miss buckets under sub54: `miss_<c>.parquet`, `dec54_<c>.parquet`).
- `dense.py retrieve <c> [k] [bs] [tag]`: e5-small top-k on GPU, ~7 min India / 5 min US on an RTX 3050 under
  `flock /tmp/amc-gpu.lock`. `DENSE_TAG=<tag> dense.py eval [half|loco]` -> `dense_eval_<mode><tag>.json`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Idea 1's France precision is not measured.** The evidence is indirect: rule precision 0.997–0.9997 on India/US; the
  Qwen audit of 113 scored pairs; bge lukewarm at 53% ≥ .5. The CE-scored subset is the part the France chain touched
  (P42 hn extras, P48 veto), not a random sample.
  - The holdout pairs our decision misses are dirtier (0.93 on 27 pairs) than the rule overall.
  - France's misses look structural rather than hard (2,643 were never candidates).
  - The rule uses only hand-written word lists (street types, articles) and the data's own rf / P3 fields. No external
    data, no country name in the logic.
- clone09 recomputes only the stacker-level context. m2's own input context inside #8c's p is kept, so a clone gets
  exactly its original's p.
- Exact clones are a crude stand-in for test's extra distractors: train has no content-identical group that is entirely
  unmatched (P54). They do not trip any SX-identity feature, since a10's twins are S1-side and name_sup only looks at
  accepted SX.
- e5-small (MIT, 118M) was used only for the dropped idea 2; nothing from it ships.

## Open questions   (for the master; each with your best guess)
- Ship all 3,590 acronym adds or only T2 (236)? Best guess: all. The unscored tiers are exactly what France blocking
  cannot see, and the Qwen audit is cleaner than LB-proven adds. If one upload is spent on it, the delta vs sub_clean
  isolates +0.0001–0.00017.
- Should France blocking get an acronym key (acronym × hn × city) in the pipeline, so candidate_pairs covers these
  natively? Best guess: yes for the final zip. It is ~3.6k extra candidate pairs, negligible for the reduction ratio.
- Worth a SageMaker bge/Qwen pass on the 2,643 unscored T1 pairs? Best guess: only if an upload slot is at stake; ~10
  min on g6e.
