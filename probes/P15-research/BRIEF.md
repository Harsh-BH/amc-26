# P15-research: what have others done on problems like ours — ideas we can still use in ~36 hours
Research probe: web search + reading, then map every idea onto OUR pipeline with a cost/benefit estimate. No heavy
compute (a small feasibility check on existing cached data is fine). Write only probes/P15-research/RESULT.md (and
small scripts under probes/P15-research/ if you check something).

Read first (to know what we already do): APPROACH.md, docs/PROBLEM.md, probes/MERGE-2.md, SUBMISSIONS.md,
probes/P6-france/RESULT.md, probes/P10-candidates/RESULT.md, probes/P11-ce-big/RESULT.md, probes/P12-synth/RESULT.md.

**Our situation (26 Sep 00:05 IST; deadline 27 Sep 22:00 final upload, code freeze 27 Sep 14:00):**
- Task: business entity resolution, 3 sources; S2/S3 records each belong to ≤ 1 S1; macro F0.5 per S1 (singletons
  count). Train US+India with labels; test adds **France with no labels** (15% of test). Candidate-set size now also
  ranks teams (smaller is better).
- Pipeline: TF-IDF blocking (reverse top-2 + forward top-10, state-partitioned) → learned pruner (4.8 cands/S1) →
  LightGBM on 65 label-free pair/context features → e5-small cross-encoder on the uncertain band (India/US) →
  argmax per S2/S3 + threshold. Holdout 0.9885–0.9893; LB best 0.981142.
- **The gap to #1 (0.988419) is France:** France ≈ 0.938, India+US ≈ 0.989. France threshold 0.78→0.95 gave +0.0011.
  Known France shifts: 35% of France S2/S3 lack a region (3% in train); IDF fitted on US/India text only.
- Tried and failed: synthetic target-country pairs (P12), self-training on pseudo-labels (P6, −3.5 pt), bigger CE
  for India/US (P11, +0.0001). Running: bge-reranker-v2-m3 LOCO on AWS (P13), France distribution fixes (P14).

**Search for (cite sources, with links):**
1. Winning solutions of similar competitions — especially **Kaggle "Foursquare – Location Matching" (2022)** (POI
   matching with names+addresses, multilingual; winners' candidate generation, features, GBDT + transformers,
   post-processing/graph merging), plus any other record-linkage / product-matching competitions (e.g. Shopee
   Price Match Guarantee, Amazon ML Challenge past years, SIGMOD programming contest ER tracks).
2. Research on **entity resolution under domain shift / unseen target domain**: domain adaptation for ER (e.g. DADER),
   zero-shot / transferable matchers (e.g. AnyMatch, Unicorn, ZeroER, Sudowoodo, Ditto's tricks), cross-lingual
   address/name matching, label-free threshold selection, calibration under shift.
3. **Post-processing / collective ER**: transitivity, clustering, S2↔S3 agreement ("if two records match each other
   they share the S1"), one-to-many assignment, graph-based correction. We only do per-record argmax today.
4. Address/name-specific tricks for business names and addresses (token weighting, abbreviation handling, house
   number logic) that GBDT features could use.

**Deliverable — RESULT.md:**
- A ranked table: idea | source | what it would change in OUR pipeline (module/step) | expected gain (India/US vs
  France, with reasoning) | effort (hours) | rule risk (external data? licence?) | fits before the 27 Sep 14:00 freeze?
- Top 3 recommendations written as concrete briefs the master can hand to a probe (inputs, change, how to validate
  on our full-corpus holdout / a France proxy, acceptance bar).
- Explicitly flag anything that would break the rules (external data, hosted APIs, non-MIT/Apache models > 8B).
Memory cap 2 GB incl. CLI. Never end your turn to wait on a background job. Finish within ~2 hours.
