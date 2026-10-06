# Where are the matches we are not capturing?

You are a senior entity-resolution scientist joining a competition team (Amazon ML Challenge 2026, team Yeti) for its final
~7 hours. Think deeply and out of the box, but stay grounded in the measured data points below. The question:
**where are the true matches our pipeline does not capture, and what can we still do, within the rules, to capture them
without losing precision?**

## 1. The problem in brief
- Three sources describe the same businesses. Every record has `entity_id`, `business_name`, `business_address` and `country`.
  - Source 1 (S1) is deduplicated. For every S1 record, list all S2/S3 records of the same business (0..many).
  - Every S2/S3 record belongs to at most one S1 (strict one owner), and true pairs always share the country.
- **Train:** US + India only. 2.21M S1, 10.3M S2/S3, with labels.
- **Test:** 1,732,544 S1: India 47%, US 38%, **France 15% (259,452 S1)**. France is unseen in training and has no labels.
- **Metric:** F0.5 per S1 = 1.25·P·R / (0.25·P + R), averaged over all S1.
  - A true singleton predicted empty scores 1.0; any prediction on a singleton scores 0.
  - A non-singleton predicted empty scores 0.
  - **LB = 0.8502 · F(India+US) + 0.1498 · F(France).**
- **The data is synthetic.** One generator makes the S2/S3 copies of each S1 with noise ops:
  - typos and dropped letters or words;
  - word swaps with per-country noise words;
  - legal-form changes and house-number noise;
  - address-less copies, website names, and Indic script in India.
- **It also makes decoys:**
  - sibling businesses: same name core plus a branch word, with the house number shifted by the offsets {3, 4, 5, 7, 9, 11, 13, 21};
  - same-address org-word swaps.

## 2. Hard rules (breaking one = disqualification)
- **Models:** every model ≤ 8B parameters, MIT or Apache-2.0, run locally. No model > 8B anywhere, not even as a teacher. No hosted LLM or embedding APIs.
- **No external data:** no geocoding, postal or city tables, business registries, libpostal or cleanco.
  - Allowed: small hand-written dictionaries (legal forms, street types, state names) and pure-algorithm libraries.
- **Allowed test-side techniques:**
  - unsupervised test-side statistics (IDF, token frequencies, blocking indexes);
  - self-training / pseudo-labels on test;
  - synthetic pairs made from the provided records;
  - a stricter rule for "countries unseen in training", as a class (not by naming France).
- **Output rules:**
  - Output every test S1 row, and never filter on {US, India}.
  - Matches ⊆ candidate_pairs.tsv. A smaller candidate set per S1 is a secondary ranking criterion.

## 3. Where we stand
- **Best public:** 0.987391, rank ~117 of 6,938.
  - Top-100 cut: 0.987804 (+0.00041).
  - Top-50 cut: 0.988905 (+0.0015).
  - #1: 0.991811 (+0.0044).
- **Estimated split** (holdout-based, not LB-measured): India/US ≈ 0.991, so France ≈ 0.967. Each −0.001 on India/US adds +0.0057 to the France estimate. #1 needs India/US ≥ 0.992 even with France at 0.99, so the gap is not France-only.
- **Time:** ~7 h to the last upload (27 Sep, ~22:00–23:30 IST), with 2–4 upload slots. The best-public upload is the final entry.
- **Compute:**
  - Laptop: 15 GB RAM, jobs ≤ 2.5 GB, ~5 GB free disk.
  - AWS SageMaker GPUs: 1 GPU per instance type (g4dn / g5 / g6 / g6e), ~$190 of credits.

## 4. The pipeline today
1. **Blocking:** TF-IDF on name char-3grams + address words (50/50), within state. Each SX keeps its top-2 S1 and each S1 keeps its top-10 SX, plus address-less and no-state fallbacks.
2. **Pruner:** LightGBM at τ 0.03, leaving 4.4–4.8 candidates per S1.
3. **Matcher:**
   - M2: LightGBM on 65 pairwise features.
   - Cross-encoder (CE): bge-reranker-v2-m3 fine-tuned on India/US, applied to the uncertain band 0.02 < p < 0.98.
   - France: Qwen3-Reranker-4B as its CE.
4. **India/US decision:**
   - A learned LightGBM stack over M2 p, CE scores on every pair, SX competitor mass / exclusivity, and S1 group counts, at t 0.71.
   - One owner per SX (argmax); caps of 5 S2 / 6 S3 per S1.
5. **France chain (unseen-country rules):**
   - OOD CE blend at a strict t ≈ 0.99, a key tier, filler stripping (fils / associés / cie).
   - **Group completion:** fill the missing source at p ≥ 0.70.
   - **Empty-S1 rescue** at p ≥ 0.90.
   - Acronym rule, legal-form conflict veto.
   - A learned combiner veto, plus address-less exact-name adds.

## 5. Data points
### 5a. The headline: predicted cluster counts vs truth
US and India train truth have the same cluster statistics (3.465 vs 3.459 links per S1, size shares within 0.1 pt). Assume France's truth matches (same generator).

| | empty S1 | links / S1 | one-source-only groups | S2+S3 per S1 | SX assigned (owned) |
|---|---|---|---|---|---|
| train truth, US | 5.58% | 3.459 | 14.0% | 4.67 | 74.0% |
| train truth, India | 5.59% | 3.465 | 13.9% | 4.68 | 74.0% |
| test predicted, US | 5.79% | 3.390 | 15.1% | 5.76 | 58.9% |
| test predicted, India | 5.78% | 3.381 | 15.2% | 5.82 | 58.0% |
| **test predicted, France** | **6.06%** | **3.155** | **17.3%** | 5.53 | 57.0% |

Cluster-size shares (0 / 1 / 2 / 3 / 4 / 5 / 6+ matches):
- truth: 5.6 / 5.4 / 17.0 / 24.0 / 21.9 / 14.6 / 11.4 %
- India predicted: 5.8 / 6.1 / 17.9 / 24.3 / 21.5 / 14.0 / 10.4 %
- **France predicted: 6.1 / 7.7 / 20.7 / 25.6 / 20.4 / 11.9 / 7.6 %**, so France's big clusters are truncated.

Implied missing true links, as a lower bound (true links missing = truth − predicted + false positives):
- **France ≥ 0.31 per S1 ≈ 80k**;
- India ≥ 0.084 per S1 ≈ 68k;
- US ≥ 0.069 per S1 ≈ 46k.

Test has more distractors than train (5.8 vs 4.7 S2+S3 per S1). If France's truth really is 3.46 links per S1, its expected owned share is 3.46 / 5.53 = 63% against 57% assigned.

### 5b. India/US loss budget (holdout, in LB units, after our best combiner)
- **Buckets:**
  - low-p false negatives 0.0021;
  - "stolen" false negatives (the SX went to another S1) 0.0016;
  - blocking misses: 0.0011 with an address, 0.0012 without;
  - pruner losses 0.0011;
  - false positives 0.0012.
  - Total ≈ 0.008.
- **Bayes floor 0.0037 LB (45% of India/US loss):** address-less SX whose S1's full name is shared by ≥ 2 S1. The record holds no tie-breaker.
- 70–90% of the low-p and stolen FNs are name-only SX shared by same-name S1. The rest are synthetic-brand renames sitting next to decoys at the same address.

### 5c. France facts (label-free unless stated)
- **Geography:** 3 régions / 4 départements and 23,847 distinct streets. The median house number is 24 (US 1,352), so numbers collide more.
- **Shared addresses and names:** 16.7% of France S1 share an exact address with another S1 (US 8.6%). 55.4% share a name core with another S1 (US 40.8%, India 55.0%).
- **Copy quality:** France copies are cleaner than US/India: about 5× more exact-name copies and fewer typos or house-number changes.
  - Legal forms appear in 69% of S1 names (SARL / SAS / SASU / SA / EURL / SCI / EI).
  - Street words appear in 95% of addresses; postcodes in 0.4%.
- **House numbers:** accepted France pairs are 92% house-number-equal (US 78%, India 50%). France siblings keep the house number only 2–6% of the time (US 2.7%, India 26%).
- **Synthetic brands:** France assigns only 54.8% of its 31k synthetic-brand SX (invented brand names), against India 89.5% and US 73.3%. Non-brand SX are assigned at 56–58% in all three countries.
- **Candidates:** 17.6% of France SX never reach the candidate set. 59.1% of unassigned France SX are candidates that the model rejected.
- **Decoy classes:**
  - Branch/decoy words (international, participations, distribution, holding): ~34k SX each, 99.5% unassigned. That is correct; they are siblings.
  - Org-word swaps at the same address (sportive→comité, collège→école, amicale→amis): 0.3–3.6% true. Also decoys.
- **Noise words** (lens in 5d):
  - développement: s 0.018, e 0.974, n 4,735.
  - groupe: s 0.084, e 0.887.
  - 14,636 unassigned France SX sit at the S1's house number with name = S1 name ± one such word.
- **Why the CEs miss them:** the CEs were trained on US/India, where a swapped corporate word means a decoy (US "group" / "holdings" swaps are 0% true). On France's noise-word swaps the bge r median is 0.0 and the Qwen median is 0.44, while the LightGBM matcher's median is 0.82.

### 5d. A label-free statistic that transfers across countries (the "lens")
- **Definition:** take one-word-swap pairs (S1 name core vs SX core) among the country's own argmax pairs, with a house number on both sides. For each swapped-in word w:
  - s(w) = the share at a sibling house-number offset;
  - e(w) = the share with the same house number.
- **Noise words (s ≤ 0.05, e ≥ 0.8):** holdout precision US 94.7% (n 3,603), India 96.9% (n 1,971).
- **Decoy words (s > 0.14):** ≤ 0.4% true.
- Adding s, e and the name relation as features to our France combiner made its add arm positive in both leave-one-country-out directions; without them it failed one.

### 5e. Other measured facts (US/India holdout or train)
- Decoy-word swaps at the exact address: US 5.1% true (n 215), India 0.7% (n 287).
- Name-disjoint pairs at the same house number and city: 97–99% true when the S1's address is unique, 40–53% when the address is shared.
- Same-address SX with invented names: 7,763 true vs 159 false (train).
- Addresses are noised per source: only 22–24% of same-source copies share the exact raw address, while decoys copy addresses verbatim (24% of exact-address groups span another S1).
- Content-identical S2/S3 duplicates always share one owner (60,319 groups, 100%).

### 5f. Metric break-evens (F0.5 per S1)
- Adding one more member to an S1 that already has members pays only if P(true) ≳ 0.75. Example: 4 true, 3 predicted and correct; a correct add gains +0.0625, a wrong add loses −0.1875.
- Filling a missing source for a one-sided group, or filling an empty non-singleton S1, pays at much lower P(true).
- A wrong add to a true singleton turns 1.0 into 0.

### 5g. What moved the public LB (France)
- **Proven rules:** group completion (fill the missing source at p ≥ 0.70) +0.00034; empty-S1 rescue (p ≥ 0.90) +0.00031.
- **+0.0007** for {Qwen CE swap, acronym rule, legal-form veto, address-less exact-name adds} while dropping ~41k looser pairs.
- **Threshold history:** a stricter France threshold helped early (t 0.78 → 0.88 → 0.95, each step gained).
- **Loose adds:** France rule-adds at p 0.3–0.9 netted ≈ 0 on the LB.
- **Lesson:** vetoes and adds that are ≥ 0.98 precise transfer to the LB; loose adds do not. Validation (LOCO) gains on France have transferred at roughly ¼–½.

### 5h. Tried and failed (leave-one-country-out France-F0.5 unless marked LB-eq)
- **Retrieval of India/US blocking misses:** 1.27M / 1.81M new pairs retrieved, only 244 / 254 true, 38 / 46 confirmable by the matcher. +0.00004.
- **Twin-candidate expansion:** when the owner shares a key it is already a candidate 91% of the time. The whole slice is worth ≤ +0.0002 LB-eq even with perfect decisions.
- **Typo correction** toward the S1 vocabulary: −0.0008 / −0.0020. **ByT5 corrector:** −0.002.
- **Loosening France t** (0.9868 → 0.975): −0.0012 / −0.0023. 99% of the extra pairs are extra members, at 50–62% true.
- **Dropping France links whose house number shifted** when the S1 has an exact one: −0.009 / −0.020.
- **Other misses:**
  - Expected-F0.5 decoding: +0.0001 (1.8 SE).
  - Raw-text corruption marks for India/US: null.
  - India/US unique-key recovery: +0.00002–0.00004 LB-eq.
  - More CE training data, and ensembles: no gain.
- **France false-positive hunts:** address-collision features, wrong-twin false positives and twin coin-flips. Their US/India analogues are 98.8–99.8% true, so there is no FP pool there.

### 5i. Staged but not yet LB-measured (don't duplicate)
- **Combined France add-ons ≈ +8k pairs:**
  - synthetic-trained CE column adds: 2,793;
  - word-role combiner adds: 1,991;
  - bge r ≥ .99 slice below t: 5,270;
  - exact-key adds: 835.
- **Noise-word swap tier:** 4,554 développement swaps. Its in-domain analogue passes; LOCO has no power for it.
- **Decoy-word removal:** −1,666.
- **Independent France CE:** electra / CamemBERT trained on synthetic pairs plus pseudo-labels.
- The high-confidence add tiers all converge on the same ~2–3k pairs.

## 6. What I want from you
1. **Explain the recall gap.** Using 5a–5e, where are France's ~80k and India/US's ~110k missing true links most likely?
   - Name the generator ops / record classes.
   - Estimate each class's size and our current capture rate from the numbers given.
   - Say what single measurement would confirm or kill each hypothesis.
2. **Propose 6–10 concrete capture ideas we have not tried** (or tried differently, with the reason it is different). Think beyond threshold tuning:
   - cluster-level / graph reasoning: one owner, source completion, and matching the known cluster-size distribution;
   - per-country label-free statistics and generator inversion;
   - test-time pseudo-labels;
   - candidate generation keyed on generator ops;
   - ... whatever else the data supports.

   For each idea give:
   - **Mechanism:** why the true match is missed today, and what signal separates it from the decoys.
   - **Evidence:** the data point(s) above that support it.
   - **Size:** the target bucket and its ceiling in LB (France × 0.1498, India/US × 0.8502).
   - **Test:** a ≤ 1-hour falsification test (label-free for France, holdout for India/US) and the gate it must pass.
   - **Estimates:** expected LB gain, precision risk, compute (CPU / GPU) and time.
3. **Rank** the ideas by expected LB per hour for the remaining ~7 h, and name the top 2 to run right now.
4. **List the traps:** ideas that will look good on validation but probably won't transfer to the LB, and why.

Rules of engagement:
- Use only numbers from this prompt, or mark assumptions explicitly.
- Nothing that breaks section 2.
- Be specific: exact rule / feature definitions, thresholds, and the statistic to compute.
- Prefer precision-safe recall (fill missing sources, fill empty non-singletons, complete truncated big clusters) over extra members below 0.75 precision.

## Optional: repo pointers (if you can read files)
- `docs/PROBLEM.md`, `docs/RIVAL_INTEL.md`, `SUBMISSIONS.md`
- `probes/{P36-iu-residual, P40-generator, P46-noise-word-tier, P69-fr-loosen, P70-fr-roles, P71-fr-noiseword}/RESULT.md`
- `code/business_entity_resolution/src/{run.py, france.py}`
