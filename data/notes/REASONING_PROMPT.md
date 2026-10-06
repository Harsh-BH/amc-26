# Prompt for an external reasoning session (rules/ideas only)

> **Compliance (read first):** organiser ruling (27 Sep): an AI assistant may read data and help *write rules*,
> but must produce **no labels, no pairs and no training data**, and no model > 8B may act as a teacher.
> So this prompt asks for **hypotheses and rule designs only**, never for a verdict on any record. It carries
> aggregate statistics plus a few **train** examples (already labelled by Amazon); it contains no test pair decisions.
> Don't paste test records into the session or ask it to classify anything.

---

## PROMPT (copy everything below this line)

You are an expert in entity resolution and in reverse-engineering synthetic data generators. Help me find **decision
rules or features** that separate TRUE noisy copies from generated DECOYS in a business entity-resolution task. Do not
give me labels for any record. I want hypotheses, tests I can run, and rule designs, ranked by expected value.

### Task
- Three sources. S1 = clean master records (name, address, country). S2/S3 = noisy records. Each S2/S3 record either
  is a noisy copy of exactly one S1 ("owned") or belongs to none (a decoy/orphan).
- Output: for each S1, the list of S2/S3 ids that are its copies. Metric: per-S1 F0.5, macro-averaged (precision
  weighs 2× recall). An S1 with no true copies scores 1 only if we output nothing.
- Countries: India 46.8% of test S1, US 38.3%, **France 15.0% with NO training labels** (unseen country). Train has
  India and US labels only.
- Leaderboard = 0.8502·F(India+US) + 0.1498·F(France). We're at **0.987391**, and the top-100 cutoff is ~0.9878.
- **Forbidden:** external data (gazetteers, postal/city tables, registries, geocoding, internet), models > 8B
  anywhere, hosted LLM labels. **Allowed:** statistics computed on the test files themselves (label-free),
  pseudo-labels, synthetic pairs made from provided records, small hand-written dictionaries (legal forms, street
  types), pretrained models ≤ 8B (MIT/Apache).

### Current pipeline
TF-IDF blocking → pruner → LightGBM (65 pairwise features: name/address similarities, house-number relation classes,
legal form, token IDF, etc.) → cross-encoder rerank (bge-reranker-v2-m3; for France also Qwen3-Reranker-4B+LoRA
trained on India/US) → per-country combiner → one owner per S2/S3 (argmax), caps 5 S2 / 6 S3 per S1.
India/US accept at p ≥ 0.71. France is stricter: p ≥ 0.9868, plus label-free rules. It fills a *missing source* or
an *empty* S1 at lower p (rescue at 0.90, group completion at 0.70), and vetoes on conflicting legal forms, among others.
France thresholds were validated by leave-one-country-out (LOCO): train on US and apply to India as if unlabelled,
and vice versa.

### What we know about the generator (measured on train truth)
- True copies: typos, case changes, accent injection, legal-form dropped or changed ("LLC" ↔ "L.L.C."), one
  **noise word** added or swapped (US "Partners"; France "Développement", "Groupe", "Fils", "Cie", "Associés"),
  tokens reordered, **address removed** (4–5% of copies vs 0.2–0.3% of decoys), house number *mangled*
  (`4944`→`004944`, digit dropped, one-digit typo), whole name replaced by gibberish syllables with the address kept.
- Decoys: copy an S1 and **replace the house number with a nearby/sibling number** (`8368`→`837`,
  `17023`→`1703`, offsets like +2/+10), or **swap one name word for a real vocabulary word** (34% of non-matches vs
  5% of matches). US "Group"/"Holdings" are decoy markers; France "Comité", "École", "Amicale", "Club", "Parents",
  "Sportive", "Anciens" are decoy markers. For each word we measure a label-free "sibling-offset share" *s*: how often
  pairs that swap in that word sit at a sibling house number. Decoy words have s ≈ 0.29–0.42, noise words s ≈ 0.02.
- S1 names are drawn **with replacement**, so same-name S1 twins exist. Row order and IDs carry no signal (checked).

### Where the remaining loss is (estimates)
| Bucket | LB loss | Notes |
|---|---|---|
| India/US: address-less SX whose name is shared by ≥ 2 S1 | ~0.0037 | no tie-breaker found (name traits, row order, IDs, group sizes all random) |
| India/US: heavily rewritten copies never retrieved or scored low | ~0.002 | widening retrieval gave 254 true of 1.8M new pairs |
| India/US: decoys accepted (sibling hn / one-word swap) | ~0.002 | |
| France total | ~0.0048 | France F0.5 ≈ 0.967 vs India/US ≈ 0.991 |

### France-specific numbers (test, label-free)
- 1.43M France S2/S3; 43% unassigned. Of those, 59% are in our candidate lists; only 17.6% of all France SX are never
  candidates (India/US 22–24%). **So France loses on decisions, not retrieval.**
- Rejected argmax pairs by matcher p: [.95, .9868) 15,231 (Qwen median .96); [.9, .95) 7,398; [.5, .9) 29,630
  (Qwen median .42). **98–99% of pairs near the threshold would be an EXTRA copy from a source the S1 already has.**
  In LOCO those are only 50–62% true, and break-even is ~73%. Additions that worked on the leaderboard all filled a
  *missing* source or an *empty* S1.
- Loosening France's threshold lost on both LOCO and the leaderboard. A looser France (+40,888 pairs) scored 0.000706
  worse than the current one.
- 14,636 unassigned France SX sit at their S1's house number with the name = S1 name ± one noise word.
- France's rate of same-address one-word-swap pairs is 13× India's per S1.
- LOCO says French-like band pairs at p 0.5–0.9 are only 20–44% true: the models are over-confident on the unseen country.

### What worked / what failed (so you don't repeat them)
Worked: house-number relation features (+1.8–2.0 pt in LOCO); Qwen reranker for France; legal-form conflict veto;
address-less exact unique name adds; a combiner trained on synthetic French-style pairs (LOCO +0.002).
Failed: typo correction before matching; a learned ByT5 corrector; raw string similarity columns (they don't
transfer across countries); more training data; ensembles/model zoos; calibration/prior-shift methods;
pseudo-labelling (helps one direction, hurts the other); loosening thresholds; wider retrieval.

### Train examples (labelled; SX = noisy record, TRUE = its real S1)
```
MISSED  SX "Yumabelokelo | 004944 Catamaran Street, City Of Oxnard, California"
        TRUE "Larry Workshop LP | 4944 Catamaran Street, Oxnard, CA"
MISSED  SX "Heritage Four | (no address)"  TRUE "Heritage Four PC | King City, OR"  (another "Heritage Four" S1 exists)
MISSED  SX "COZY SAEDDO | 1123 IRVING STREET, WINSTON-SALEM, NC"  TRUE "Cozy Seafood | 1121 Irving Street, Winston-salem"
MISSED  SX "Unique Agr0 | (no address)"  TRUE "Unique Agro Private Limited | Kottayam, Kerala"
FALSE+  SX "Inc Armstrong Aurora | FORT WORTH, TX, 837 PIPESTONE DR"  (decoy) matched to
        "Armstrong Aurora Inc. | 8368 Pipestone Drive, Fort Worth, TX"
FALSE+  SX "Miron, Chelsy, Ph.D. Ltd | 1703 BELLEVUE COURT, MILLVILLE, DE" (decoy) matched to
        "Miron, Chelsy, Ph.D. | 17023 Bellevue Court, Millville, DE"
FALSE+  SX "Apex Media Private Private | E##701, Ahmedabad" (decoy) matched to "Apex Media Private Limited | E-811, …"
```
Note: true `1121`→`1123` (a digit typo) and decoy `17023`→`1703` (a digit drop that looks like a sibling). Both are
"one-digit edits". That's the core ambiguity.

### What I want from you
1. **House-number forensics.** Propose a precise taxonomy of house-number edits (zero-pad, digit drop, digit
   substitution, transposition, sibling offset ±k, length change) and argue which classes the generator uses for
   copies vs decoys. What *joint* evidence (e.g. hn edit class × whether the name also changed × whether the street
   token also changed × source S2 vs S3) would separate `1121→1123` from `17023→1703`? Give a table of testable cells.
2. **Group-level (set) reasoning.** Each S1 has a small truth set (we cap at 5 S2 / 6 S3; typical size ~3.5; singletons 5.6%).
   Suggest label-free constraints or signals at the S1-group level: slot filling, per-source counts, consistency of
   edits across members of one group, "the generator made k copies with an independent op each". How would you
   decide an extra same-source member?
3. **Unseen-country transfer.** France has no labels. Which statistics can be estimated label-free on France and
   used to *re-estimate* per-word or per-op likelihood ratios, instead of transferring India/US ones? (We did this
   for words with the sibling-offset share s; what else could be measured the same way: op frequencies,
   co-occurrence of ops, position of the edit?)
4. **Address-less same-name twins (~0.0037, the biggest loss).** Is there *any* principled tie-breaker left, e.g. the
   generator's sampling process, per-S1 copy-count distribution, how many copies each twin already has, the source
   of the SX, name-casing/format fingerprint shared across copies of the same S1? Or is expected F0.5 better served
   by a policy (assign to the twin with fewer copies; assign to none; assign to all?). Remember: one owner per SX,
   macro F0.5 per S1, and precision weighs 2×. Work out the expected-value math for each policy with 2 and 3 twins.
5. Rank all your ideas by (expected LB gain) × (probability it works) / (implementation hours). We have ~7 hours
   left and can run LOCO checks quickly. For the top 5, give the exact test to run and the pass criterion.

Be concrete. Show your math. Say explicitly when you think something is a dead end.
