# NIGHT (Aditya laptop, 27 Sep 03:00 IST →) — master notes

Branch `night/aditya` (local, never push main). origin/main fetched every 30 min by `fetch_loop.sh`
(log /tmp/claude-1000/amc_fetch.log); merge by hand. Only Harsh uploads.

## Where the score is lost (from P36/P40 RESULTs)
- LB 0.986304 (#37_gc70) → loss 0.0137. India/US holdout loss ≈ 0.0083 LB-eq, of which 0.0037 is P40's "Bayes floor"
  (address-less SX whose true S1's full name is shared by ≥ 2 S1). France ≈ 0.964 → ≈ 0.0054 LB.
- #1 at 0.990 ⇒ plausibly France solved to ~0.99 with India/US ≈ ours. France is the lever (max ≈ +0.004).
- Harsh is running P44 (meta stack, India/US), P45 (stage ensembles), P47 (France shape matching). Don't duplicate.

## Log
- 03:10 P40's "Bayes floor" re-checked with raw names (twin_raw.py): SX copies the true twin's raw name 4.3% vs the
  other twin's 5.3% → no surface leak; floor holds. DROP.
- 03:13 Unique-core address-less SX (al_core*.py): train 96.6% US / 97.7% India true for the unique-core S1, but the
  in-domain model already accepts 96%; leftovers at p<0.5 are mostly false (word-drop variants of other S1s). DROP.
- 03:18 Read 45 unaccepted France argmax pairs (P31 base, p 0.3–0.99). Launched agy (gemini-3.1-pro-high) probes:
  P48 filler-phrase tier (same address, "& Fils"/"Cie"/"& Associés"... beyond P46), P49 fuzzy street (elision /
  abbreviation / typo with equal name + hn), P50 exact-name at a different address. Common rules: AGY_COMMON.md.
- 03:25 P48 + P50 OOM-killed in 3 min (full pairs_France in pandas); agy died with them (scope OOMPolicy=stop). Relaunched with MemoryHigh/OOMPolicy=continue + MEMORY GOTCHAS in AGY_COMMON; P50 queued after P49 (RAM).
- 03:50 merged origin/main (P44 sub44 India/US combiner +0.0003 staged on sub46_on43). P48 v1 rejected (loose lens unvalidated) → REVIEW1.md, rerun queued. P49/P50 running; their python children get OOM-killed and retried (agy survives).
- 03:40 P49 fuzzy street: NEGATIVE (holdout pred==False precision US 2.4% / India 0.8%; neighbouring-branch decoys). Only 2,487 France pairs are true normalisation bugs (space after elision). DROP. P50 relaunched with OOM-preferred children (BASH_ENV/ZDOTDIR oom_score_adj 1000).
- 03:48 P49 REOPENED: old bar (pred==False precision) is wrong for France. Revised bar = negatives-per-S1 bound (AGY_COMMON). P49 cell: 15,512 rejected France argmax pairs; −6,100 taken by hn-extra (p≥.8, non-empty) → ~9,412 NEW (776 fill empty S1), bound ≥0.956 (US rate) / ≥0.715 (India rate). Rough +0.0002 LB. P49 rerun queued with REVIEW1; P48 rerun running with REVIEW1.
- 03:52 agy kept getting OOM-killed inside cgroup scopes (P48/P50 x3) → dropped scopes; rsswatch.sh kills any .venv python > 2.2 GB RSS instead (log /tmp/claude-1000/rsswatch.log). Max 2 agy at once; queue P49 → P50 → P48.
- 03:58 P49 final: NEGATIVE under the revised bar too — exact-street adds all covered by P28 bxneg; only 2,443 fuzzy pairs truly new (after P42/P37/P28 dedup), India-rate bound 0.724 < 0.75, 30% add1 profile unlike the accepted cell. DROP. (My 9.4k estimate ignored P28/gc.) Launched P51 France FP audit (precision side — nobody has looked).
- 04:00 P50 final: rule valid (S3, raw-name equal, k=1; bound 0.84 India / 1.0 US) but only 150 France pairs (≈ +0.00001). Park as an optional add-on. Launched P52 gap survey (reads all RESULTs, challenges the floor, runs cheap tests).
- 04:10 P52 survey: weak. Its twin 'hypothesis A' re-tested correctly (twin_dupname.py): reverse and weak (true twin 8.1% vs 6.1%; split-case 43% vs 50% chance). Floor stands. Most of its France 'untried' levers are already in the base.
- 04:12 agy Gemini quota EXHAUSTED (all Gemini models, resets ≈ 08:15). GPT-OSS-120B still works → P48 + P51 relaunched on it. P51 partial scan: accepted sibling-offset France pairs are NOT a FP signal (holdout accepted hc=1 precision 0.98 US / 0.99 India). Name-eq × hn-eq rejected France pairs (30k) = mix of address-less twins, street typos, different-street decoys: no clean win. Queued P53: pure drop1 at identical address (~3.5k, bound 0.976 US / 0.72 India).
- 04:18 P51 CLOSED by master: negative (no supported FP cell). GPT-OSS p51.py rejected (wrong statistic, 0-support cell). Lesson: GPT-OSS needs narrow mechanical tasks.
- 04:22 P53 (run by master, GPT-OSS gave up): drop1 at identical address → only 1,026 new France pairs, bound 0.979 US / 0.737 India, ≈ +0.00004; P28 likely overlaps. DROP. France cell rules are saturated.
- 04:25 France blocking: 261,886 SX never a candidate; 13% share a core with some France S1, 3.9% core+city, 0.9% (≈2.3k) core+city+hn → blocking misses ≤ ~2.3k strong pairs (≈ +0.0001). Parked.
- 04:27 P48 CLOSED: filler variants are core-drop1 after R1; at identical address only ~20 unaccepted filler pairs remain after hn-extra/P28/P46. v1's 19.6k was a proxy-base artefact.
- 04:33 **N54 FOUND**: France same-hn org-word-swap decoys (comite/ecole/amicale/club…, lens s>0.2 n≥20) — in-domain cell 0/65 US, 1/759 India true; France accepts ≈5.4k (4.7k via rules p<0.9868, mostly hn-extra). Removal script p54.py (ruled/all). Expected +0.0002–0.0005; break-even 29% false. → push branch for Harsh.
- 04:36 Renamed night probes N48–N54 (Harsh launched P48-fr-ce-all / P49-twin-prior). Merged origin/main. PUSHED branch night/aditya (N54 + FOR_HARSH.md).
- 04:40 N54 admixture test: removed band ≈ 90–100% decoy → est. +0.0004–0.0006. Warned in FOR_HARSH that P47's fill must exclude the cell. Pushed.
- 04:50 N54 smoke test on local #31-era output: ruled removes 1,075 pairs (base predates hn-extra), 1 s, official validator PASS (system python3; .venv validator > 2.2 GB gets killed by rsswatch).
- 04:55 Profiled simulated hn-extra (35.9k) and gc (10.1k) adds: apart from N54's decoy-word swaps, they are name-equal / noise-word / brand pairs — clean. N54 is the only FP family in the add rules.
- 05:10 **N55 FOUND** (and N48 closure was wrong): ≈12.7k France same-address noise-word swaps (fils/cie/associés/groupe) still rejected; in-domain analogue 0.998/0.997; France lens ≈1% decoy for fils/associés. Script n55.py (strict/all); chained N54→N55 smoke test PASS. Est. +0.0004–0.0007. Pushed.
