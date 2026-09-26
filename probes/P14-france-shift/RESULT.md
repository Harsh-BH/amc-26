# P14-france-shift — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
France's biggest fixable shift is the **IDF**, not the missing state. The train IDF gives France's most common address tokens
(city, région, "rue") near-max weight: 10.7–14.8, vs 3–5 for US "st"/"tx". On a labelled France-sized proxy, that under-counting costs
**−1.23 pt F0.5**, and refitting the IDF with the country's own S2/S3 at its 14.4% test share wins back 96% of it (0.9716 → 0.9834, pooled 0.9839).
**Upload #7 with France swapped for `prob_test_France_idf144.parquet` at t 0.975** (D4's operating point). US/India stay byte-identical.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
1. **LB tests (26 Sep), in this order.** Each is one command, then the validator (combine.py prints the command):
   - **A (first): `idf144` at t 0.975.**
     `.venv/bin/python probes/P14-france-shift/combine.py output/sub7 data/work/p14/prob_test_France_idf144.parquet 0.975 output/sub8a`
     0.975 is `t_eq_d4`: France's empty-S1 rate equals D4's 6.85%, the best France LB point so far (6.79% empty, 3.01 matches per S1;
     D4 6.85% / 2.99). Same operating point, different ranking, so the LB delta measures the IDF fix alone.
   - **B (threshold step, only if A beats #7): A at t 0.95** (6.43% / 3.10), to see if the fixed France wants a looser t.
   - **C (optional, spare slot only): `idffill` at t 0.98** (IDF fix + state filled from the city; 6.90% / 2.97):
     `prob_test_France_idffill.parquet 0.98 output/sub8c`. The US proxy says fill is neutral, but it differs from A on 21.8k France
     S1 (8.4%), so the LB would still separate them.
   - `combine.py` is checked: it rebuilds `output/sub6c` from #6's France probabilities at 0.95 byte for byte (both files).
     #7 changes no France row vs #6c. The candidate list for France is every pair in the file, which is what M2 scored after the pruner:
     5.03 per S1 for A, 5.00 for B (#6 5.14).
2. **src change if A wins (P9).** In `block.py` / `features.py`, the vectorizers for a country label without training data are
   `variant.idf_vecs`: the train TfidfTransformer's df (recovered from `idf_`), plus the df of a seeded 168,224-row sample of that
   country's P3 S2/S3 texts. That is share 0.144 = the country's share of test S2/S3. `idf_ = ln((1+n+m)/(1+df+df_c)) + 1`.
   Blocking, name_cos / addr_cos and the context features all use it. US/India keep the train IDF → holdout and test unchanged.
   Cost: one extra France blocking + X + M2 run (≈12 min on a quiet box; 90 min at load 20).
3. **Fill state from city: don't adopt** (US proxy: +0.00008 at t 0.69, −0.00004 at the oracle, candidate recall −0.0007),
   unless C beats A on the LB. A S2/S3 with no parsed state, one of whose places has a ≥ 95%-pure
   state among ≥ 10 same-split S1, gets ", <the state as S1 writes it>" appended to its raw address (`variant.city_states`,
   `fill_state`, `state_display`, `append_state`). It touches 31.9% of France S2/S3 and 0.0% of US/India (train and test). There,
   a state-less record never has an address, so US/India are byte-identical by construction.
4. **P6's france.py rules (fr6):** neutral at the model stage (2.4k France S1 change, profile identical). Fold them in with the next
   src change if convenient; they don't need an LB slot.
5. **France threshold:** P6's label-free rule gives **0.78** on #6, **0.78** on idf144 and **0.80** on idffill, against the LB-best
   0.95 on #6. The rule under-shoots on France (LB: 0.78 → 0.980017, 0.88 → 0.981024, 0.95 → 0.981142). Until the LB says
   otherwise, keep France at D4's empty-S1 rate (6.85%) rather than at a fixed t. `compare.py` prints `t_eq_d4` for any file.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| feature-shift-test | test-1file | largest standardised shifts, France vs US/India, on accepted pairs: sx_margin −1.69 (0.23 vs 0.42/0.44), place_n_1 −1.62, state_both −1.24 (0.64 vs 0.98/0.97), hn_eq +0.59, city_eq +0.56 | keep |
| state-less-means-address-less | train+test | S2/S3 without a state that still carry places: train India / US 0.0 / 0.0%, test India / US 0.0 / 0.0%, test France 31.9% (state-less 34.9%) | keep |
| idf-mismatch-proxy-India3 (world-IN_pooled / IN_loco / IN_locoown) | train India dl+up+ka, 264,823 S1 / 1,239,617 S2+S3, fold 0 scored (26,467) | M2 F0.5 at t 0.69: pooled IDF (India = 40% of the pool) 0.98393; US-only IDF 0.97161 (−0.0123, flat across t: 0.9709–0.9716 for t 0.5–0.75); US-only + own S2/S3 at 14.4% 0.98335 (oracle 0.98366 at t 0.73). Band 0.3–0.9: 3.8 / 5.9 / 3.9%. Candidate recall 0.9898 / 0.9800 / 0.9896 | keep |
| world-US_pooled | train US nc+oh+il, 255,579 S1 / 1,194,014 S2+S3, fold 0 (25,466) | M2 F0.5 0.98635 at t 0.69 (oracle 0.98654 at 0.65), band 2.9%, candidate recall 0.9932 | reference |
| world-US_blank | same US world, state segment removed from the raw address of 380,431 S2/S3 (33% of stated → 35.5% state-less, France 34.9%), end to end (blocking, rules, features) | M2 F0.5 0.98612 at t 0.69 (**−0.00023** vs 0.98635; oracle 0.98630 at 0.64); band 3.01% (2.92%); candidate recall 0.9931 (0.9932) | missing state is a 0.02-pt lever |
| world-US_blankfill | US_blank, then fill from city: 277,144 filled (73% of the blanked; US cities are less state-pure than France's), fill precision vs the true S1's state 0.9964 (202,853 true pairs), state-less 35.5 → 12.3% | M2 F0.5 0.98620 at t 0.69 (blank 0.98612, base 0.98635); oracle 0.98626 (blank 0.98630); candidate recall 0.9924 (blank 0.9931) | fill neutral: drop |
| profile-sub6 | test-France | #6 France: t_P6 0.78, empty 5.69 / 6.20 / 6.85% and 3.24 / 3.13 / 2.99 matches per S1 at t 0.78 / 0.88 / 0.95; band 0.3–0.9 11.9% of argmax pairs; accepted (p ≥ 0.95) sx_margin 0.223, addr_cos 0.758 | reference |
| profile-idf144 | test-France | t_eq_d4 0.975 (6.79% empty, 3.01 per S1); band 11.9 → 9.5%, p ≥ 0.95 66.2 → 69.1%; accepted sx_margin 0.223 → 0.306 (US/India 0.44/0.42), addr_cos 0.758 → 0.813; t_P6 0.78; at t 0.95: 6.43% / 3.10 per S1, 44.8k S1 answers differ from #6; 5.03 kept candidates per S1 (#6 5.14) | keep → LB A |
| profile-idffill | test-France | IDF 0.144 + fill: band 9.2%, p ≥ 0.95 69.7%, accepted sx_margin 0.306, addr_cos 0.848; 5.00 kept candidates per S1; t_P6 0.80, t_eq_d4 0.98 (6.90% / 2.97); 52.1k S1 differ from #6 at t 0.95, 21.8k from idf144 (each at its t_eq_d4) | keep → LB B |
| profile-idfdom | test-France | train-only fallback (158 French domain words at the US street-type IDF 4.13; cities stay at max): band 11.3% (#6 11.9%, idf144 9.5%), p ≥ 0.95 68.5%, accepted sx_margin 0.261, addr_cos 0.886; t_P6 0.86, t_eq_d4 0.98 (6.89% / 2.98); 39.2k S1 differ from #6 at t 0.95 | fallback if test statistics are ruled out |
| profile-fill | test-France | fill alone: 31.9% of France S2/S3 filled (3 régions, 50 pure places); band 12.0% (#6 11.9%), accepted sx_margin 0.214; 22.2k S1 differ at t 0.95; t_P6 0.79; t_eq_d4 0.945 | label-free neutral |
| profile-fr6 | test-France | P6 france.py as France's rules.feats (same blocking as #6): 2,367 S1 differ at t 0.95 (+1,253 / −1,231 pairs), band 11.94%, p ≥ 0.95 66.3%, t_P6 0.78, t_eq_d4 0.95 | neutral: fold in, no LB slot |

Why IDF, mechanically: TF-IDF name/address cosines drive P3 blocking and 10 of the 65 M2 features: block_score, name_cos, addr_cos,
and the s1/sx rank / gap / margin context. France's commonest address tokens were unseen or rare in the train pool,
so hashing gives them max IDF. Every same-city pair then shares a heavy "lille" / "hauts de france" / "rue" core.
Unrelated same-city records score close together (accepted-pair sx_margin 0.22 vs 0.42–0.44 in US/India), and the uncertain band
triples (11.9% vs 3–4%). The India proxy reproduces the pattern with labels: a wider band (3.8 → 5.9%), a flat −1.2 pt loss at every t,
and lower candidate recall. Adding the country's own text to the IDF pool undoes all three.

## What failed      (approaches tried and dropped, with the number that killed them)
- **Fill state from city (levers 1+2):** labelled US proxy, state blanked to France's rate end to end. The missingness itself costs
  only −0.00023, and filling recovers +0.00008 at t 0.69 (−0.00004 at the oracle; candidate recall 0.9931 → 0.9924). On France test
  the fill alone does not look more like training either: band 11.9 → 12.0%, accepted sx_margin 0.223 → 0.214, 22k S1 changed.
  With the train IDF, the appended région is itself a max-IDF token.
- **Missing state as the explanation of France's uncertainty:** state-less and stated France S2/S3 have the same p profile
  (argmax p ≥ 0.95: 65.2% vs 65.6%; band 12.0 vs 11.4%; sx_margin 0.19 vs 0.18). The one real state effect is in the rules. A state-less
  record may only join name groups of ≤ 50 S1 (`rules.MAX_NOSTATE_CHAIN`), which blocks 7.3% of state-less France S2/S3 (2.6% of all).
  Their rule acceptance is 0.52 vs 0.61.
- **P6's france.py rules at the model stage:** 0.9% of France S1 change and the profile is identical. Not worth a slot, so the
  idf+fill+fr6 combination was not run.
- **Missingness-matched retraining (lever 2):** not run, because the loss it would recover is 0.0002. With state blanked to France's
  rate end to end (blocking, rules, features), m2_big goes 0.98635 → 0.98612 on the US proxy. P6's score-time blanking found 0.0000.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `variant.py NAME [--idf S] [--idf-dom] [--fill] [--fr6] [--from W]`: rebuilds France test end to end under a variant, in
  `data/work/p14/w_NAME` (ER_WORK redirect + symlinks, src untouched). Output: `prob_test_France_NAME.parquet` (s1, sx, p; every
  pruner-kept pair) + `stats_NAME.json`. The models are #6's (m2_big, pruner τ 0.03). France has no CE.
  - Quiet box ~12 min / 2.6 GB. At load 20: 90 min, mostly blocking.
  - Helpers: `idf_vecs(v, share)`, `dom_vecs(v)`, `city_states(r1)`, `fill_state(rx, city)`, `state_display(addr, country)`,
    `append_state(raw, add)`.
- `world.py NAME --country C --states a,b,c [--blank F] [--fill] [--idf pooled|loco|loco-own] [--share S] [--from W]`: the
  labelled 3-state proxy. Every world S1 and S2/S3 competes, fold-0 S1 are scored with M2 p only.
  Output: `world_NAME.json` + `curve_NAME.csv` + a ledger row. 11 min / 3.7 GB peak (selection), 2.3 GB with `--from`.
- `compare.py NAME…`: label-free profile vs #6: empty rate / matches per S1 at t, t_P6, t_eq_d4, band, churn. Logs `profile-NAME`.
- `combine.py SRC PROB T OUT [--country France]`: swaps one country's rows of a staged submission (both files). Streams, 1 GB.
- `shift.py [t]`: the feature-shift ranking (`data/work/p14/shift.csv`).
- Files in `data/work/p14/`: `prob_test_France_{sub6,idf144,idffill,fill,fr6,idfdom}.parquet` (s1, sx, p), `stats_*.json`,
  `world_*.json` + `curve_*.csv` + `prob_world_*.parquet` (proxies), `log_*.txt`. All were verified to open with full row counts
  after the power loss. Scratch work dirs are deleted, and any variant is rebuilt by its command above.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Open question 4.** The France-aware IDF reads unlabelled test S2/S3 text (168k France rows), and so does the fill (test S1 city → state).
  This is the same category as statistics the pipeline already takes from test: chain counts over the test S1 pool, candidate-list
  context, and P6's threshold rule on test p. No labels, no external data, no new dependency (sklearn TfidfTransformer).
  Train-only fallback: `--idf-dom` → `prob_test_France_idfdom.parquet`. French domain words already listed in normalize.py get
  US street-type IDF; city names stay at max IDF. It gets about 40% of the label-free shift (band 11.9 → 11.3% vs 9.5%). Upload it at
  t 0.98 only if the organisers rule out test statistics.
- **The proxy is India under a US-only IDF.** Its mismatch is milder than France's (many India tokens occur in US text), so the
  France gain may be larger or smaller than 1.2 pt. The LB decides; A isolates it.
- **The share 0.144 is a choice:** France's share of test S2/S3. On the proxy it leaves 0.0006 of the 0.0123 on the table vs 40%.
  A larger share was not tried on France.
- **Threads:** re-runs after the reboot used 8 blocking threads (15 before). sp_matmul_topn is exact, so only exact score ties at the
  top-k boundary could differ.
- **The pruner and M2 were trained on train-IDF features.** France under the new IDF has feature ranges like US/India (that is the
  point), but it is still an unseen country: the P6 LOCO loss (5–7 pt) mostly remains, so thresholds still matter.

## Open questions   (for the master; each with your best guess)
- **Does the IDF fix transfer to the LB?** Best guess: yes, +0.001–0.002 LB (France +0.7–1.3 pt), from the proxy's 1.2 pt and France's
  band shrinking by the same relative amount (11.9 → 9.5% vs 5.9 → 3.9%).
- **Is fill worth it?** No: the whole missing-state loss is 0.0002 on the US proxy, and fill recovers about a third of it, within noise.
- **Should US/India get a per-country IDF too?** Best guess: no. Their tokens are in the train pool at 40–60% shares, and the proxy's pooled
  run is the best of the three.
