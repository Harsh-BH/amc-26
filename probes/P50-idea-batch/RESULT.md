# P50-idea-batch — RESULT (final, 09:20; P50 also served as the master's final-stack integrator 04:50–09:20)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **None of the six ideas passes its gate.**
  - A (hn suffixes) and B (legal forms) are already handled by the parser.
  - C (France exclusivity veto) is ≤ +0.00006 LOCO, and France's vetoed pairs are CE-endorsed (r median 0.996).
  - D (CE order TTA) collapses (swap AUC 0.66). F (20 seeds) is ±0.
  - E (full-holdout refit) is +0.000039 ± 0.000019 at 1.5× data: positive everywhere, below the gate. The master kept it: sub54's India/US is this refit + zoo columns.
- **As integrator I built and validated (--check-ids PASS) the final candidates:**
  - `sub_iu54_frgc70` (the default final, zipped by P43);
  - `sub_iu54_best` (upload #2: Qwen France + P55 no-rescue veto + P62 acronym + dup, expected ≈ +0.00035);
  - fallbacks `sub_iu54_fr38`, `sub_iu54_frgc70_acro`, `sub_iu54_fr38_55n`, `sub_iu54_fr55b(_norescue)`, `sub_clean`, `sub38_clean_p55`.
- Tooling left for the master: `asm50.py` (France stage ledger + subset assembler; it reproduces #54 exactly), `p55n.py`, `rm_pairs.py`, `acro_fit.py`, `splice_lines.py`, `diffc.py`, `e_ship.py` / `e_zoo.py` (combiner refit + test scoring).

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Upload order** (as in WAKEUP.md):
  1. `output/sub_iu54_frgc70` (default final, md5 7f559944…);
  2. `output/sub_iu54_best` (matching 24176ee789f081191deef58cc3cc519b, candidates a8b425b0…, validator p50-iu54best-0722). `run.py --recipe sub_iu54_best` reproduces it byte for byte (P43). If its LB beats #1, it becomes the final; tell P43 to build its small-candidate zip (~25 min).
- **If #2 disappoints, isolate the France pieces with the staged single-change fallbacks:**
  - `sub_iu54_fr38`: Qwen only, 0a8b2a1f…, expected +0.0002;
  - `sub_iu54_frgc70_acro`: acronym only, eb150488…, +0.00015;
  - `sub_iu54_fr55b_norescue`: P55 veto without the rescue stage, b4878e54…, small;
  - `sub_iu54_fr38_55n`: Qwen + that veto, a4689807….
  - `sub_iu54_fr55b` (6c8cd096…) empties 154 S1, all of them gc70's LB-proven rescue90 pairs. Don't prefer it.
- **Do not adopt:** C (France exclusivity veto), D (CE order swap), F (more seeds), a hn-suffix veto, LEGAL changes, or P52's lower bars on top of P55 (never jointly measured; 41% of its adds have P55 c < .6).
- **E / zoo:** they are already inside sub54's India/US (boosters `data/work/p50/fullzoo_s{0..4}.txt`, columns te44.FINAL + l_r_all + 16 `z_*`). P43 has ported them to src.

## Evidence         (table: ledger name | world | key metrics | verdict)
| idea | ledger name | world | key metrics | verdict |
|---|---|---|---|---|
| A | A-hn-suffix-range | test France (label-free) + holdout US/India | Parser already maps 12 bis/12bis/12b/12-14/12 à 14/12/14 → 12. France suffix share 6.3% S1 / 5.7% SX. hn-equal pairs with the same suffix on both sides: 0.927 accepted (no-suffix 0.895); one-sided bis: 0.04 accepted, ~90% at p<.5. Range with the other end = the other hn: 27 chain pairs (10 accepted, all p≥.99). Veto label check: one-sided / conflicting suffix on hn-equal pairs, #34 accepted precision India 0.9991 / US 0.9997 | drop |
| B | B-legal-form-coverage | test France (label-free) | Generator forms SARL/SAS/EURL/SA/SASU/SCI/EI/SNC (+ dotted, bracketed, SÀRL) all fold into LEGAL. Legal tokens left in cores: 69 S1 / 476 SX (core fallback). Typo-like last tokens 0.11%, mostly real words | drop |
| C | C-fr-exclusivity-veto | LOCO (P47 frames, P42 recipe, 2 analogues × 2 directions) | Best both-positive cell (chain, m ≥ .3, bar .99): +0.000029 ± 0.000019 US→India / +0.000062 ± 0.000019 India→US. Best India→US cell +0.000114 is −0.000012 US→India. Removed precision 0.35–0.55 | drop (gate +0.0003) |
| C | C-veto-ce-proxy | LOCO + test France | bge r of the removed pairs. LOCO true removals median 0.55 / 0.50, false 0.45 / 0.41. France's 1,385 vetoed pairs median **0.996** (79% ≥ 0.8), vs France's kept band pairs 0.84. Competitor-mass pairs are 0.34% of accepted France pairs vs 0.02–0.03% in LOCO | drop (France is a different population) |
| D | D-ce-order-tta | holdout, 20k band argmax pairs (g5.xlarge, 66 s) | Swapped (SX, S1) AUC 0.659 vs 0.983 (India 0.75, US 0.61); logit corr 0.34; average 0.875; stacking the swap +0.00002 AUC. Re-scoring jitter vs r_all up to 0.05 | drop |
| E | E-learning-curve | holdout halves | vs #34 at 25/50/75/100% of a half: h0→h1 +0.000492/+0.000535/+0.000572/+0.000656; h1→h0 +0.000343/+0.000450/+0.000441/+0.000505. 100% vs 50%: +0.000121 ± 0.000041 / +0.000055 ± 0.000034 | keep (curve rising) |
| E | E-trainsize-4fold-1.5x-vs-1x | holdout, 4 S1 folds, every S1 reported once | **+0.000039 ± 0.000019 (2.1 SE)**; India +0.000019, US +0.000078; half0 +0.000057, half1 +0.000021; folds +0.000020 / +0.000036 / +0.000067 / +0.000034 | below gate (standard practice) |
| F | F-seed-bag-5-vs-20 | holdout halves | 20 vs 5 seeds −0.000014 ± 0.000018 / −0.000004 ± 0.000017; 10 vs 5 +0.000008 / −0.000002 | drop |
| zoo | zoo-as-combiner-cols | holdout halves | 16 P45 members as extra sub44r-combiner columns: +0.000025 ± 0.000029 / +0.000023 ± 0.000029 (US −0.000007 in h0→h1); stats-only arm +0.000010 / +0.000019 | inconclusive (in sub52_max only) |
| stage | stage-sub50e | test | Rebuilt test frames + shipped boosters reproduce #48 byte for byte. 3,393 India/US lines change; profile max test/holdout ratio 1.44; PASS (p50-val50e-0927-0451, md5 dd148cc9…) | staged |
| stage | stage-sub52_fr-then-deleted | test | sub50e + cls80 + C veto: PASS (md5 fabd7f79…); deleted for disk, C unsafe | dropped |

**Integrator builds** (validator = official script with --check-ids on SageMaker ml.m5.xlarge):

| folder | chain | vs reference | validator |
|---|---|---|---|
| sub52_max | #50e India/US + 16 zoo columns (full refit); France #48 + cls80 (+1,925) | India/US 2,135 lines vs #50e; profile PASS (max 1.45) | PASS, md5 b96ce2f1 |
| sub_all_v2 | = #54 (hard links) | — | P54's PASS |
| sub_all_v3 (hold) | #52_max → P55 (−3,198 / +4,120) → P54 u .995 (+2,770) → dup (+49) | India/US over cap: 37 lines, all pre-existing; France 2, from dup | PASS, md5 7f27d610 |
| **sub_clean** | build38 clean(bge): recipe (hn .80) → gib +1,434 → P48 −1,160 → P54 u .99 +8,539 → cls80 +1,037 → P55 −3,197 / +1,129 → dup +49 | vs #54: France +1,149 / −11,059, 303 S1 emptied; vs gc70: 195 S1 empty that gc70 filled | **PASS, md5 4e0e33df** |
| sub_clean_iu34 | #clean France + gc70 India/US + dup (+10) | vs #clean: India +3,266 / −2,713, US +2,846 / −2,555 | PASS, md5 6acbd1fd (offloaded) |
| sub38_clean_p55 | P38 sub38_clean (Qwen) → P55 (−2,147 / +2,328) → dup (+50) | vs #clean: France +2,867 / −10,319 | PASS, md5 24733e0d |
| sub_iu54_fr38 | default (sub54 India/US + gc70 France) with France := P38's sub38 (Qwen CE swap) | 6,885 France lines, +3,685 / −3,431 | PASS, md5 0a8b2a1f |
| sub_iu54_fr55b | default − P55 band veto (all stages) | −805 pairs, 154 S1 emptied (= rescue90 pairs) | PASS, md5 6c8cd096 |
| sub_iu54_fr55b_norescue | default − P55 band veto without its rescue stage (p55_veto_norescue.parquet, 3,044 pairs) | −652, 3 emptied | PASS, md5 b4878e54 |
| sub_iu54_fr38_55n | fr38 − the same list (bge-chain list applied where present; ~480 Qwen-lifted low-c pairs deliberately kept) | −619, 4 emptied | PASS, md5 a4689807 |
| sub_iu54_frgc70_acro | default + P62 acronym set re-filtered (acro_fit: SX owned / caps) | +3,559 (2,643 new candidates), 81 filled | PASS, md5 eb150488 / cand d1d19094 |
| **sub_iu54_best** | fr38_55n + acronym re-filtered (+3,538) + dup (+44) | vs default France +7,104 / −3,887, 244 emptied / 218 filled | **PASS, md5 24176ee7 / cand a8b425b0** |

Conflict checks run before stacking:
- P55's combiner c on other pieces' adds, share below .6:
  - P52 cal: 608 of 1,475 (median .68);
  - P54 rescue u .99: 2,073 of 7,629 (median .91);
  - cls80: 264 of 1,925 (median .95).
- P54 dup re-adds pairs other stages dropped:
  - 8 India/US pairs the combiner scored < .71;
  - 3 P55-vetoed France pairs;
  - 1 pair removed between #46_on43 and #48.
- The C veto would hit 52 cls80 adds (it is restricted to recipe pairs, so 0 in practice).

## What failed      (approaches tried and dropped, with the number that killed them)
- **A, hn-suffix veto** (drop hn-equal pairs whose bis/ter/letter suffix is one-sided or differs): these pairs are 0.999 true among the accepted on US/India.
- **B:** nothing missing from LEGAL.
- **C:** the LOCO gain is ≤ +0.00006 India→US. On France the rule would remove 1,385 CE-endorsed pairs (1,327 S1, 53 emptied).
- **D:** the CE is order-sensitive. The swapped order is near chance for US (0.61), and averaging halves the gain.
- **F:** more seeds are ±0.00002.
- **Subtractive clean build:** removing hn extras below .9 from #54 was superseded by the master's proper rebuild with hn kept at .80. Subtraction also under-fills, because the freed SX / caps are not re-admitted.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `a_hn.py`, `a_loco.py`: A counts + label check (5 min, 0.9 GB).
- `b_legal.py`, `b_typo.py`: B (40 s).
- `c_excl.py` (LOCO grid, 2 min), `c_proxy.py` (CE proxy), `c_build.py BASE OUT M BAR [ADDS_FROM]` (France veto builder, 11 s).
- `d_build.py`, `d_eval.py`: D channel + readout. The job is P13's `bge_ckpt` on `amc26-p50/data/ce/`.
- `ef.py` (E curve + F, 4.5 min, 1.1 GB), `e_cv.py` (4-fold 1.5× vs 1×, 3 min).
- `e_ship.py build c | fit | probs TAG PREFIX | splice TAG SRC OUT | profile TAG`: the te44 path pointed at `data/work/p50`. Build takes 3 min/country and peaks at 1.9 GB.
- `zoo50.py` (holdout zoo arms), `zoo_test.py c k n | merge` (streams the members' test preds), `e_zoo.py fit | probs` (chunked predict).
- `asm50.py ledger | build OUT --iu FOLDER --fr STAGES [--dup] [--tmpl F] | test`:
  - the ledger tags every France chain pair (france.npz rows) by stage: base / rescue / gc / hn (hn90 / hn95), gib / xf / tier46, veto48, cls80, resc54 (+ hn-extras-type flag), p55veto / p55add, resc995, p52, dup;
  - build is a subset assembly with India/US from any folder;
  - rebuilding #54 gives 0 differing lines.
- `splice_lines.py OUT BASE SRC India,US`, `diffc.py NEW REF...` (per-country diff), `overlap.py`, `cc_check.py`, `caps.py`.
- `p55n.py OUT.parquet v=.6 scope=band skip=2`: P55's veto as a pair list with stages skipped (+ `p55_c_all.parquet`, c for every frame pair).
- `rm_pairs.py BASE OUT PAIRS.parquet`: remove a pair list from any base. `acro_fit.py BASE OUT.parquet`: P62's acronym set re-filtered on a base (SX owned / caps, T2 → T1 → T3). Apply with `probes/P54-fresh-batch/build54.py BASE OUT adds.parquet [--nodup]`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Everything is LightGBM / rapidfuzz / the existing CE; no new models or external data. France thresholds stay label-free (LOCO / audits).
- E / zoo inherited the combiner's holdout caveat: competitor rows carry in-sample p. P57's honest frame and P62's test-density clones confirmed the combiner (+0.0004–0.0006), so the caveat is resolved.
- **sub_clean's expected gain is an estimate** from P38's audit precisions (P46 0.08, P40 0.19). #54 under-delivered its LOCO expectation 4–5×, so treat LOCO-based expectations (P55's +0.0003–0.0004) as upper bounds.
- 195 France S1 are empty in sub_clean where gc70 had matches (P55 and P48 vetoes). Per the master, they were reported, not restored.
- sub_iu54_best's acronym adds are validated on train only (0.997 / 0.9997). On France the only CE audit is a non-random 113-pair subset (98% Qwen r ≥ .5). Its 2,638 new candidates need the pipeline's own acronym block (P43 ported it).

## Open questions   (for the master; each with your best guess)
- Does sub_iu54_best beat sub_iu54_frgc70 on the LB? Best guess: yes, by +0.0002–0.0004. If not, upload the single-change fallbacks (fr38, then frgc70_acro) to find the piece that hurts.
- P52's lower bars were never measured on top of P55. Best guess: leave them out (41% of its adds have c < .6).
- P54 rescue on top of P55: u .995 was P54's own LOCO pick. Best guess: it's irrelevant now, since the default France is gc70 without P54.
