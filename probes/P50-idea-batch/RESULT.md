# P50-idea-batch — RESULT (06:10; P50 is also the master's sub_all integrator, updated as pieces land)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **None of the six ideas passes its gate.** A (hn suffixes) and B (legal forms) are already handled by the parser. C (France exclusivity veto) is ≤ +0.00006 LOCO, and France's vetoed pairs are CE-endorsed (r median 0.996), so it would cut true pairs. D (CE order TTA) collapses (swap AUC 0.66). F (20 seeds) is ±0. E (full-holdout refit) is +0.000039 ± 0.000019 for 1.5× vs 1× data: positive in every fold and half, but below the gate.
- **Staged (all validator --check-ids PASS):**
  - `sub50e` = #48 + the full-holdout refit, ≈ +0.00006–0.00009;
  - `sub52_max` = + zoo columns + cls80;
  - `sub_all_v3` = + P55 + P54 u .995 + dup (ON HOLD);
  - **`sub_clean`** = the master's audit-driven rebuild of #54 without P46 / P40, + P55 + dup. It is the next upload.
- Integrator tooling: `asm50.py` (France stage ledger + subset assembler; it reproduces #54 exactly), `splice_lines.py`, `diffc.py`, `overlap.py`, `cc_check.py`, `caps.py`.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Upload `output/sub_clean`** (md5 4e0e33df…, PASS). Fallbacks: `sub_clean_iu34` (#34's India/US; use it if P57's honest frame rejects the combiner) and `sub38_clean_p55` (the Qwen France variant).
- **Do not adopt:** C (France exclusivity veto), D (CE order swap), F (more seeds), a hn-suffix veto, or LEGAL changes.
- **E (full-holdout refit):** ship the refit only if the combiner itself survives P57's audit. Boosters: `data/work/p50/full_s{0..4}.txt` (te44.FINAL + l_r_all) or `fullzoo_s{0..4}.txt` (+16 zoo columns). Apply them with `e_ship.py probs <tag> full` / `e_zoo.py probs`. The test frames are rebuilt in `data/work/p50/T_{India,US}.parquet`.

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
| sub_clean_iu34 | #clean France + gc70 India/US + dup (+10) | vs #clean: India +3,266 / −2,713, US +2,846 / −2,555 | submitted |
| sub38_clean_p55 | P38 sub38_clean (Qwen) → P55 (−2,147 / +2,328) → dup (+50) | vs #clean: France +2,867 / −10,319 | submitted |

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

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Everything is LightGBM / rapidfuzz / the existing CE; no new models or external data. France thresholds stay label-free (LOCO / audits).
- **E / zoo inherit the combiner's holdout caveat.** Competitor rows carry in-sample p (P44). P57 is re-testing the combiner out of sample; if it fails, sub_clean_iu34 is the fallback.
- **sub_clean's expected gain is an estimate** from P38's audit precisions (P46 0.08, P40 0.19). #54 under-delivered its LOCO expectation 4–5×, so treat LOCO-based expectations (P55's +0.0003–0.0004) as upper bounds.
- 195 France S1 are empty in sub_clean where gc70 had matches (P55 and P48 vetoes). Per the master, they were reported, not restored.

## Open questions   (for the master; each with your best guess)
- Ship the full-holdout refit (sub50e's boosters) under sub_clean's France? Best guess: yes if P57 passes the combiner (+0.00006–0.00009), else #34.
- P52's lower bars were never measured on top of P55. Best guess: leave them out (41% of its adds have c < .6).
- Should P54's rescue use u .995 when P55 follows (P54's own LOCO pick)? sub_clean uses u .99 as the master specified. Best guess: u .995 is safer (+0.00105 / +0.00097 measured on P55's decision).
