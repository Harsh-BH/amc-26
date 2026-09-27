# P43-final-package — RESULT (27 Sep 14:50: third task, the final = sub66_final; earlier tasks below)
## TL;DR — third task (27 Sep 13:45 → 14:50): P64 + P66 in src, the final zip on sub66_final (LB 0.987391)
- **`run.py --recipe sub66_final` (now the default) rebuilds `output/sub66_final/matching_results.tsv` byte for byte**
  (md5 c6246285…, 0 differing lines). Candidates: 1 France line differs, the #54-file dup candidate (same as `sub_iu54_best`).
  - Stage counts equal the staged builds: P66 class 10,403 of 40,822 address-less argmax pairs; 1,900 adds on #37's base
    (= `output/sub66`); P64 veto 344 (= `output/sub64`); 1,884 adds fit on 1,810 S1; 0 P64 conflicts; second dup pass 0.
  - New code: `france.lf_sig / lf_class / lf_veto` (P64, verbatim `lf.py` + P3 text names), `france.xfc` (verbatim
    `rule66.xfc`), `france.xfc_adds` (rule40.add = `gc_add(..., sib | ~cls, t, ones, empty=True)`), `run.xfc_frame`
    (#37's frame: bge post-rules p, its argmax, `complete(t, 0.90, 0.70)`) and `run.xfc_fit` (fr66 build's fit: SX
    owned → drop, caps on the line, P64 class 4 on raw names, appended sorted after the line). Recipes `sub64`,
    `sub64_frgc70`, `sub66`, `sub66_final`. Only `sub66_final` was run in full (28 min, paused by memguard ~10 min).
  - Pitfall fixed: `np.isin(object_array, list)` loops per value (≈10^11 compares); use `pd.Series.isin`.
- **Final zip = `output/sub66_final_small`** (md5 dd580e52…, 30 files, 93.9 MB; official validator `--check-ids` PASS
  on the unzipped zip, SageMaker p43-val-zip66; backed up to S3 + Drive `final/`): the sub66_final matching file +
  `sub_iu54_best_small`'s candidates (7,660,700, 4.42/S1, md5 759a20e6…). That is what `--drop-sibling-cands` writes for
  this recipe: P64 / P66 / the second dup pass never touch run.py's candidate list (P66's adds are non-sibling argmax
  pairs, which the decoy stage keeps). Checked: all 5,805,047 matched pairs are inside it; not re-run in full.
  - code.zip of `output/sub64` and `output/sub66_final` rebuilt from main's src (hard link removed first).
  - Old final zip (sub_iu54_frgc70) kept as `final/archive/Yeti_submission_sub_iu54_frgc70.zip` on S3 and Drive.

## TL;DR — second task (27 Sep 06:30 → 08:50)
- **The final zip is built, validated and backed up** (27 Sep 08:50). `output/final/Yeti_submission.zip` (md5 96ef3224…,
  93.8 MB, 30 files, the rulings' layout) holds `output/sub_iu54_frgc70_small`.
  - Its `matching_results.tsv` is **byte-identical to the master's final `output/sub_iu54_frgc70`** (md5 7f559944…).
  - The candidate set is small: 7,658,108 pairs, 4.42 per S1.
  - The official validator with `--check-ids` passes on the folder and on the unzipped zip.
  - Backups: S3 `amc26-offload/final/` (zip, folder, standby, boosters, stage inputs; older zips in `archive/`) and
    `gdrive:amc26-offload/final/` (zip, folder, standby). Copy only; nothing deleted.
- **`run.py` (default recipe `sub_iu54_frgc70`) rebuilds every current candidate's matching file byte for byte from cached
  inputs:** `sub_iu54_frgc70`, `sub_iu54_best`, `sub_iu54_fr38_55n` and `sub_clean`.
  - `sub_clean_iu34` rebuilds as ID sets, 0 lines.
  - `sub54` is 2 France lines off: it descends from the union #43_r90, which is not a sequential rule.
  - New code: `src/iu_stack.py` (India/US combiner, by a subagent), `france.chain` (27 Sep France stages), P62's
    acronym block, `run.dup_add`. Country logic stays keyed on `unseen()`.
- **Standby: `output/sub_iu54_best_small`** has the same matching file as `sub_iu54_best` and 7,660,700 candidates;
  validator PASS. If the LB makes `sub_iu54_best` the final, re-zip in ~5 min:
  `package.py --team Yeti --out output/sub_iu54_best_small --dest output/final`, then `solution_p43.py` on SageMaker,
  then copy to S3 and Drive. Main = c1f94bb, pushed.

### Second task (27 Sep 06:30 → 07:45): what was ported and verified
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| final-zip-sub_iu54_frgc70 | test | matching **byte-identical** to output/sub_iu54_frgc70; small candidates 7,658,108 (France 4.48 / India 4.75 / US 4.00 per S1); validator `--check-ids` PASS (folder: p43-val-final-0730; unzipped zip: p43-val-zipfinal-0740); zip = 30 files, 93.8 MB | keep |
| (subagent iu-port) sub52_iu | test | `iu_stack` p'' bit-identical to P50's fullzoo probs on every argmax pair (India 3,685,878 / US 2,910,231). India/US lines = sub52_max's (0 lines; vs sub54 only the 14 dup copies). `--iu a10` = sub37_gc70 byte for byte. 580 s, 1.41 GB | keep |
| fr-chain-port-vs-sub_clean_fr | test | `france.chain` (ORDER_CLEAN) on P42's frame = sub_clean_fr, 0 S1 differ. Stage counts = P38's: recipe 835,532 / brand +1,434 / veto −1,160 / rescue +8,539 / class +1,037. + P55 (−3,198 / +1,129) = sub_clean but for its 49 dup copies | keep |
| sub_clean_iu34-recipe-set-repro | test | full run.py run = s3 sub_clean_iu34: 0 differing lines as sets, both files; dup France 49 / India 7 / US 3. 11 min, 1.32 GB | keep |
| sub_iu54_fr38_55n-recipe-byte-repro | test | matching byte-identical; Qwen rescue +1,043 / gc +9,155 (= P38); P55 no-rescue veto list 3,044, 619 present (= P50) | keep |
| sub_iu54_best-recipe-byte-repro | test | matching **byte-identical** (md5 24176ee7…). Acronym: block 102,236 one-token SX → 11,495 rule pairs → 11,371 unambiguous → change set 3,590 (on #clean's lines, as staged) → 3,538 kept on the base, 2,638 new candidates (= P50). France dup +44. Candidates: 1 France line differs (a #54-file dup candidate this lineage does not produce). 13 min, 1.42 GB | keep |
| (chaincheck s54) | test | ORDER_54 (brand ∪, xf ∪, P46 swap, #48's veto on P42's frame, class, fr54 rescue): 34 France S1 differ from sub54 = 32 dup copies + 2 pairs (P38's on43 rebuild of the union #43_r90 is 2 rows short too) | keep (reported) |

Code changed (main): `src/france.py` (chain + P42/P47/P54/P55 stage functions, veto54), `src/run.py` (recipes sub_iu54_frgc70
[default], sub_clean, sub_clean_iu34, sub54, sub52_iu; france31 chain inputs; dup_add + ordered tails in write_lists; iu hook),
`src/iu_stack.py` (new, subagent), `src/a10.py` (frame() split, subagent), `src/model.py` (tag reads `<file>.tag`: the
bge ckpt was offloaded, `data/work/p13/ckpt/ce_bge_pool.pt.tag` = 3bd0b930). Docs: Documentation_template (4.4c, 4.9, LB rows,
appendix), APPROACH, SETUP, SUBMISSIONS, README — the final LB is a `[FINAL LB]` slot.

---
# Night task (27 Sep 00:50 → 03:05)
## Recommendation   (exactly what the master should adopt: module, function signature, config)
**Code (already in main):**
- `run.py`:
  - The default recipe is `sub37` = `sub34_veto` + `rescue=0.90` + `gc=0.70`.
  - New recipes: `sub37_small`, `sub37_hn90`, `sub34`, `sub34_veto`, `sub41`, `sub41_hn90`, `sub42_both`, `sub40x`, `sub40x_t70`, `sub43_r90`, `sub46_on43`, `sub46g_on43`.
  - New flags for the unseen country (0 turns one off). They run in this order after argmax + t: `--rescue T`, `--gc T`, `--hn-extra T`, `--brand T`, `--noaddr T`, `--swap S_MAX`. Two more: `--veto TAU`, and `--t-post T` (post-rules at T, decision at the country's t).
  - `unseen()` = the test S1 labels minus the train S1 labels (France).
  - `a10_stage` and `ensure_a10` handle the a10 stacker.
- `a10.py` (new): P34's stacker.
- `france.py`:
  - `C` is set by run.py; the module names no country.
  - `complete(q1, qx, is2, p, am, acc, h1, hx, rescue_t, gc_t, hn_t, brand, noaddr, swap)` adds pairs and never removes them. It uses `rescue`, `gc_add`, and the masks below.
  - `brand_mask` (P41 + P36's syllables), `noaddr_mask` (P40), `swap_mask` + `lens_table` (P46).

**Uploads.** When a winner is chosen, rebuild it with run.py and upload that file. Then build the zip from the `_small` version:
```bash
.venv/bin/python code/business_entity_resolution/src/run.py --split test --stage model --recipe <r> --out output/<r>          # upload this
.venv/bin/python code/business_entity_resolution/src/run.py --split test --stage model --recipe <r> --drop-sibling-cands \
    --out output/<r>_small --no-validate                                                    # same matching file, 4.42/S1
cmp output/<r>/matching_results.tsv output/<r>_small/matching_results.tsv
.venv/bin/python code/business_entity_resolution/src/package.py --team Yeti --out output/<r>_small --dest output/final
```
- Each run takes ~7 min with the caches (1.4 GB under a 1.5 GB cap).
- To validate with `--check-ids`, upload the files plus `validate_submission.py` to `s3://…/amc26-p43/<dir>` and run `probes/P43-final-package/solution_p43.py` via `sm_launch.py`. The docstring has the command. It takes 50 s on spot ml.m5.xlarge, and the test TSVs are already at `s3://…/amc26-p43/test`.

| staged (master / probe) | run.py | checked | upload |
|---|---|---|---|
| `sub37_gc70` (LB 0.986304) | `--recipe sub37` | byte-identical, both files | (uploaded) |
| `sub37_gc70_hn90` | `--recipe sub37_hn90` | byte-identical, both files | as staged |
| `sub41`, `sub41_hn90` | `--recipe sub41` / `sub41_hn90` | byte-identical, both files | as staged |
| `sub42_both` | `--recipe sub42_both` | byte-identical, both files (flag form) | as staged |
| `sub40x` | `--recipe sub40x` | ID sets identical (1,829 lines differ in ID order; P40 appends unsorted) | the run.py file |
| `sub43_r90` (union) | `--recipe sub43_r90` → **`output/sub43r90_recipe`** | union ⊂ recipe, +507 pairs, all in #37_gc70 (p ≥ 0.9868); validator `--check-ids` PASS | **`sub43r90_recipe`** |
| `sub46_on43` (union base) | `--recipe sub46_on43` → **`output/sub46on43_recipe`** | = P46's builder on `sub43r90_recipe`, byte for byte; validator `--check-ids` PASS | **`sub46on43_recipe`** |
| `sub42_both_on41`, `sub43_safe`, `sub46_on43safe` | not in src (unions) | — | rebuild as a recipe first |

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| (lineage, `linediff.py`) | test | sub31 → sub34: India 7,574 / US 5,274 lines. → sub34_veto: France 5,994. → sub34v_rescue90: France 1,205. → sub37_gc70: France 9,322. Each step touches one part only | — |
| sub37-recipe-byte-repro | test | `--recipe sub37` = `output/sub37_gc70`, **both files byte-identical** (md5 cdf47562… / b3ccba0d…).<br>tier 253,136 / bxneg 523,707 / veto 7,753 / cap 18 (= P31); rescue +1,205 S1; gc +10,084 pairs to 9,322 S1 (= P37); a10 on 3,685,878 India + 2,910,231 US argmax pairs.<br>12.5 min cold a10, 6 min cached; 1.40 GB | keep |
| sub37_small-final-candidates | test | matching cmp-identical to sub37_gc70.<br>7,658,102 pairs = **4.42/S1** (France 4.48 / India 4.75 / US 4.00); decoys dropped 170,080 / 125,346 / 416,972; 42,182 S1 have no candidate, all with empty answers.<br>**Validator `--check-ids` PASS** | keep |
| final-zip-validated / final-zip-v2-validated | test | `Yeti_submission.zip`: 93.8 MB, 29 members (`output/`, `code/business_entity_resolution/{src, README.md, requirements.txt}`, `Documentation_template.md`).<br>Unzipped TSVs cmp-identical to sub37_final_small; src = main's; 5 self-checks + parse pass from the unzipped copy.<br>**SageMaker unzip + validator `--check-ids` PASS** | keep |
| ensure_a10-refit-bit-identical | holdout | the `run.ensure_a10` refit is **byte-equal** to `data/work/p34/a10_h0.txt`; 84 s | keep |
| sub37_hn90-flag-byte-repro | test | = `output/sub37_gc70_hn90`, both files (+18,180 pairs / 17,222 S1) | keep |
| sub42_both-flags-byte-repro | test | `--t-country France=0.99 --t-post 0.9868 --rescue 0.97 --hn-extra 0.8` = `output/sub42_both`, both files. Without `--t-post`, 31 France lines differ | keep |
| sub41-brand-port-byte-repro | test | = `output/sub41`, both files (16,786 bare brand SX; +5,742 pairs / 5,641 S1). `sub41_hn90` is also byte-identical | keep |
| sub40x-noaddr-port-set-repro | test | ID sets = `output/sub40x` (+2,441 / 2,360 S1); candidates identical | keep |
| sub43_r90-recipe-vs-union | test | 496 France S1 differ. The union lacks 507 recipe pairs and has 0 the recipe lacks; all 507 are #37_gc70 pairs. The union's brand / address-less pairs were computed on gc70, so pairs dropped by t 0.99 were never re-offered to those rules | drop (the union) |
| sub46_on43-swap-port-byte-repro | test | = P46's builder on sub43r90_recipe, byte for byte (+3,185 pairs / 3,147 S1 = P46; lens words incl. développement) | keep |
| test_pipeline | dev slice + injected labels | `sub37_small --unseen France` (e5 CE): rules 0.8575, model 0.9966 (rescue / gc / decoy stages run). Default recipe: model 0.9968 | pass |

## What failed      (approaches tried and dropped, with the number that killed them)
- **The local official validator with `--check-ids` under the 1.5 GB cap:** it swap-thrashed (31 s of CPU in 6 min), and memguard pauses `.venv/bin/python` jobs. It now runs on SageMaker ml.m5.xlarge spot: 50 s per run, 5 runs.
- **`--t-country France=0.99` alone for sub42_both:** it re-runs the post-rules at 0.99, and 31 France lines differ. P42 decided at 0.99 on post-rules computed at 0.9868, hence `--t-post`.
- **Reproducing the master's union files (sub43_r90, sub46_on43) with a sequential recipe:** 496 France S1 short. Their levers were applied to different bases (see Evidence).

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `linediff.py A B [file] [sets]`: per-country differing lines, or differing ID sets, of two submission dirs. 20 s, 0.4 GB.
- `a10refit.py`: checks `run.ensure_a10` against `data/work/p34/a10_h0.txt`. 84 s, 1.4 GB.
- `solution_p43.py`: SageMaker job running the official validator with `--check-ids` on a zip or two TSVs. The train channel holds the validator and the files; the test channel is `s3://…/amc26-p43/test`.
- Outputs in main's `output/`:
  - `sub37_final_small/`, `final/Yeti_submission.zip`;
  - `sub43r90_recipe/`, `sub46on43_recipe/` (their candidate files are hard links to sub37_gc70's).
- S3, `s3://sagemaker-114872709824-ap-south-1/amc26-offload/final/`:
  - `Yeti_submission.zip` (md5 4a297f54…);
  - `archive/Yeti_submission_0216_894e62f.zip`;
  - `sub37_final_small/`;
  - `models/p34/a10_h0.txt`.
  - `rclone check`: 0 differences. Nothing was deleted anywhere.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **The default recipe changed from sub31 to sub37 in main.** A probe that calls `run.run` / `model_parts` with the default recipe now gets a10 plus rescue / gc. Probes that call `use("sub8c")` etc. are unaffected.
- **Byte-level reproducibility holds only for run.py-built files.** Probe builders (P40, P42's unions, the master's #43 / #46) append IDs unsorted or merge pair sets. If one of them wins the public LB as it is, the zip's code rebuilds it only as a set (sub40x) or as a superset (#43 / #46). That is why the table says to upload the `_recipe` versions.
- **"France" remains in identifiers and data.** It appears in `T_FRANCE_OVERRIDE`, `W_FRANCE`, `FRANCE`, the `france.py` module name and `--t-country France=`. It also appears in `normalize.py`'s hand-written `STATES["france"]` région list (allowed: state names) and in the `fr_departements` learner, which filters test rows to that label. No decision branch names a country; the docs say so.
- The S3 zip holds `sub37_final_small`, i.e. the current best-public. When a 27 Sep upload wins, rebuild the zip (command above).
- **Assumed:** merging into main was allowed ("merge to main only when every check passes"). I fast-forwarded, did not push, and wrote no probe files into commits (RESULT.md and the scripts here are untracked for the master).

## Open questions   (for the master; each with your best guess)
- **Upload the `_recipe` versions instead of the union files?** Yes. The difference is 507 pairs that #37_gc70 already had accepted on the LB. The zip then reproduces the upload byte for byte.
- **Push main?** Your call. P43 made 5 commits (2da4276 … 35ef4e3). `origin/main` already has the first three,
  so it is behind by 2 (226350c: P46 port, 35ef4e3: SUBMISSIONS row).
- **Is `data/work/p9/a10_test_*` (25 MB) needed in the offload?** No: it rebuilds in ~6 min from the a10 booster, which is on S3.
