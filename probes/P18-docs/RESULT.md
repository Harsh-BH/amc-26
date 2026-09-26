# P18-docs — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- Four docs now describe the current #8c / #9 pipeline: `Documentation_template.md` (rewritten), `APPROACH.md`, and
  `README.md` / `SETUP.md` (drift fixed). The pipeline is blocking → pruner (4.83 cands/S1) → m2_big → bge CE → US
  stacker → decision, with France's five options.
- **Package dry run passes.** `package.py --team Yeti --out output/sub8c` builds a **97.4 MB, 26-file**
  `Yeti_submission.zip` (227 MB uncompressed) in 24 s. The layout matches PROBLEM.md, and the unzipped TSVs are
  byte-identical to `output/sub8c` (validator PASS).
- **Recommendation:** merge. After the 26 Sep uploads, fill the `TBD` cells (France final option + LB scores). Two
  things are missing from the zip: the bge training code, and bge's model files on Drive (see Recommendation).

## Recommendation   (exactly what the master should adopt: module, function signature, config)
1. **Merge the four docs.** Only these files changed, plus `probes/P18-docs/RESULT.md`.
2. **Fill the TBDs after the 26 Sep LB.** Search `TBD` in `Documentation_template.md`. It appears in:
   - 4.6: the France options table's "public LB" column, and the line **"Final France option: TBD by 26 Sep LB."**;
   - 5: "Public leaderboard … final submission's score is TBD";
   - 5.2: the #8c / #9a / #9b / #9s / final rows.

   Then search `TBD` in `APPROACH.md` section 8 (#8c, #9a/#9b/#9s, final). If the final is a `sub9*` recipe, also:
   - update 5.1's last row (India/US is unchanged; France has no holdout number);
   - update Appendix B's France profile table;
   - if the final is `sub9a` / `sub9d`, update 3's candidate table: France's kept candidates become 5.03 / S1 instead
     of 5.14 (P14).
3. **Build the final zip from the final output folder:**
   `.venv/bin/python code/business_entity_resolution/src/package.py --team Yeti --out output/<final> --dest .`
   - It takes `Documentation_template.md` from the repo root. It must not contain the string `[Your Team Name]`; it
     doesn't.
4. **Ship the bge training code in the zip (for the 8–9 Oct code review).**
   - Today `src/` can *load* and *score* bge (`rerank.load_bge` / `score_bge`), but the fine-tuning lives in
     `probes/P13-aws-rerank/solution_p13.py`, which `package.py` does not include. `run.ensure_ce` points reviewers at a
     path that is not in the zip.
   - Smallest fix: copy `solution_p13.py` into `src/` (e.g. `src/train_bge_sagemaker.py`), since `package.py` zips
     `src/*.py`. Then change the README sentence "fine-tuned on AWS by P13's SageMaker job
     (`probes/P13-aws-rerank/solution_p13.py` …)" to the new path.
5. **Upload the #8c model files to Drive.** Drive `amc26-offload/models/` has `p4`, `p4b`, `p10` and `p11` (checked
   read-only with `rclone lsl`), but not these:
   - `p13/ckpt/ce_bge_pool.pt` (624 MB)
   - `p13/r/` (38 MB)
   - `p17/stack_sup-bge-l21.0_US.txt`

   Without them a teammate cannot run paths A/B for #8c. SETUP still marks them "upload pending". I fixed p11's stale
   "pending" mark. All six local sha256 values match SETUP's table.
6. **Stale comments in files outside this probe's edit scope** (src/ is off limits, and the brief lists only the docs).
   All of them ship in the zip:
   - `requirements.txt` line 2 still says "One pretrained model: intfloat/multilingual-e5-small". Suggested:
     `# Pretrained models: BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M, default CE, fine-tuned on AWS) and`
     `# intfloat/multilingual-e5-small (MIT, 118M, #4-#7's CE, fine-tuned by rerank.py train).`
   - `src/run.py` docstring line 3: "submission #5 (--recipe sub4: #4)" → "submission #8c (--recipe sub9a|…|sub4)".
   - `src/model.py` docstring: "Submission #5: booster …".
   - `src/rerank.py` header: "intfloat/multilingual-e5-small (MIT, 118M params)", which should also mention bge.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| `P18-docs/package-dry-run-sub8c` | test | `Yeti_submission.zip` 97.4 MB, 26 files, 227.1 MB uncompressed, sha256 `9ecbdba2…`; built in 24 s; `package.py --selftest` ok; `zipfile.testzip()` clean | keep |
| (same row) | test | unzipped `matching_results.tsv` / `candidate_pairs.tsv` `cmp`-identical to `output/sub8c`. Official validator **PASS**: 1,732,544 rows each; 102,724 empty answers, 22,446 S1 with no candidate | keep |
| (docs check) | — | every Markdown table in the four docs has a consistent column count (script check). One pre-existing break (APPROACH example table) was fixed | keep |
| (numbers check) | — | every number in the new docs was taken from the pipe ledger, `SUBMISSIONS.md`, `probes/P4…P17/RESULT.md` or `MERGE-*.md`. Examples: #8c 0.98990 / half 1 0.98956 = pipe `p9-model` 26 Sep 07:45; LB 0.980017 / 0.849163 / 0.981024 / 0.981142 / 0.980954 = pipe rows | keep |

**Zip contents** (layout = PROBLEM.md "Final package"):
```
output/matching_results.tsv          96.6 MB  (41.5 MB compressed)
output/candidate_pairs.tsv          130.2 MB  (55.8 MB compressed)
code/business_entity_resolution/README.md, requirements.txt
code/business_entity_resolution/src/ block.py features.py io_tsv.py metric.py model.py normalize.py package.py pairs.py
    prune.py rerank.py rules.py run.py s0_exact.py stack.py submit_probs.py test_pipeline.py test_rules.py
    test_rules_cands.py train_matcher.py translit.py worlds.py            (21 files, 227 KB)
Documentation_template.md            29 KB
```
Not in the zip, by design: model files, `data/`, `probes/`, `ledger/`, and the root docs APPROACH/SETUP/SUBMISSIONS.
The zip is at `~/.cache/p18-scratch/Yeti_submission.zip` (scratch; delete it freely). Nothing was uploaded.

**What changed in each doc:**
- **`Documentation_template.md`** keeps the official template's headings (1–6, Appendix A "Code Artefacts",
  B "Additional Results"). It is rewritten to the #8c pipeline:
  - the pruner in 3, with a candidate table and the τ curve;
  - 4.1–4.6: m2_big, bge with e5 as history, France's scorer, the US stacker, the threshold, and France's LB-validated
    label-free threshold with the five-option table (final = "TBD by 26 Sep LB");
  - 5.1: an ablation with every step's full-corpus holdout number (all S1 and report half, paired Δ, ledger names);
  - 5.2: every LB upload, with the 26 Sep rows TBD;
  - 5.4: a what-failed table (P2, P4, P5, P10, P11, P12, P13, P14, P15, P16, P17, Qwen);
  - Appendix B: the models + licences table, the France profiles, hardware / AWS cost.
- **`APPROACH.md`:**
  - a probes table P1–P18;
  - discoveries 7 (house-number siblings) and 8 (France IDF + missing state);
  - pipeline steps 4 (pruner) and 7 (US check);
  - a new section 7, "What happened on 25–26 September" (LB France diagnosis, the stricter bar, pruner, bge on AWS,
    siblings, the IDF fix);
  - the results table through #8c / #9 (TBD);
  - new dropped ideas and problems, 26 Sep plan, glossary.
- **`README.md`:**
  - intro and setup now name both pretrained models, with the optional bge base download;
  - "rebuilds every step" now excepts the bge checkpoint;
  - the CE runtime row gains bge timings;
  - the package section uses `--team Yeti`, and its artefact list covers bge (624 MB, AWS-trained, not retrained) and
    the stacker.
- **`SETUP.md`:**
  - paths B/C timings (with the pruner, B ~2–2.5 h; C needs the bge checkpoint);
  - the models folder is ~820 MB, not 110 MB, and p11 is no longer marked "pending";
  - install adds the bge base (optional, `HF_HUB_DISABLE_XET=1` note);
  - the holdout commands add `--ce-missing fallback`, which a laptop without the bge base weights needs;
  - the time table covers bge / the stacker;
  - France recipes are in "other commands", and the status line points at the 26 Sep queue.

## What failed      (approaches tried and dropped, with the number that killed them)
- Nothing failed. One first build attempt printed nothing, because its output was piped through a `grep`; the rerun
  printed "structure ok". It was not a build error.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- No code. The package build is `package.py --team Yeti --out <dir> --dest <dir>`: 24 s, < 0.3 GB, zip ≈ 97 MB for #8c.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Assumption: final ranking uses the private LB.** The methodology doc presents the public LB as a diagnostic, and
  says so openly: four France-only uploads chose France's threshold. That is the one parameter with no labels.
  Reviewers may read this as LB tuning. The doc frames it as calibration of an unlabelled country with a label-free
  rule for any *other* unseen label, which is accurate.
- **`sub9a` uses unlabelled test text** (PROBLEM.md open question 4). The doc flags it and names `sub9d` as the
  train-only fallback.
- **The holdout 0.98990 is partly in-sample.** The US stacker and the blend w were fitted on half 0. The doc says the
  report half (0.98956) is the honest figure, and that competitor band pairs keep the e5 blend (`--ce-missing fallback`).
- **Numbers that mix yardsticks.** P13's +0.00059 was measured on unpruned lists and P17's +0.00055 on the pruned list;
  5.1 gives both. #8a has no all-S1 run, so its row shows only Δ.
- SETUP path timings for B/C were **re-derived from the README's measured per-step times, not re-run**. Examples:
  pruned M2 scoring 3 + 18 + 5 min; bge scoring ~2 h on the RTX 3050, which is P13's FLOP estimate, not measured.

## Open questions   (for the master; each with your best guess)
- **Copy `solution_p13.py` into `src/` before the final zip?** Best guess: yes (Recommendation 4). The code review will
  look for how the 568M CE was trained, and the zip currently can't show it.
- **Which France option is final?** The docs say "TBD by 26 Sep LB". Best guess from the evidence: `sub9a` (IDF fix) if
  its LB delta vs #8c is ≥ +0.0002, stacked with `sib` if that also wins. The doc's France table is set up so that each
  delta reads directly.
- **PDF or .md for the doc?** Best guess: keep .md. `package.py` accepts both, and PROBLEM.md allows either; the tables
  render on GitHub/Unstop viewers.
