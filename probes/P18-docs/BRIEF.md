# P18-docs: bring the final documents and package up to the current pipeline
Read: SUBMISSIONS.md (every submission + LB score), all probes/*/RESULT.md of P9–P17, probes/MERGE-*.md, ledger
(`.venv/bin/python ledger.py show exp` / `show pipe`), code/business_entity_resolution/README.md, docs/PROBLEM.md
("Final package"), code/business_entity_resolution/src/package.py. Own worktree; edit only the docs listed below +
probes/P18-docs/.
Tasks:
1. **Documentation_template.md** (the methodology doc for the final zip, Team Yeti — keep the template headings):
   update to the current pipeline: blocking → learned pruner (4.83 cands/S1) → LightGBM m2_big → cross-encoder
   (bge-reranker-v2-m3, Apache-2.0 568M, fine-tuned on SageMaker A10G; e5-small history) → US house-number/support
   stacker → decision; France: label-free threshold via LB-validated profile, IDF fix / bge / sibling options (mark
   which one is final as "TBD by 26 Sep LB" — the master fills it). Ablation table with every step's full-corpus holdout
   number and every LB score from SUBMISSIONS.md. Models + licences table (e5-small MIT, bge-reranker-v2-m3 Apache-2.0,
   LightGBM MIT). What failed (P5 stack, P12 synthetic, P16 8-fold, capacity, state fill…). No page limit; clear.
2. **APPROACH.md** (plain language, for anyone): add what happened on 25–26 Sep (France diagnosis via LB, threshold,
   pruner, bge on AWS, house-number siblings, IDF), keep it simple; update the results table.
3. **Package dry run:** build `Yeti_submission.zip` with package.py from output/sub8c into a scratch dir, check the
   structure matches docs/PROBLEM.md, and report its size and contents list; do not upload anything.
4. README/SETUP consistency check (commands, recipes, timings). Fix obvious drift.
Commit on your branch; the master merges. Memory 2 GB. Never end your turn to wait.
