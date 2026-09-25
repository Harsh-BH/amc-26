# P4b-rerank: hard-pair reranker
Fine-tune a multilingual cross-encoder (xlm-roberta-base MIT, or muril-base-cased Apache-2.0, or
multilingual-e5-small MIT) on train pairs in P4's uncertain band (0.1 < p < 0.9) using the "name | address"
text of both records. GPU under `flock /tmp/amc-gpu.lock`; RTX 3050 6 GB → fp16, max_len 64, batch sized to fit.
Measure holdout F0.5 when blending reranker and GBDT scores on the band, and inference cost for the full test
band. Adopt only if the gain is >= 0.003 F0.5 and the full-test inference fits the time budget.
Deliverable: probes/P4b-rerank/rerank.py with train/predict and the blend rule.

## UPDATE (12:55) — runs in parallel with P4/P5 (read probes/MERGE-1.md and probes/P8-pipeline/RESULT.md first)
- The decision number is the FULL-CORPUS holdout, not the mini-world.
  `code/business_entity_resolution/src/run.py --split holdout` = fold-0 S1 while all train S1 and S2/S3 compete.
  Current references: rules 0.7708; P5's proxy LightGBM + decision rule 0.921 on P3's holdout candidates.
- Data you can use now:
  - Full-corpus candidates for ALL train S1 are being written by P4 to data/work/p4/ (India done: 12.4M pairs,
    14.1/S1, recall 0.991). Train on folds >= 2 only; never on folds 0/1.
  - Holdout candidates with is_true: data/work/p3/cands_holdout_full_{India,US}.parquet.
  - Texts: data/work/p3/txt*/ (P3's cleaned name+address texts per id).
- Uncertain band: until data/work/p4/prob_holdout.parquet exists, define the band with P5's proxy probabilities or
  block_score. When P4's probabilities land, redo the band and the final evaluation on them.
- P8's diagnosis of rule false positives: 92.5% are parentless look-alikes sharing the exact name core with address
  score 4 (house numbers present on both sides). A reranker reading full "name | address" text may catch street
  mismatches the features miss — test that specifically.
- GPU: RTX 3050 6 GB, always under `flock /tmp/amc-gpu.lock`. Models: xlm-roberta-base (MIT), muril-base-cased
  (Apache-2.0), multilingual-e5-small (MIT). Download weights once. Disk is tight (~6 GB free): prefer small models,
  delete checkpoints you drop.
- Adopt only if full-corpus holdout F0.5 rises by >= 0.003 over the same pipeline without the reranker, and full-test
  inference on the band fits in ~1 h.
