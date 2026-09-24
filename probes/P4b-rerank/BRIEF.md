# P4b-rerank: hard-pair reranker
Fine-tune a multilingual cross-encoder (xlm-roberta-base MIT, or muril-base-cased Apache-2.0, or
multilingual-e5-small MIT) on train pairs in P4's uncertain band (0.1 < p < 0.9) using the "name | address"
text of both records. GPU under `flock /tmp/amc-gpu.lock`; RTX 3050 6 GB → fp16, max_len 64, batch sized to fit.
Measure holdout F0.5 when blending reranker and GBDT scores on the band, and inference cost for the full test
band. Adopt only if the gain is >= 0.003 F0.5 and the full-test inference fits the time budget.
Deliverable: probes/P4b-rerank/rerank.py with train/predict and the blend rule.
