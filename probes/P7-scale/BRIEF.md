# P7-scale: can the full run fit, and how to chunk it
Goal: an execution plan for the full test (1.73M S1, 9.97M S2/S3) on a 15 GB laptop.
Measure: parquet load RAM (pandas pyarrow strings vs polars) per source; TF-IDF char-3gram fit/transform RAM
on 5M strings; sparse_dot_topn throughput per chunk; features for ~50M candidate pairs (k≈30) must stream in
chunks — design it; LightGBM predict throughput; total wall time. If local won't fit, cost an AWS run (EC2 or
SageMaker with 64 GB, $200 credits; see docs/SAGEMAKER.md) and say which stages to move.
Deliverable: probes/P7-scale/SCALE.md (RAM and time budget per stage, chunk sizes) and a reusable
chunked-execution helper in probes/P7-scale/chunks.py.
