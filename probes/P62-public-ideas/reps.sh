#!/usr/bin/env bash
# clone09 stress replicate, sequential (each build peaks ~1.8 GB); frames deleted after eval (disk)
set -e
P=probes/P62-public-ideas/clone09.py
.venv/bin/python $P build India 1.9 0 pois
.venv/bin/python $P build US 1.9 0 pois
.venv/bin/python $P eval 1.9 0 pois
rm -f data/work/p62/A_clonepois1.9s0_*.parquet
