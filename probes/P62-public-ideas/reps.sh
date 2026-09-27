#!/usr/bin/env bash
# clone09 replicates, sequential (each build peaks ~1.8 GB)
set -e
P=probes/P62-public-ideas/clone09.py
for spec in "0.9 1 bern" "1.9 0 pois"; do
  set -- $spec
  .venv/bin/python $P build India $1 $2 $3
  .venv/bin/python $P build US $1 $2 $3
  .venv/bin/python $P eval $1 $2 $3
done
