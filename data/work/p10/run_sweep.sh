#!/usr/bin/env bash
set -e
cd /home/harsh1/github-repos/amc-26-wt/p10-candidates
# testq done
.venv/bin/python probes/P10-candidates/sweep.py pruner.txt
echo SWEEP_DONE
