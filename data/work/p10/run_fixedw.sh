#!/usr/bin/env bash
cd /home/harsh1/github-repos/amc-26-wt/p10-candidates
until grep -qE "BIG_DONE|Traceback" data/work/p10/big.log || ! pgrep -f run_big.sh >/dev/null; do sleep 20; done
.venv/bin/python probes/P10-candidates/fixedw.py 0.02 0.03 0.05
echo FIXEDW_DONE
