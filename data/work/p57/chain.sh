#!/usr/bin/env bash
cd /home/harsh1/github-repos/amc-26
for w in oos6 ins4 oos4 oos2; do P57_THREADS=14 .venv/bin/python probes/P57-iu-transfer/ev57.py $w > data/work/p57/ev_$w.log 2>&1; done
echo CHAIN_DONE >> data/work/p57/ev_oos2.log
