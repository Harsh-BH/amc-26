#!/usr/bin/env bash
cd /home/harsh1/github-repos/amc-26
while kill -0 1473878 2>/dev/null; do sleep 5; done
for c in India US; do .venv/bin/python probes/P57-iu-transfer/p57.py build $c swap > data/work/p57/build_swap_$c.log 2>&1; done
for w in sw swi c4i c4o; do P57_THREADS=14 .venv/bin/python probes/P57-iu-transfer/ev57.py $w > data/work/p57/ev_$w.log 2>&1; done
echo CHAIN_DONE > data/work/p57/chain3.done
