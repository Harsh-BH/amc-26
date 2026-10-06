#!/usr/bin/env bash
cd /home/harsh1/github-repos/amc-26
while kill -0 1488232 2>/dev/null; do sleep 5; done
export P57_THREADS=14
P57_ARMS=rel44r,rel54 .venv/bin/python probes/P57-iu-transfer/ev57.py full > data/work/p57/ev_full_rel.log 2>&1
.venv/bin/python probes/P57-iu-transfer/hnmix.py full sw > data/work/p57/hnmix.log 2>&1
.venv/bin/python probes/P57-iu-transfer/dw.py full sw > data/work/p57/dw.log 2>&1
P57_ARMS=rel44r,rel54 P57_REFIT=rel54 .venv/bin/python probes/P57-iu-transfer/ev57.py sw > data/work/p57/ev_sw_rel.log 2>&1
P57_ARMS=rel44r,rel54 P57_REFIT=none .venv/bin/python probes/P57-iu-transfer/ev57.py c6 > data/work/p57/ev_c6_rel.log 2>&1
P57_REFIT=sub54,sub44r .venv/bin/python probes/P57-iu-transfer/ev57.py swi > data/work/p57/ev_swi.log 2>&1
for w in c4i c4o; do P57_REFIT=none .venv/bin/python probes/P57-iu-transfer/ev57.py $w > data/work/p57/ev_$w.log 2>&1; done
echo CHAIN_DONE > data/work/p57/chain5.done
