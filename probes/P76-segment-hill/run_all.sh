#!/usr/bin/env bash
# P76 step 2: every India/US arm (fit h0->h1 and h1->h0) + label-permuted null runs; logs to data/work/p76/run_all.log
cd "$(dirname "$0")/../.."
export P76_PROCS=${P76_PROCS:-8}
PY=probes/P76-segment-hill/py.sh
H=probes/P76-segment-hill/hill.py
$PY $H single
for a in joint_d joint_dek; do $PY $H run $a 24 20; done
for a in caps_only extra_only fam_lf fam_org fam_twin fam_noaddr fam_ntok fam_hn fam_state fam_pair_na fam_rk; do $PY $H run $a 20 10; done
echo ALLDONE
