#!/usr/bin/env bash
# P76 step 3b: France per-segment chain bars on LOCO. $1 = joint | fams. Logs to data/work/p76/run_fr_$1.log
cd "$(dirname "$0")/../.."
export P76_PROCS=${P76_PROCS:-2}
PY=probes/P76-segment-hill/py.sh
H=probes/P76-segment-hill/fr_hill.py
if [ "$1" = joint ]; then $PY $H run joint 2 6; else
  for a in fam_twin fam_n_am fam_noaddr fam_ntok fam_lf2 fam_org; do $PY $H run $a 3 8; done; fi
echo ALLDONE
