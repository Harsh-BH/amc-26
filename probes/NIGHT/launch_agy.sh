#!/bin/bash
# usage: probes/NIGHT/launch_agy.sh <PROBE_ID> [model] [extra prompt]  — runs in foreground; call with run_in_background
P=$1; M=${2:-gemini-3.1-pro-high}; X=$3
cd /home/epitaph/repos/ml-challenge/amc-26
OOMC=$PWD/probes/oom_child.sh; env SHELL=/bin/bash BASH_ENV=$OOMC ZDOTDIR=$PWD/probes/NIGHT/zdot \
  agy -p "You are probe $P. Read probes/$P/BRIEF.md and follow it exactly, together with probes/NIGHT/AGY_COMMON.md (read its MEMORY GOTCHAS section). Reuse any files already in probes/$P/ from an earlier interrupted attempt. $X Work until the RESULT.md is final." \
  --model "$M" --dangerously-skip-permissions > data/work/agy_$P.log 2>&1
echo "$P exit $? at $(date +%T)"; tail -n 15 data/work/agy_$P.log
