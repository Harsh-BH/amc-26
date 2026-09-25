#!/usr/bin/env bash
# Keeps the laptop out of OOM. When MemAvailable < LOW MiB, SIGSTOP the largest *probe command* -- a python process
# with oom_score_adj 1000, which only probe tool commands have (oom_child.sh); never a claude CLI, never the
# master's own jobs -- so it stops growing and swap can take its pages. Resumes everything once MemAvailable > HIGH.
# Run it in tmux (Claude Code's memory safeguard reaps its own background tasks, not tmux).
# usage: probes/memguard.sh [LOW=1500] [HIGH=3500]      DRY=1 only logs what it would pause.
# Also pauses when the disk has < DISKLOW MiB (default 1000) free; resumes above 2x DISKLOW.
main() {
  local LOW=${1:-1500} HIGH=${2:-3500} LOG=data/work/memguard.log paused=() avail pid
  cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
  while :; do
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    disk=$(df -BM --output=avail / | awk 'NR==2{print int($1)}')
    # a full disk fails every probe's writes at once: treat < DISKLOW MiB free like low RAM (pause, alert)
    if (( disk < ${DISKLOW:-1000} )); then echo "$(date +%T) DISK LOW ${disk}M free" >> "$LOG"; fi
    if (( avail < LOW || disk < ${DISKLOW:-1000} )); then
      pid=$(for p in $(pgrep -x python); do
              [[ $(cat /proc/$p/oom_score_adj 2>/dev/null) == 1000 && $(awk '{print $3}' /proc/$p/stat 2>/dev/null) != T ]] \
                && echo "$(awk '/VmRSS/{print $2}' /proc/$p/status) $p"
            done | sort -rn | awk 'NR==1{print $2}')
      if [[ -n $pid ]]; then
        echo "$(date +%T) avail=${avail}M pause pid=$pid ($(tr '\0' ' ' < /proc/$pid/cmdline | cut -c1-90))" >> "$LOG"
        [[ ${DRY:-0} == 1 ]] || { kill -STOP "$pid"; paused+=("$pid"); }
      fi
    elif (( avail > HIGH && disk > ${DISKLOW:-1000} * 2 && ${#paused[@]} > 0 )); then
      for p in "${paused[@]}"; do kill -CONT "$p" 2>/dev/null; done
      echo "$(date +%T) avail=${avail}M resumed ${paused[*]}" >> "$LOG"; paused=()
    fi
    [[ ${DRY:-0} == 1 ]] && return 0
    sleep 5
  done
}
main "$@"; exit
