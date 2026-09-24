#!/usr/bin/env bash
# Exits (waking the master) when a probe finishes -- DONE written -- or dies without writing DONE
# (its session's claude process is gone for >60 s after launch).
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
main() {
  while :; do
    for d in probes/*/; do
      [[ -e $d/.collected || ! -e $d/session_id ]] && continue
      if [[ -e $d/DONE ]]; then touch "$d/.collected"; echo "finished: $d $(cat "$d/DONE")"; exit 0; fi
      local sid; sid=$(cat "$d/session_id")
      if (( $(date +%s) - $(stat -c %Y "$d/session_id") > 60 )) && ! pgrep -f "claude -p .*$sid" >/dev/null; then
        touch "$d/.collected"; echo "finished: $d DIED-WITHOUT-DONE (no claude process for session $sid)"; exit 0
      fi
    done
    sleep 20
  done
}
main; exit
