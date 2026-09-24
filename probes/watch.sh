#!/usr/bin/env bash
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
while :; do
  for d in probes/*/; do
    if [[ -e $d/DONE && ! -e $d/.collected ]]; then touch "$d/.collected"; echo "finished: $d $(cat "$d/DONE")"; exit 0; fi
  done
  sleep 20
done
