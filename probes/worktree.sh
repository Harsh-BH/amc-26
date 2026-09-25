#!/usr/bin/env bash
# usage: probes/worktree.sh <name>  -> ../amc-26-wt/<name> on branch wave/<name>, cut from main's HEAD, with the
# gitignored data/ and .venv symlinked in. Launch probes from inside it: (cd ../amc-26-wt/<name> && probes/launch.sh …)
set -euo pipefail
main() {
  local MAIN; MAIN=$(dirname "$(git -C "$(dirname "$0")" rev-parse --path-format=absolute --git-common-dir)")
  local WT="$(dirname "$MAIN")/amc-26-wt/$1"
  git -C "$MAIN" worktree add -q -b "wave/$1" "$WT" HEAD
  ln -s "$MAIN/data" "$WT/data"; ln -s "$MAIN/.venv" "$WT/.venv"
  echo "$WT"
}
main "$@"; exit
