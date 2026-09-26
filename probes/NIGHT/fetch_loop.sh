#!/bin/bash
# fetch origin/main every 30 min over HTTPS (SSH blocked); log new commits. Merging is done by hand.
cd /home/epitaph/repos/ml-challenge/amc-26
while true; do
  old=$(git rev-parse origin/main)
  timeout 120 git -c credential.helper= -c "credential.helper=!gh auth git-credential" fetch -q https://github.com/Harsh-BH/amc-26.git +refs/heads/main:refs/remotes/origin/main
  new=$(git rev-parse origin/main)
  echo "$(date '+%F %T') fetch rc=$? $old -> $new" >> /tmp/claude-1000/amc_fetch.log
  [ "$old" != "$new" ] && git log --oneline $old..$new >> /tmp/claude-1000/amc_fetch.log
  sleep 1800
done
