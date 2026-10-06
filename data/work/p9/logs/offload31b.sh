#!/usr/bin/env bash
cd /home/harsh1/github-repos/amc-26/data/work/p9
for w in fr_idf144 fr31; do
  rclone copy $w gdrive:amc26-offload/work/p9/$w --skip-links && rclone check $w gdrive:amc26-offload/work/p9/$w --skip-links --one-way 2>&1 | tail -1
done
