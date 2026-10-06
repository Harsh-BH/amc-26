#!/usr/bin/env bash
cd /home/harsh1/github-repos/amc-26/data/work
rclone copy p31/c_hn.txt gdrive:amc26-offload/models/p31/ && echo "c_hn ok"
for d in p19/r p26/bge/r p29/r p31/r; do rclone copy $d gdrive:amc26-offload/work/$d/ && echo "$d ok"; done
rclone copy p30/r/r_pool_France.parquet gdrive:amc26-offload/work/p30/r/ && echo "p30 ok"
rclone lsf -R gdrive:amc26-offload/models/p31 gdrive:amc26-offload/work/p31 2>&1 | head
