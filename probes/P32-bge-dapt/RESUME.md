Master here — one more kit fact: **the team member has MULTIPLE H100s** (80 GB each). Make run.sh run the independent
chains in PARALLEL, one per GPU: detect the GPU count (nvidia-smi -L, non-fatal) or take `--gpus N` / respect
CUDA_VISIBLE_DEVICES; launch each chain (LOCO a, LOCO b, France, …) as its own process pinned to one GPU
(CUDA_VISIBLE_DEVICES=i), log to logs/<chain>.log, wait for all, then validate the r files. With 1 GPU it runs them
sequentially as now. README: expected wall time with 1 vs N H100s, how to check progress, what to send back (the r files).
Then zip kit+data and upload to gdrive:amc26-offload/dapt_kit/ (rclone copy only, never `rclone link`) and report the
Drive path + size in RESULT.md immediately — the user is waiting for it. Your AWS jobs keep running; continue after.
Never end your turn to wait.
