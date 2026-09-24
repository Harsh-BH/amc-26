You were killed again at 04:13 — this time the kernel OOM killer picked the CLI itself while your
eval_addr.py ran alongside a backgrounded e5 job waiting on the GPU lock (two Python processes + CLI > 2.5 GB).
The runner now makes every tool command the OOM killer's first choice, so this cannot kill you again, but
do not run two memory-heavy Python processes at once: run the e5 embedding eval and the address eval one
after the other, each under ~1.5 GB. Continue from where you stopped (address eval, then e5/LaBSE comparison),
keep logging to the ledger, and finish RESULT.md with a clear recommendation.
