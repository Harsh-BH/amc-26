# P52-gap-survey — read everything, find the levers nobody has pulled (NO heavy compute; reading + small queries only)
Read probes/NIGHT/AGY_COMMON.md (hard rules), docs/PROBLEM.md, APPROACH.md, SUBMISSIONS.md, probes/NIGHT/NOTES.md,
and EVERY probes/*/RESULT.md and BRIEF.md (≈ 50 probes). Also skim `.venv/bin/python ledger.py show exp | tail -200`.

Facts: LB best 0.986304 (#37_gc70); top-3 on the LB are 0.9908. India/US holdout ≈ 0.9903 (holdout ≈ LB), India/US loss
≈ 0.0083 LB-eq of which P40 calls 0.0037 a Bayes floor (address-less SX of same-name S1 twins). France ≈ 0.964
(≈ 0.0054 LB). Arithmetic: a team at 0.9908 with our India/US would need France ≈ 0.994 — so the top teams are very
likely ALSO better on India/US, i.e. either the "floor" is beatable or the non-floor 0.0046 is much more compressible
than P36/P40 concluded.

Deliverable probes/P52-gap-survey/RESULT.md:
1. A table of every lever tried (probe, what, result, LB/holdout number) — compact.
2. Challenge the floor: list concrete hypotheses for signal that could separate same-name twin S1s for an address-less
   SX (anything P40 did NOT test — check its RESULT carefully), each with a cheap test (duckdb query on train truth) —
   and RUN the cheap tests yourself (memory ≤ 1.5 GB, one python at a time, duckdb memory_limit 1GB). Report numbers.
3. The top 5 untried levers for India/US and top 5 for France, each with expected LB, cost on a 4 GB-GPU / 7 GB-RAM
   laptop (or on AWS for the teammate), and the first experiment to run. Be concrete and skeptical; no generic advice.
Deadline: 07:00 IST.
