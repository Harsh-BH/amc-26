You were killed at 03:49 because your last command (/tmp/p2_addr.py, the cross-script ADDRESS analysis on the
train world) exceeded your memory cap; the runner then stopped your whole session. The runner is fixed: an
oversized command now just exits 137 and you continue. Your cap is 2.5 GB INCLUDING ~0.7 GB for this CLI,
so keep each Python step under ~1.5 GB (use the dev world or a sample of the train world, fewer columns).
Continue your brief from where you stopped: your earlier finding (cross-script names are token-level
transliterations, 96.8% positional alignment, ~1,343 Indic token types → a learned token dictionary) is
the lead to pursue. Write a draft probes/P2-script/RESULT.md now, then continue.
