You were killed at 04:01: your "re-featurize train world with new normalizer" command exceeded your memory cap
(2 GB including ~0.7 GB for this CLI) and the runner then stopped your whole session. The runner is fixed: an
oversized command now just exits 137 and you continue. Your cap is now 2.5 GB including the CLI — keep each
Python step under ~1.5 GB: featurize a SAMPLE (e.g. 100k true pairs + 100k random pairs) or the dev world,
load only the columns you need, and process in chunks.
You logged 0 ledger rows so far — log every measurement you make (from ledger import log). Finish the brief:
the taxonomy shares per class/country/source with F0.5-at-stake ranking, look-alike (chain) analysis,
city_key agreement true vs random, and the before/after lift of normalize.py. Update RESULT.md as you go.
