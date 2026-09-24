You were killed at 04:40: a Python step of your full-corpus evaluation peaked at your 4 GB cap and the old
runner tore down your whole session. The runner is fixed (an oversized command now just exits 137 and you
continue). Your cap is 3.5 GB INCLUDING ~0.7 GB for this CLI — keep every Python step under ~2.3 GB:
build the corpus index per country (India, US separately), query holdout S1 in chunks, free each country
before the next, float32 sparse matrices, and write intermediate results to data/work/P3/ so a kill loses
one chunk, not the run.
New input since you started: P2-script finished — promote-ready module probes/P2-script/translit.py.
Apply translit.to_latin() to S2/S3 business_name and business_address BEFORE your normalization and
translit.fold() to both sides of name n-grams; re-measure recall_cross_script (it was 0.37–0.98 before).
P1's probes/P1-taxonomy/normalize.py (name_core etc.) is also available — test it as the name key.
Your RESULT.md is still the 03:50 draft: update it now with your 21 ledger rows' findings, then finish the
full-corpus measurement and the recommended cfg.
