# NIGHT (Aditya laptop, 27 Sep 03:00 IST →) — master notes

Branch `night/aditya` (local, never push main). origin/main fetched every 30 min by `fetch_loop.sh`
(log /tmp/claude-1000/amc_fetch.log); merge by hand. Only Harsh uploads.

## Where the score is lost (from P36/P40 RESULTs)
- LB 0.986304 (#37_gc70) → loss 0.0137. India/US holdout loss ≈ 0.0083 LB-eq, of which 0.0037 is P40's "Bayes floor"
  (address-less SX whose true S1's full name is shared by ≥ 2 S1). France ≈ 0.964 → ≈ 0.0054 LB.
- #1 at 0.990 ⇒ plausibly France solved to ~0.99 with India/US ≈ ours. France is the lever (max ≈ +0.004).
- Harsh is running P44 (meta stack, India/US), P45 (stage ensembles), P47 (France shape matching). Don't duplicate.

## Log
- 03:30 P40's "Bayes floor" re-checked with raw names (twin_raw.py): SX copies the true twin's raw name 4.3% vs the
  other twin's 5.3% → no surface leak; floor holds. DROP.
- 03:45 Unique-core address-less SX (al_core*.py): train 96.6% US / 97.7% India true for the unique-core S1, but the
  in-domain model already accepts 96%; leftovers at p<0.5 are mostly false (word-drop variants of other S1s). DROP.
- 04:05 Read 45 unaccepted France argmax pairs (P31 base, p 0.3–0.99). Launched agy (gemini-3.1-pro-high) probes:
  P48 filler-phrase tier (same address, "& Fils"/"Cie"/"& Associés"... beyond P46), P49 fuzzy street (elision /
  abbreviation / typo with equal name + hn), P50 exact-name at a different address. Common rules: AGY_COMMON.md.
