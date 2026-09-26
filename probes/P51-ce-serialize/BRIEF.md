# P51-ce-serialize: Ditto-style structured serialization for the bge cross-encoder (user idea: a code-built "summary" per row)
Idea: instead of raw "name | addr" text, feed the CE a deterministic, code-generated summary of each record with our parsed
fields, e.g. "[NAME] camera amicale [LEGAL] sarl [HN] 8 [STREET] r konrad adenauer [CITY] nantes [BRAND] no [NOADDR] no".
Prior art: Ditto (Li et al., VLDB 2021: serialization with [COL]/[VAL] tags + domain-knowledge injection helped transformer
matchers). No LLM involved — the summary comes from src/normalize.py + src/france.py parsers (legal forms, fixed '(n)' hn
parse, street canonicalisation incl. French types, learned départements/cities, P36 gib.is_gib brand flag, address-less
flag). France-specific formats ("(11)", "N°", "R.", "Sàrl") map to the same canonical tokens as India/US training data,
so the CE's knowledge should transfer better to France.
Read: probes/P13-aws-rerank (bge training/scoring job, pair channels, LOCO protocol; the dict-comment crash lesson),
src/bge_train.py, probes/P38-qwen4b-fr (gate script + build38.py France chain; ~8 SE Qwen−bge gate machinery),
probes/P45-ens-stages (r_all scoring + holdout OOF columns). Own worktree; code in probes/P51-ce-serialize/.
Plan:
1. Write serialize(df_side) → string for S1 and SX (India/US/France identical code path; no country-name branching).
   Show 20 examples per country. Keep the raw name too (in its own [RAW] field) so nothing is lost.
2. Two LOCO bge fine-tunes on AWS with the new serialization, same recipe/epochs/pair sets as P13 (train US → score India
   holdout band; train India → score US holdout band). Single-GPU types in parallel (ml.g5.2xlarge + ml.g6.2xlarge or
   g6e.2xlarge if free; P38 may hold g6e.xlarge; quota 1 per type). Smoke job first. Hyperparameters from a config file.
3. Gate with P38's gate script: France-recipe LOCO F0.5 vs the raw-text bge (same pairs, paired per S1), both directions.
   Also report vs Qwen if P38's LOCO scores are available. Pass = both directions ≥ +0.0010 France-F0.5 over raw bge.
4. If it passes: pooled fine-tune (India+US non-holdout), then score (a) France band pairs → rebuild France via
   build38.py-style chain on the current best stack, and (b) every India/US holdout + test argmax pair (as P45 did for
   r_all) → hand to P44's combiner as an extra column (tell P44/P45 via data/work/p51/ + a note). Build output/sub51 on
   the current best staged file (sub44r or its successor), validator --check-ids, SUBMISSIONS.md row with expected LB.
Budget $30 AWS. Memory 2 GB local. DISK ~1.9 GB: keep data on S3. Never end your turn to wait.
Deadline: LOCO verdict by 08:00, sub51 by 12:00 (hard stop 14:00).
