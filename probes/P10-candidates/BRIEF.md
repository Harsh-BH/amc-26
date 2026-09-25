# P10-candidates: shrink candidate_pairs.tsv without losing F0.5 (new ranking criterion)
Organiser update (25 Sep, data/notes/unstop_play_page_v2.txt): "Candidate generation counts toward the final ranking...
The approach that generates a smaller candidate set per Source 1 entity will be ranked higher in the final evaluation
beyond the public/private leaderboard." They review candidate_pairs.tsv AND the code that produces it. By the rules,
candidate_pairs.tsv = the exact set the final matching model runs inference over (the LAST filtering stage).
Read probes/MERGE-2.md, probes/P3-block/RESULT.md, probes/P4-match/RESULT.md (incl. "Improvement round"),
probes/P9-integrate/RESULT.md first. You work in your own worktree; write code only under probes/P10-candidates/
(P9 is editing src/ in parallel — do not touch src/).

Current (#5): P3 blocking (TF-IDF reverse top-2/forward top-10, ~14/S1 at recall 0.9916) ∪ rule-accepted pairs
= 27.5M test pairs = 15.9 candidates/S1; matcher m2_big (data/work/p4/m2_big.txt, 65 features in
data/work/p4/X/*) + P4b cross-encoder blend on India/US band -> argmax per S2/S3, t 0.69 (France t 0.78).
Final matches 3.38/S1; true matches average ~3.47/S1. Full-corpus holdout F0.5 0.9885 (P5 report half).

Goal: a learned PRUNING stage between blocking and the matcher (meta-blocking), giving the Pareto curve
candidates/S1 vs final F0.5 on the full-corpus holdout, and a recommended operating point:
- Target: <= 6 candidates/S1 (ideally 4-5) with final F0.5 loss <= 0.0005 vs 0.9885 on the SAME yardstick
  (P5's bench, report half; blend + thresholds refit on the tune half if needed).
- The pruner must be a distinct, light stage (e.g. logistic regression or a small GBDT on ~10-15 blocking-level
  signals: block_score, block ranks both directions, via, name_core equality, token Jaccard, house-number / city /
  state agreement, rule_acc, candidate counts). Train it on train folds >= 2 only (never 0/1). Keep S1-side and
  S2/S3-side views: e.g. keep a pair if pruner p >= tau, plus guard rails (always keep each S2/S3's top-1 and rule-
  accepted pairs) — compare variants.
- IMPORTANT: m2_big was trained with context features computed over the FULL blocking list. Keep computing those
  context features over the blocking list (they are blocking-stage statistics), so m2_big's p on kept pairs is
  unchanged; the pruner only removes pairs. Then the final answer can only lose pairs the pruner dropped — measure
  exactly that loss.
- Also report the simple alternative for comparison: P3's cheaper blocking config (keep_r=1, keep_f=5 -> 6.9/S1,
  recall 0.9881) and a plain per-S1 top-k cap by block_score.
Deliverables:
1. probes/P10-candidates/prune.py: train(), prune(pairs, feats) -> kept mask; the chosen tau; the model file under
   data/work/p10/.
2. Pareto table (candidates/S1, pair recall of the candidate set, final F0.5 per country) in RESULT.md + ledger rows.
3. For the recommended point: test candidate set and the resulting #6 files:
   data/work/p10/prob_test_{c}.parquet (the pruned rows with the #5 blended p) so the master can write matching_results +
   candidate_pairs with src/submit_probs.py. Check France too (no labels: report its candidates/S1 and that its
   answer profile stays in P6's bands: empty 4.5-7.5%, matches/S1 >= 2.9).
Memory cap 4 GB incl. the CLI, swap allowed. Disk ~10 GB free: do not copy the X files; read them.
