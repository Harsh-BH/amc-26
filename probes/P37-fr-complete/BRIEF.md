# P37-fr-complete: complete one-sided France groups (the biggest France error found on 26 Sep 23:30)
Base = output/sub34v_rescue90 (LB 0.98596, best). France probs: data/work/p31/fr31v0.1_t9868.parquet (s1,sx,p), t 0.9868.
The empty-S1 rescue logic that gained +0.00031 is in SUBMISSIONS.md / the #34v_rescue90 commit (argmax per SX,
non-sibling via probes/P15-research/sib_rule.py, best per S1, p ≥ t_low). Own worktree; code under probes/P37-fr-complete/.

Finding (master, label-free, verified on train truth):
- In truth every SX belongs to exactly one S1; ≤ 5 S2 and ≤ 6 S3 per S1. India/US test predictions match train truth's
  per-S1 S2/S3 distribution almost exactly; France does not.
- Share of S1 by group shape (France / India / truth): S2+S3+ 0.732 / 0.789 / 0.805; S2-only-missing (S3+ no S2)
  0.111 / 0.079 / 0.075; S3-missing (S2+ no S3) 0.097 / 0.074 / 0.065. ≈ 14k excess one-sided France S1.
  Mean predicted SX per S1: France 3.10 vs India/US 3.38 (truth 3.46) → France misses members inside non-empty groups.
- One-sided France S1: 28,729 have S3 but no S2 (20,102 have an S2 candidate; best p 0.9–0.987: 3,412; 0.7–0.9: 1,736);
  25,041 have S2 but no S3 (17,456 with an S3 candidate; 0.9–0.987: 3,070; 0.7–0.9: 1,467). ~98% of those best
  candidates are unassigned, ~80% are their own argmax. ≈ 8.6k one-sided S1 have NO candidate in the missing source.
  Eyeballed 14 examples at 0.9–0.987: most are clear typo/format variants of the same business.

Tasks:
1. Group-completion rule: for a one-sided France S1, add the best candidate of the missing source if it is argmax for that
   SX, unassigned, non-sibling, keeps caps, and p ≥ t_gc. Grid t_gc ∈ {0.95, 0.9, 0.8, 0.7}. Also a variant that adds
   *every* such free argmax member ≥ t_gc to any France group (not only one-sided) — the per-S1 member deficit is 0.28.
2. Validate label-free-for-France: LOCO proxy (train one country, apply the same rule on the other's holdout at the
   quantile-matched p, since France p is off-domain) and the shape check (after the rule, France's shape shares should
   move toward India's, not past them). Gate as usual: LOCO ≥ +0.0005 both directions, or report precision of the added
   pairs by direction; break-even for adding to a partially-correct group ≈ 0.68.
3. Retrieval for the ~8.6k one-sided S1 with no candidate in the missing source: a narrow second blocking pass
   (house-number + street token key within the same département/city string from the data itself — no external tables),
   scored by the France chain. Measure on LOCO what it recovers.
4. Build output/sub37_gc<t> = sub34v_rescue90 with France groups completed (India/US byte-identical), validator PASS,
   2–3 staged variants ranked by expected LB, plus run.py integration notes. Update SUBMISSIONS.md upload queue.
RESULT.md with tables. Memory 3 GB. DISK NEARLY FULL (~1.5 GB free): keep intermediates small, never write >300 MB
locally. Never end your turn to wait. Deadline 27 Sep 10:00 IST (first variant as early as possible).
