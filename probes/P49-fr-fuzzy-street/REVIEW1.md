# Master review of RESULT.md v1 — verdict REOPENED
Your negative verdict used the old bar (precision among pred==False), which the master has since shown is wrong for
France — read the REVISED VALIDATION BAR in probes/NIGHT/AGY_COMMON.md. Master's quick numbers (probes/P49-fr-fuzzy-
street/cellcheck.py, P31 proxy base; note p49.py's tail had a syntax error, fixed by the master):
  France arm_eq 584,651 pairs, 97.3% accepted, 15,512 rejected argmax pairs; arm_miss 45,554, 90.7% acc, 3,717 rej am.
  US holdout arm_eq 183,561 pairs, prec 0.9989, negatives 0.0016 per S1; India 89,071, prec 0.9899, 0.0103 per S1.
  ⇒ bound on France rejected-pair precision ≈ 1 − 0.0016·259,452/15,512 = 0.973 (US rate), 0.827 (India rate).
Tasks:
1. Recompute this properly: the cell per S1 in US/India/France (France is ~2.25/S1 vs US ~1.4 — explain: is France's
   address noise lower? count exact-street vs fuzzy-street shares per country), and the bound for the pairs your rule
   would ADD, separately for (i) exact street key equal (P28-like — check what P28 already takes) and (ii) fuzzy only.
2. De-duplicate against the current base's rules: drop pairs P42 hn-extra takes (free argmax non-sibling, hn equal,
   p >= 0.80, S1 group non-empty at the base), P28's same-address rule (probes/P28-addr-anchor), P37 gc (p>=0.70 one-sided
   groups). Report the truly NEW pairs by p band, how many fill EMPTY S1, and the bound for each band.
3. Label-free France check: among the NEW pairs, name_rel / legal-form / SX source mix vs the same cell's ACCEPTED France
   pairs (true-like) — if the new pairs look like the accepted ones, good; if they are dominated by something else, flag.
4. If it passes the revised bar: `p49.py build <base> <out>` adding only the NEW pairs (free SX, candidates only, caps
   5/6), selftest, and the expected LB ≈ Σ over added pairs of (gain if true × prec − loss if false × (1−prec)), use
   ~+0.08/1.73M per true add to a non-empty group and −0.2/1.73M per false add as a rough guide (be explicit).
