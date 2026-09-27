# P77-lb-forensics: learn from ALL scored submission files + test-set patterns (user request, 27 Sep ~16:55) — FAST
The public LB is the only France label source. We have ~14 scored uploads (SUBMISSIONS.md rows marked "uploaded … LB …":
#5 0.980017, D1 0.849163 (France-blank diag on #5), D2 0.981024, D4 0.981142, 6c 0.980954, 9ab 0.984072, 31 0.9856, 34_veto
0.985647, 34v_r90 0.98596, 37_gc70 0.986304, 54 0.986685, 66_final 0.987391; plus any others in SUBMISSIONS/ledger pipelines
world=test-public-lb — collect them all, with exact scores). Files: output/<name>/matching_results.tsv, most OFFLOADED to
s3:sagemaker-114872709824-ap-south-1/amc26-offload/output/<name> (see output/<name>.OFFLOADED; AWS_PROFILE=harsh rclone).
Disk is tight (~3 GB): stream one file at a time, keep only compact per-pair change sets (parquet of (s1, sx) diffs), delete.
A. LB-DELTA DECOMPOSITION. For each consecutive (and each informative non-consecutive) pair of scored files, compute the
   per-country change set (pairs added / removed, S1 touched, S1 emptied/filled). Tag every changed pair with label-free
   classes: country, source S2/S3, hn relation (exact / pad / sibling-offset / other / none), name relation (P20 name_rel:
   eq / sub1 / add1 / drop / disjoint / gib), address-less, legal-form class (P64), same-source-extra vs missing-source vs
   empty-S1 fill, decoy-word (N54 lens s>.25) / noise-word (s≤.05), matcher p band, Qwen r band (data/work/p38/r where present).
   Then fit a small non-negative / ridge regression: ΔLB_k ≈ Σ_class value_class · (Δcount_class,k)/N_S1 (per-S1 metric: use
   per-S1 counts; France weight 0.1498). Report per-class implied value (precision proxy) with bootstrap CIs, and which classes
   the LB has actually rewarded or punished. Be honest about identifiability (14 points, many classes: collapse classes, use
   priors from LOCO precision, report what is and isn't identified).
B. TEST-SET PATTERNS (label-free, France vs India/US): distributions that differ in ways our model may mishandle — S2/S3 per
   S1, share address-less, hn relation mix of top candidates, name_rel mix, noise/decoy word rates, duplicate-content SX groups,
   candidate-set sizes, empty rate of our predictions vs train truth 5.58%. Any France segment whose predicted empty/accepted rate
   deviates strongly from what the train law implies → list with counts and 3 raw examples each.
C. Output: RESULT.md with (1) the class-value table from the LB, (2) the top 5 actionable implications for the staged files
   output/sub76 (= sub66_final + P67q + P70a + t975ce + P71a − N54) and output/diag_frblank, (3) any cheap rule that the LB
   evidence supports — if one exists, build it on output/sub76 (France lines only), validator --check-ids, SUBMISSIONS row.
RAM ≤ 2 GB, AWS CPU allowed ($5) if streaming big files is faster there. Message the lead (amc-26-a7). Target 18:30.
Never end your turn to wait.
