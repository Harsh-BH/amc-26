# P53-s2s3-graph: S2↔S3 similarity as match evidence (Foursquare/Shopee triangle trick)
Why: our stacker's context (competitor mass, ranks, same-source counts; P44) looks only through S1. Direct S2–S3 similarity
is a separate signal: if S2 x strongly matches S1 a and S3 y is very similar to x, raise P(y→a) (Foursquare 1st/4th
place GNN/graph post-processing, Shopee neighbour blending). P35 tested a narrow "address-less twin propagation" rule and
failed — read it and do NOT repeat that exact rule; this is a learned feature over all SX.
Current best = output/sub48 (India/US = P44 combiner + P45 r_all, reproducible via probes/P44-meta-stack te44.py /
solution_p44.py and data/work/p45/oof/; France = sub46_on43 France − P48 vetoes). Own worktree; code in probes/P53-s2s3-graph/.
Task:
1. For each S1 a's candidate block, compute S2×S3 similarity between candidate SX (cheap: normalized name/addr similarity,
   hn equality, e5 cosine from P45's e_* approach; CE only if cheap). Features for pair (a, y): max_x min(p(x→a), sim(x,y)),
   count of confident same-S1 partners similar to y, sim to the best confident partner, and the reverse for competitors.
2. India/US: add as columns to P44's combiner; holdout half-split, paired SE; gate India+US ≥ +0.00010 at ≥ 3 SE,
   neither half negative (user: don't leave small gains, but it must be positive both ways).
3. France: LOCO both directions as an extra acceptance/veto rule on top of the current recipe analogue.
4. If anything passes: build output/sub53 on the current best staged file (sub52_max if present, else sub48), validator
   --check-ids, SUBMISSIONS.md row. Memory 2 GB local; SageMaker CPU ($10). DISK ~2 GB. Never end your turn to wait.
Deadline: verdict by 07:30, file by 09:00.
