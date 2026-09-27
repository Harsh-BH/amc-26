# P76-segment-hill: which KINDS of companies fail, and hill-climb per-segment decisions (user request, 27 Sep ~16:45) — FAST
Base: output/sub66_final (LB 0.987391) = India/US sub54 combiner (src iu_stack.py; honest OOF frames data/work/p57/A_full_{India,US}.parquet
from P57) + France chain (src/france.py). Staged next: output/sub76 (all France add-ons + N54).
Prior: P56 showed finer threshold grain by SOURCE and GROUP STATE doesn't pay (p'' calibrated in those cells). Company-TYPE
segments were never tested — do that now, rigorously.
1. DEEP DIVE (India/US, labelled holdout, per-S1 F0.5 loss): segment every S1 by label-free company type and report loss share,
   F0.5, FP/FN counts, S1 count, and 2 raw examples of each error kind per segment:
   - legal form class (pvt ltd / llp / llc / inc / corp / none / other; P64 lf.py sig), org-word type (school/college,
     club/association/sports, hospital/clinic, temple/trust/society, restaurant/cafe/food, traders/enterprises/industries,
     consultancy/services, construction/realty, church, dental/medical, law, bank/finance …; build from top tokens),
   - name length (tokens), name is a common-name twin (core shared by ≥2 S1), S1 address has hn / unit / floor,
     India state / US state (top 10), S1 group size (# accepted), address-less SX share.
   Rank segments by (loss share − S1 share). Tell me which company kinds are over-represented in the loss and WHY (FN vs FP).
2. HILL CLIMB (India/US): per-segment acceptance threshold t_seg (start at 0.71), coordinate ascent over segments with step
   0.01–0.05, objective = per-S1 macro F0.5 on holdout half h0; evaluate the final t vector on h1 (and the swap h1→h0).
   Regularise: segments with < 2,000 S1 keep global t; cap |t_seg − 0.71| ≤ 0.2. Gate: both directions ≥ +0.0001 LB-eq at
   3 SE (paired). Also try the ONE best single-segment change only (less overfit). Include clone09 density check if it passes.
3. FRANCE: same segmentation on France test (label-free counts: accepted/empty rates vs India/US per segment → which company
   kinds look under-/over-matched). Hill-climb France per-segment bars on the LOCO chain analogue (P69's machinery,
   probes/P69-fr-loosen) — gate: US→India ≥ 0 and India→US ≥ +0.0003 France-F0.5, both halves.
4. If anything passes: build on output/sub76 (India/US lines for step 2, France lines for step 3), validator --check-ids,
   SUBMISSIONS row with expected LB. Commit code; message lead amc-26-a7 with the deep-dive table (top 10 segments) + verdicts.
RAM ≤ 2.5 GB local (frames are small); AWS CPU allowed ($5) if needed. Disk ~5 GB. Deep-dive table by 17:30, verdict 18:30.
Never end your turn to wait.
