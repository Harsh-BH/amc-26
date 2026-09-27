# P78-final-pkg2: package the NEW best (sub76, LB 0.988292) reproducibly; re-zip on sub77 if it wins — FAST
Current final/Yeti_submission.zip = sub66_final (0.987391) — STALE. New best: output/sub76 (md5 a7b2cfe0…, uploaded 16:56,
LB 0.988292). Pending upload: output/sub77 (= sub76 + P41 brand adds, md5 9e4fb898…).
How they were built (lead, by hand): probes/P75-combine/combine.py `sub66_final sub75a +sub67q +sub70a +sub66_t975ce`;
sub75a_n54 by probes/P74-n54-adapt/build74.py; sub76 = sub75a_n54 + sub71a's adds vs sub66_final (inline, same logic as
combine.py, adds taken relative to sub66_final so N54 removals are not re-added); sub77 = sub76 + sub41_on66's adds vs sub66_final.
Candidates: sub66_final_small's + missing add pairs. Each add-on file came from a probe: P67 (probes/P67-fr-synth, sub67q),
P70 (probes/P70-fr-roles, sub70a), P69 (probes/P69-fr-loosen, sub66_t975ce), P71 (probes/P71-fr-noiseword, sub71a),
P41-on66 (probes/P70-fr-roles/p41_on.py), N54 (src france.decoy_mask, recipe sub66_final_n54).
Tasks:
1. Make the build reproducible: a single script code/business_entity_resolution/src/final_build.py (or run.py recipe steps)
   that, from `run.py --recipe sub66_final` output plus each probe's documented build command (or their cached change sets:
   data/work/p67/changes.parquet, data/work/p70/changes.parquet, …), rebuilds sub76 (and sub77) BYTE-FOR-BYTE. Extend combine.py
   with the relative-to-base add mode used for 71a/41 and fold it into src. Verify md5 of both. Document each step in SETUP.md.
   If a probe's add set cannot be regenerated from code in time, ship its change set parquet inside code.zip with a note
   (ponytail: comment) — honest, not hidden.
2. Package: package.py --team Yeti for sub76 → output/final/Yeti_submission.zip (keep the old one as final/archive/…sub66_final.zip).
   Unzip → validator --check-ids (SageMaker, as P43 did). Backups S3 + Drive final/ (rclone copy only; never rclone link).
3. Docs: [FINAL LB] = 0.988292 (sub76) in README, APPROACH, Documentation_template (add P67/P70/P69 t975ce/P71a/N54 rows +
   methods), SUBMISSIONS. Commit + push.
4. Be ready to re-zip for sub77 within 20 min when the lead says it scored higher.
RAM ≤ 2 GB, disk ~3 GB (use hard links; don't duplicate 200 MB files). Target 18:15. Message lead amc-26-a7. Never end your turn to wait.
