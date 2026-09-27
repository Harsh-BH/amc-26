Master (27 Sep ~06:30): new work for the final package; the user is asleep, so take sound decisions and stay autonomous.
LB so far: gc70 0.986304 → sub54 0.986685 (uploaded 05:32; ~1/4 of the expected gain). Upload candidates staged by P50 (the
integrator; SUBMISSIONS.md has rows): output/sub_clean (sub54's France chain rebuilt without P46/P40, + P55 combiner, dup
last; India/US = P44 combiner + r_all, full-holdout refit + zoo), sub_clean_iu34 (same France, gc70's India/US), sub38_clean_p55
(Qwen France). P57 decides the India/US source (~08:00); P59/P60/P61 may add changes by ~12:30.
Task: port EVERY stage the final may use into src/ so `run.py --recipe <name>` rebuilds the final file from cached inputs:
France: P42 joint recipe (probes/P42-fr-joint-tune/solution_p42.py decide), P41 gib (probes/P41-fr-brand), P47 cls80
(probes/P47-fr-shape-match/p47.py), P48 low-CE veto (probes/P48-fr-ce-all), P54 in-band CE rescue u .99 (probes/P54-fresh-batch),
P55 combiner (probes/P55-fr-combiner/p55.py, v .6 band u .95), dup consistency last (probes/P54-fresh-batch/build54.py), plus the
optional P46 tier / P40 xf_noaddr as flags (off in sub_clean). India/US: P44 combiner (probes/P44-meta-stack te44.py /
solution_p44.py / ctx.py) + P45 r_all (data/work/p45/oof, te45.py) + P50 full-holdout refit + zoo (probes/P50-idea-batch asm50.py),
with a flag to fall back to #34 (a10). Recipes: `sub_clean`, `sub_clean_iu34`, `sub54`. Acceptance: each recipe reproduces its
output/ folder's matching_results.tsv as sets per line (report any differing lines). Keep country logic keyed on UNSEEN, not
"France". Then: small candidate set for the final (as you did for sub37: output/<final>_small, validator --check-ids), fill the
docs (Documentation_template.md, APPROACH.md, SETUP.md, SUBMISSIONS.md) with today's methods and LB history leaving a slot for the
final number, build Yeti_submission.zip with package.py from the recipe whose folder P50 marks FINAL (default sub_clean until told
otherwise; rebuild the zip if the final changes), unzip into a temp dir and validate, back up zip + final folder to S3
(amc26-offload/final/) and Drive (gdrive:amc26-offload/final/, rclone copy only; never rclone link). Commit src/ and docs to
main with the attribution lines. Heavy rebuilds may use SageMaker CPU ($10). Memory 1.5 GB local. DISK ~2.5 GB.
Deadlines: stage ports + reproductions by 11:00; zip/docs/backups for sub_clean by 12:30; re-zip for the final within 30 min
of P50 marking it.
