# P74-n54-adapt: adopt Aditya's N54 decoy-word removal (branch origin/night/aditya) on top of the final — FAST
User asked 27 Sep ~15:45. Base: output/sub66_final (LB 0.987391, best; `run.py --recipe sub66_final` rebuilds it byte-for-byte).
N54 (read: `git show origin/night/aditya:probes/N54-fr-decoyword-removal/RESULT.md`, `...:probes/NIGHT/FOR_HARSH.md`,
`...:probes/N54-fr-decoyword-removal/p54.py`, `...:probes/N54-fr-decoyword-removal/test_port.py`): France decoys copy the S1
name and swap/add ONE org word (comite, ecole, amicale, club, parents, sportive, anciens …) often at the SAME house number.
Label-free lens (P46 lens_table on France argmax pairs): decoy words = n_hn ≥ 20 & sibling-offset share s > 0.25; noise words
sit at s ≈ 0.02. In-domain the same cell is 0/65 (US) and 1/759 (India) true. LOCO on a gc70-style chain: US→India +0.0296,
India→US +0.00011/+0.00019 France-F0.5. src port on the branch: `france.decoy_mask(s1, sx, am, acc, s_min=.25)` + run.py
`DECOY` global applied after every France add rule (recipes sub_iu54_frgc70_n54 / sub_iu54_best_n54 = recipe + decoy=0.25).
It was NEVER uploaded; main does not have it; sub66_final does not include it.
Tasks:
1. Cherry-pick the src port only (france.decoy_mask + run.py DECOY hooks) from origin/night/aditya onto main (no other branch
   files; do NOT merge the branch). Add RECIPES["sub66_final_n54"] = dict(RECIPES["sub66_final"], decoy=0.25). Check where the
   DECOY hook sits relative to P64 veto / P66 adds / dup in the current run.py and put it after all France add rules
   (including P66 xfc and acro) and before dup_last; P66 adds are exact full-name matches, so they should not be hit (verify).
2. Build output/sub66_final_n54 (France lines only differ; India/US byte-identical). Report pairs removed, S1 emptied, the
   Qwen r distribution of removed pairs (a removal is good if they look like decoys: low r), top removed words, and how many
   removed pairs came from which stage (chain / rescue90 / gc70 / acro / P66). Test the port with test_port.py's check.
3. Re-run N54's LOCO on the CURRENT France chain analogue (the one P69 used, sub66_final-style: gc .7, rescue .9, P55N veto,
   acro, P64), both directions, h0/h1. Gate: both directions ≥ 0 and India→US ≥ +0.0001 (it's a removal, so precision-side).
4. Overlap: with P70's (amc-26-46) word-role veto/adds (data/work/p70/changes.parquet once it exists) and P71's noise-word
   tier — N54 is the decoy side of the same lens, so report overlaps and conflicts (a pair P70/P71 ADDS that N54 would drop).
5. Validator --check-ids (SageMaker spot, as P69 did), SUBMISSIONS row with expected LB, commit src + recipe to main.
RAM ≤ 2 GB local (France-only instrumented run as in probes/P69-fr-loosen/p69.py, flock /tmp/p69.lock). Disk ~5 GB.
Target: verdict 16:30, file 16:45. Message the lead (amc-26-a7) with numbers. Never end your turn to wait.
