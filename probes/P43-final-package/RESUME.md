# RESUME (27 Sep ~13:45) — port P64 + P66 into run.py, stage sub66_final as a byte-exact recipe

Current state: final/Yeti_submission.zip = sub_iu54_frgc70 (validated). New likely final: output/sub66_final
(md5 c6246285 matching) = output/sub64 (sub_iu54_best − P64 France legal-form conflict veto, 344 pairs) + P66 France
address-less adds (1,884 pairs). Both validator PASS. Their code.zip is currently a hard link to sub_iu54_best's (stale).

Tasks, in order:
1. Port P64 veto into src/france.py: see probes/P64-legal-form-feat/fr64.py / frtest.py / attach.py (lf.pair_class conflict →
   drop France pair). Recipe `sub64` = sub_iu54_best + veto; `sub64_frgc70` = sub_iu54_frgc70 + veto.
2. Port P66 xfc adds (probes/P66-fr-noaddr/fr66.py, RESULT.md). Spec from P66: in the France loop after gc,
   pool = am & ~acc & (p >= 0.3) & SX address empty; cls = (fkey(n1)==fkey(nx)) & (fkey count among France S1 == 1) &
   (S1 core count among France S1 == 1) & (len >= 5); acc |= rule40.add(..., cls, 0.3). Skip any add P64 vetoes.
   Recipe `sub66_final` = sub64 + adds (dup consistency last), `sub66` = sub_iu54_frgc70 + adds.
3. Acceptance: `run.py --recipe sub66_final` reproduces output/sub66_final/matching_results.tsv as sets per line; same for sub64.
   Report differing lines.
4. Rebuild code.zip for sub64 and sub66_final from the new src (replace the hard link: rm then write, do not write through it).
5. Do NOT re-zip final/ yet — the lead will tell you which recipe is FINAL after the user's LB upload. Prepare the command.
6. Commit src/ to main with attribution. Memory 1.5 GB local. Disk ~1.9 GB — keep new files small; no new big caches.
Message the lead session when 1–4 are done (target 15:00).
