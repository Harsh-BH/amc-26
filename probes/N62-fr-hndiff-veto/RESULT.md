# N62-fr-hndiff-veto — RESULT (laptop, 27 Sep ~16:30 IST) — NEGATIVE, no file
Question: rival research (SaiMukhund v3_france: "France pairs whose house numbers differ need p > 0.98") + N57's best
near-miss cell ('eq','none','hn_diff','0.7-0.9': IN→US +0.0075, US→IN −0.0010). Is there a France hn-differ veto to ship?

## TL;DR
- **No. The shipped France chain is already clean of house-number sibling decoys.** On the submitted sub31-era file
  (output/matching_results.tsv), accepted France pairs at hx − h1 = +D are ≤ the −D mirror for every generator offset
  (1:142/169, 2:145/182, 3:108/145, 4:126/179, 5:129/136, 7:102/141, 9:101/136, 11:79/105, 13:104/99). Only +21 shows an
  excess (77/46 ≈ 30 pairs, ≈ +0.000005 LB). Decoys sit at +D only (France unaccepted candidates +168,753 / −11,228).
- **Trap for other probes:** the `acc` column of `data/work/p31/pairs_France.parquet` is PRE-veto. On it the same cells look
  decoy-heavy (+sib 1,999 / −1,106; p∈[.98,.995) +1,249 / −362, est. precision .25), with an expected +0.00010–0.00016 LB
  that the real file does not have. Size France levers on a real matching file (or P69's frame), not on the p31 `acc`.
- LOCO (n62.py loco, N57's gc70 chain) splits by direction exactly as N57 found. Every hn-diff removal loses US→India
  (removed precision .91–.98) and gains India→US (.17–.42). Once vetoes are applied, the France accepted hn-diffs look
  truth-like, so the India→US reading does not apply to the shipped France.

## Side results (for the lead / N61)
- N61's decoy-word veto target is REAL on the shipped file: 1,681 accepted France pairs have "groupe" in the SX core but not
  the S1 core, and 257 have "développement" (96% at the S1's house number). participations/holding/distribution/international
  are ≈ 0 (already removed). sub66_final's full-coverage P30 veto may have removed part of these; count on Harsh's box.
- The label-free "decoy word repeats across copies" test (wordrep.py) is uninformative. Noise words are stripped from cores
  (no calibration point), and known decoy words rarely repeat (US group .133, holdings .058). France groupe .128 and
  développement .067 match the US decoy profile: consistent with N61, not proof.

## Files
n62.py (loco | mirror <base> | build <base> <out>), kinds.py, sign.py, mirror.py, expect.py, wordrep.py, loco.log.
Re-check on sub66_final in 1 min on Harsh's box: `.venv/bin/python probes/N62-fr-hndiff-veto/n62.py mirror output/sub66_final`
(if removed ≤ mirror, there is nothing to do).
