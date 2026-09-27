# P66-fr-noaddr: France under-matches address-less SX (31% vs India 57% / US 62%) although they are ~98% true copies
URGENT, possibly the biggest France lever left. Master's measurements (27 Sep ~09:45):
- TRAIN truth: address-less SX are 4.41% of true copies but 0.29% of decoys → an address-less SX is a true copy 97.7% of the time.
  Of true address-less copies, the owner's normalised name is UNIQUE among S1 (same country) 65.2%; shared by 2+ S1 34.8% (the
  twin floor).
- TEST (sub_iu54_frgc70 = gc70 France): address-less share France 3.0% (~43k SX), India 2.4%, US 2.9%. MATCHED: France 31.2%,
  India 57.4%, US 62.0% (with-address SX: 57.5 / 58.1 / 58.8). ⇒ ~29k French address-less SX left unmatched; if France follows the
  train law, most are true. Cause: France's chain needs p ≥ 0.9868–0.99; address-less pairs can't reach it (name-only evidence),
  while India/US's combiner (t 0.71) accepts them. The Qwen/bge audit of P40's xf_noaddr (est 0.19) is NOT evidence against it:
  CEs score address-less pairs low by construction. NFG's public repo reports the same law (+0.0017 val from assigning them).
Prior art: P40 xf_noaddr (full normalised name equal to a UNIQUE France S1; +4,672 pairs; LOCO +0.00008 / +0.00002 at 0.9 —
read probes/P40-generator RESULT.md addendum), P49/P58 (twin floor), P35. Base = output/sub_iu54_frgc70 (default final) and
output/sub_iu54_best. Own worktree; code in probes/P66-fr-noaddr/.
Plan (fast):
1. Profile France's unmatched address-less SX: are they candidates? best p / argmax S1? is the argmax S1's name unique in France?
   name similarity (raw + core) to argmax S1 and margin to runner-up; does the S1 already have members from that source (caps)?
2. Rule family (country-invariant): assign an address-less SX to its argmax S1 when (a) the S1's name core is unique among the
   SX's candidate S1s (or unique in the country), (b) name similarity ≥ s and margin ≥ m over the runner-up, (c) caps and
   one-owner respected; optionally a small GBDT over [name sims, uniqueness, margin, group state, p] trained on India/US
   address-less pairs (country-invariant inputs only).
3. Validate: LOCO both directions with the France-chain analogue (P37/P42 machinery: the target country's address-less SX under
   the France recipe, i.e. strict t) — does the rule recover them at high precision? Also train-truth precision by (uniqueness,
   similarity) cell. Gate: India→US ≥ +0.0003 France-F0.5 with US→India ≥ 0 (or precision ≥ 0.85 of the added pairs in both).
4. Build output/sub66 = sub_iu54_frgc70 + France address-less adds, and output/sub66_best = sub_iu54_best + the same (France
   only; India/US byte-identical), validator --check-ids, SUBMISSIONS rows with expected LB (dF × 0.1498); tell P43 to port.
Memory 2 GB local; SageMaker CPU ($10). DISK ~2.2 GB. Never end your turn to wait. Deadlines: verdict 11:30, files 12:30.
