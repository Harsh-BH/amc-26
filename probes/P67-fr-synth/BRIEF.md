# P67-fr-synth: synthetic FRENCH training pairs from France's own test S1 records → fine-tune the CE / combiner on French text
User asked (27 Sep 13:50) for legal "external data / augmentation". Rules (docs/PROBLEM.md §rulings): external datasets are BANNED
(no gazetteers, postal/city tables, libpostal, cleanco, internet data, no model > 8B even as a teacher). EXPLICITLY ALLOWED:
"synthetic pairs from the provided records", pseudo-labels on test, test-side stats, small hand-written dictionaries (legal
suffixes, street types), wordfreq, HF weights MIT/Apache ≤ 8B. This probe uses only those.
Why: France's loss is over-confidence in the band (LOCO p .5–.9 only 20–44% true) and its models never saw French text
(bge CE trained on India/US). A French-text model calibrated on French synthetic pairs may separate true copies from French
decoys (siblings / same street / same brand) better than transfer does.
Plan (fast; ~9 h to deadline, need a verdict by 18:00, file by 19:30):
1. Generator (reuse probes/P40-generator catalogue.py — the op catalogue learned from India/US train truth: typo rates, legal-form
   drop, abbreviation, token drop/swap, address removal, gib brand, house-number/zip perturb, case/accent). Apply those ops, at
   India/US train rates, to France TEST S1 records → positives (S1, synthetic SX). Negatives = HARD French decoys: the real test
   S1-vs-other-S1 near neighbours (same street / shared name core / same city from the blocking index), and synthetic copies of a
   DIFFERENT S1 in the same block. A small hand-written French street-type/legal dictionary is allowed (avenue/av/ave, boulevard/bd,
   rue/r, chemin/ch, impasse/imp, place/pl, SARL/SAS/SASU/EURL/SCI/SNC...).
   SANITY FIRST: apply the same generator to India/US train S1 and check the synthetic pairs' feature distributions (name/addr sims)
   match the real train pairs; if not, fix rates before going on.
2. Fine-tune: continue bge-reranker-v2-m3 (the production CE; reuse the P59/P45 SageMaker training code) on synthetic French pairs
   (+ a replay slice of real India/US pairs to avoid forgetting), 1 epoch, g6e/g5 via sm_launch.py --profile harsh. Score France's
   band pairs (P55's France frame) → r_fr.
3. Validate WITHOUT France labels (LOCO analogue): train the same recipe with the target country held out, i.e. generate synthetic
   pairs from the TARGET country's S1 (US or India), fine-tune, and measure on its real labelled pairs whether the synthetic-tuned CE
   beats the transfer CE inside P55's combiner (add r_syn as a column; P65's ev65.py harness reproduces P55 exactly — reuse it).
   Gate: India→US ≥ +0.0003 France-F0.5 AND US→India ≥ 0 at the shipped cell.
4. If it passes: output/sub67 = output/sub66_final with France's combiner/band decisions re-run with r_fr (India/US byte-identical),
   validator --check-ids, SUBMISSIONS row with expected LB (dF × 0.1498), and a Qwen-audit + P64-veto check on the change set.
Also cheap side-arm (do it while the GPU trains): wordfreq French/English token commonness as a combiner feature (allowed lib) —
distinctive-token overlap between SX and S1; LOCO-gate it the same way.
AWS budget $15 (profile harsh ONLY). Memory 1.5 GB local. DISK ~1.9 GB — write big stuff to S3, not local. Never end your turn to
wait. Own code in probes/P67-fr-synth/. Message the lead with the verdict numbers.
