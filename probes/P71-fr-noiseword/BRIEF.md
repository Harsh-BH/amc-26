# P71-fr-noiseword: French noise-word tier + France hn-shift drop + small France rule gains, on output/sub66_final — FAST
Launched 27 Sep ~15:20 IST by the user's second session (lead = amc-26-46 decides uploads). Evidence: docs/RIVAL_INTEL.md.
Base: output/sub66_final (LB 0.987391, best; `run.py --recipe sub66_final` rebuilds it; France chain in src/france.py).
We sit at public rank ~117 of 6,938; top-100 cut was 0.987804 at 11:53 and rising. Every +0.0001 counts, small gains included.
DEADLINE: verdicts + staged files by 19:30 IST (final uploads close ~23:30). Never end your turn to wait.

A. NOISE-WORD TIER (main arm). On sub66_final, 14,636 France SX are UNASSIGNED although they sit at the S1's house number and
   their name = S1 name + one noise word, or one S1 word swapped for one (groupe 5,203 swap / 1,432 insert; développement 4,754 /
   1,386; associés 835; 14,419 have a single such S1). S2/S3-vs-S1 token over-representation on France test: développement 10.6×,
   groupe 4.7×, international 30×, participations 34,280:1 (median word 0.9×) → generator copy-noise, not decoys. The CEs (bge,
   Qwen) are trained on India/US where "group/holdings/sons/services" marks decoys (0–4% true), so they reject these; P50's
   `sub_clean` dropped the P46 tier because a Qwen judge estimated precision 0.08 — that judge is biased in exactly this way.
   LB evidence elsewhere: a team's filler fix (+2,831 pairs) gained +0.000138; another team's per-country noise-ratio name
   features gained +0.0056 LB. Our own caveat: #54's bundle of France rule-adds netted ≈ −0.0001 (it also held P54 rescue ≈45%
   precise and P40 xf_noaddr ≈0.19 — not this tier alone).
   Do: (1) learn the noise-word list LABEL-FREE per country from S2/S3-vs-S1 token ratios (test-side stats are allowed), not a
   hand list; (2) build the add set on sub66_final: free argmax-or-single-S1 SX, identical house number (both non-empty), S1 street
   + city tokens present, name diff ⊆ {noise words} (insert) or one non-legal S1 word ↔ one noise word (swap); respect caps 5/6,
   one owner, P64 legal-form veto, sibling offsets excluded; (3) evidence per sub-tier (dev+fils+associés+et / groupe / swap vs
   insert): hn-equal share, sibling-offset share, the −D mirror estimator (count at −D estimates true share at +D; true hn noise is
   symmetric), US/India analogue precision with the US/India noise list (learned the same way), and the France Qwen/bge scores
   as a diagnostic only; (4) stage output/sub71a (dev+fils+associés+et+france), sub71b (+groupe), validator --check-ids.
B. FRANCE HN-SHIFT DROP. Drop a France link whose house number differs by 3–100 from the S1's when that S1 already has an
   exact-number link: 4,540 links (0.0175/S1) on sub66_final. Gate on India→US LOCO (US-like hn behaviour; France accepted pairs
   are 92% hn-equal, US 78%, India 50%). Use P37/P42 LOCO machinery. Stage output/sub71c if it passes.
C. SMALL: (i) P40's full-name address-less adds where the core is SHARED (sub66_final keeps only P66's unique-core `xfc`); P40
   LOCO p ≥ 0.9 was +0.000077 / +0.000024. (ii) Parser gaps: "Appt/Apt/app" read as a house number (~1,200 France S1) and
   "12 B / 12 T / bis / ter" (1,795 S1, 20.8k SX) — quantify how many France decisions they flip; fix only if cheap and safe.
Combine every arm that passes into output/sub71_best (on sub66_final; India/US byte-identical) with an expected LB each, and
write the SUBMISSIONS.md-style row text into RESULT.md (do NOT edit SUBMISSIONS.md; the lead does). Message the lead session
amc-26-46 with verdict numbers when done (SendMessage). Memory 2 GB local; DISK has only ~2 GB free — keep outputs small,
delete scratch, big intermediates to S3 (s3://sagemaker-114872709824-ap-south-1/amc26-p71/, `--profile harsh` ONLY). Own code in
probes/P71-fr-noiseword/.
