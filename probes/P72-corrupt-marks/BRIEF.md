# P72-corrupt-marks: raw-text corruption marks as combiner features (India/US + France) + India/US unique-key recovery
Launched 27 Sep ~15:20 IST by the user's second session (lead = amc-26-46 decides uploads). Evidence: docs/RIVAL_INTEL.md.
Base: output/sub66_final (LB 0.987391). India/US = sub54's combiner (P44 + P45 r_all + zoo, P50-E full refit; src iu_stack.py);
France = P55 combiner chain (src/france.py). Public rank ~117 of 6,938; top-100 cut 0.987804 and rising → small gains count.
The leader must beat us on India/US by ≥ 0.0018 (LB = 0.8502·IU + 0.1498·FR), so India/US levers matter too.
DEADLINE: verdict + staged file by 19:30 IST. Never end your turn to wait.

A. CORRUPTION MARKS. Our normaliser lowercases/folds accents BEFORE any feature, so the combiners never see how a record was
   written. Measured on train, owned SX vs distractor SX: all-lowercase name 5.3–8.0% vs 1.8–3.9%; all-caps name (S2) 16–23% vs
   11–16%; accents injected 5–7.5% vs 2.7–4.9%; name tokens 3.1–3.5 vs 3.9–4.2; empty address 3.8–4.9% vs 0.2–0.3%; DBA/aka only on
   copies (0.9985–0.9996 linked); domain-style names 0.961–0.964 linked. Distractor mutations: a name word replaced by another real
   vocabulary word (0.34 of non-matches vs 0.05 of matches); "public" only on the SX side (1.6% vs 0%); numbers change by small
   arithmetic steps (12590→12594) whereas true copies truncate (4712→471).
   Build per-SX and per-pair features from RAW text: case pattern (lower/upper/title/mixed) and whether it differs from S1's,
   accents added/removed vs S1, token count and delta vs S1, punctuation/bracket/leading-symbol noise, DBA/aka/formerly markers,
   domain-style name, number-change type (truncation vs arithmetic step), vocabulary-swap indicator. Add them to (1) the India/US
   combiner — gate on the holdout halves exactly like P44/P45 (fit h0 → report h1 and swap; gate ≥ +0.00015 LB-eq at ≥ 3 SE both
   ways), and (2) the France combiner — gate on both LOCO directions (P55's harness / P65's ev65.py reproduces P55 exactly).
B. INDIA/US UNIQUE-KEY RECOVERY (a rival measured +0.000178 ± 0.000033 on a weaker base): for band pairs with 0.30 ≤ p < 0.70,
   no shared name token and a non-empty SX address → shared address tokens minus street/unit words; if exactly ONE S1 in the
   country has an address containing all of them, accept. For 0.10 ≤ p < 0.70 and an empty SX address → shared name tokens minus
   legal/filler words; if exactly one S1 name in the country contains all of them, accept. Gate on the holdout halves as above.
If anything passes, stage output/sub72 (= sub66_final with only the passing part changed), validator --check-ids, expected LB,
row text for SUBMISSIONS.md in RESULT.md (do NOT edit SUBMISSIONS.md). Message the lead session amc-26-46 with the numbers.
Compute: the combiner refits are heavy — run them on SageMaker CPU (ml.m7i.4xlarge / c7i.4xlarge on-demand, quota 2 each;
m7i.2xlarge 15) via the sm_launch.py / P50 e_zoo pattern, `--profile harsh` ONLY, budget $6. Local memory 1.5 GB; DISK has only
~2 GB free — big frames to S3 (s3://sagemaker-114872709824-ap-south-1/amc26-p72/). Own code in probes/P72-corrupt-marks/.
