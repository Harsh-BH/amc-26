# P73-fr-independent: a France model NOT trained on India/US labels — French-pretrained CE + France-only data (user idea)
Launched 27 Sep ~15:20 IST by the user's second session (lead = amc-26-46 decides uploads). Evidence: docs/RIVAL_INTEL.md.
User hypothesis: every France score comes from models trained on India/US labels (M2, bge CE, Qwen LoRA), which import India/US
decision rules (e.g. "an added business word marks a decoy") that are wrong for France. Test an INDEPENDENT France scorer.
Base: output/sub66_final (LB 0.987391). France = P55 combiner chain (src/france.py); France est. F0.5 ≈ 0.97.
COORDINATE: P67-fr-synth (session "P67-fr-synth") is already building SYNTHETIC French pairs from France test S1 (gen67.py) and
fine-tuning OUR India/US bge on them (+ replay). Reuse its generator/data (ask it via SendMessage for paths/S3 keys); do NOT
duplicate its arm. P73's difference: NO India/US labels anywhere in training, and a French-pretrained backbone.
DEADLINE: verdict by 19:30 IST, staged file by 20:30. Never end your turn to wait.

1. Backbone (licence MUST be MIT/Apache-2.0, ≤ 8B; verify on the HF card and record it): first choice a French cross-encoder
   such as antoinelouis/crossencoder-camembert-base-mmarcoFR (check licence), else almanach/camembertv2-base or camembert-base
   (MIT) with a fresh 1-logit head; control = the BASE multilingual BAAI/bge-reranker-v2-m3 (NOT our India/US-tuned checkpoint).
   Pull weights on AWS (HF_HUB_DISABLE_XET=1), not the laptop (slow link, 2 GB disk free).
2. France-only training data: (a) P67's synthetic French pairs (positives = generator ops on France S1; negatives = real hard
   French neighbours: same street / shared core / same city, and synthetic copies of a different S1); (b) France test
   pseudo-labels from P61's agreement rule (combiner & Qwen ≥ 0.7 → positive, both ≤ 0.3 → negative; hard negatives kept).
   No India/US labelled pairs. 1–2 epochs, lr ~2e-5 (camembert) / 5e-6 (bge), max_len 96, "name | address" per side.
3. Validation without France labels (LOCO analogue of "independent"): run the SAME recipe on a labelled country as if unseen —
   train only on that country's synthetic pairs + its pseudo-labels (from the model trained on the OTHER country), with the base
   (not India/US-tuned) backbone, and measure on its real labels whether the independent CE, as a column in P55's combiner
   (P65's ev65.py harness reproduces P55), beats the transfer CE. Gate: US-as-target ≥ +0.0003 France-F-equivalent AND India ≥ 0.
   (An English/multilingual backbone stands in for "native-language pretrained" on US.)
4. If it passes: score France band + confident pairs with the independent CE → output/sub73 = sub66_final with France re-decided
   (India/US byte-identical), validator --check-ids, expected LB (dF × 0.1498), row text in RESULT.md (do NOT edit SUBMISSIONS.md).
5. Optional arm if GPU time remains: Qwen on ALL France pairs (not only the band) as a two-model veto of confident France pairs
   (veto when bge AND Qwen are low; re-admit P30 vetoes Qwen strongly endorses) — a rival's largest France gain (+0.00106 LB)
   came from scoring every France pair with a Qwen3-4B LoRA. LOCO-gate it with P38's LoRAs.
Compute: SageMaker GPU on types P67 is NOT using: ml.g5.2xlarge, ml.g5.8xlarge, ml.g6.4xlarge, ml.g6.2xlarge, ml.g4dn.* (quota 1
each; P67 holds g5.4xlarge, g6e.8xlarge, g6e.16xlarge). sm_launch.py / P59–P67 launch code, `--profile harsh` ONLY, budget $12.
Local memory 1.2 GB; DISK ~2 GB free — everything big lives in S3 (s3://sagemaker-114872709824-ap-south-1/amc26-p73/).
Message the lead session amc-26-46 with the verdict. Own code in probes/P73-fr-independent/.
