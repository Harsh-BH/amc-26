# P51-ce-serialize — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Drop: the Ditto-style serialization helps the bge CE out of domain, but it misses the bar and loses to Qwen.** France procedure (P38's gate, mean-OOD weights), serialized bge − raw bge: **+0.00053 ± 0.00019 (US→India), +0.00038 ± 0.00012 (India→US)**. The bar was ≥ +0.0010 in both directions.
- Unseen-country band AUC r rises a lot, from 0.822 / 0.829 to **0.865 / 0.858** (Qwen3-4B: 0.856 / 0.869), but the F0.5 gain is less than half of Qwen's. Serialized − Qwen: −0.00087 / −0.00071. Adding it to Qwen (logit mean) gives +0.00014 ± 0.00012 / +0.00002 ± 0.0001. In-domain AUC is unchanged (US 0.9535 vs 0.9525, India 0.9666 vs 0.9675).
- No pooled job, no sub51, no India/US column. For France's CE, P38's Qwen stays the candidate. Spent **$2.74** of the $30 budget.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Adopt nothing into the pipeline.** Steps 4(a) and 4(b) were conditional on the gate and were not run. P50 gets no change set.
- **Reusable pieces, if anyone needs them:**
  - `probes/P51-ce-serialize/sub51.sh <r.parquet> <a,b,c> <tag> [BASE]` rebuilds France with any band CE r and OOD weights on the current stack: build38 blend → tpick → P31 post → on43 → P46 swap → splice into BASE.
  - `veto51.py` re-applies P48's veto with that chain's own r and blend.
  - Both were verified on bge inputs: 2 France lines differ from output/sub44r, and 23 from output/sub48 (see Evidence).
  - A Qwen France rebuild on the current best could use them:
    ```
    sub51.sh data/work/p38/r/r_qwen_pool_test_France.parquet 0.8943,0.3729,-0.0521 qwen output/sub48
    veto51.py … <qwen r>
    splice.py … <veto>
    ```
    Caveat: P48's veto threshold (0.2) was picked on bge's r distribution.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| **gate51-ser_locoUS_holdout+ser_locoIndia_holdout-vs-bge_locoUS_holdout+bge_locoIndia_holdout** | holdout-dense-half1 | P13's France procedure on LOCO (source in-domain blend, mean-OOD weights on the target, shipped rule, P6's label-free t), paired per S1 on the target's half 1. **Serialized − raw bge: US→India +0.00053 ± 0.00019 (627 S1 up / 539 down), India→US +0.00038 ± 0.00012 (471 / 654)**; cross-fit weights +0.00047 ± 0.00018 / +0.00046 ± 0.00014. Over no-CE: raw bge +0.0028 / +0.00148, **serialized +0.00333 / +0.00187**. Raw + serialized logit mean − raw: +0.00061 / +0.00038. Serialized mean-OOD w [0.8706, 0.3079, −0.0785] (raw bge [0.9241, 0.2595, 0.0516]); t_dst 0.78 / 0.76 | **drop** (bar +0.0010 both ways) |
| gate51-ser_locoUS_holdout+ser_locoIndia_holdout-vs-qwen_locoUS_holdout+qwen_locoIndia_holdout | holdout-dense-half1 | Same procedure, control = P38's Qwen3-Reranker-4B LoRA LOCO. **Serialized − Qwen: −0.00087 ± 0.00019 / −0.00071 ± 0.00014.** Qwen + serialized logit mean − Qwen: **+0.00014 ± 0.00012 / +0.00002 ± 0.0001** (ens over no-CE +0.00434 / +0.0026 vs Qwen +0.0042 / +0.00257) | drop |
| p51-loco-band-auc-indomain-vs-ood | holdout band | Band AUC r, unseen country (US→India / India→US): raw bge 0.8219 / 0.8288, **serialized 0.8654 / 0.8583**, Qwen 0.8564 / 0.8688. In-domain (US / India): raw 0.9525 / 0.9675, serialized 0.9535 / 0.9666, Qwen 0.9582 / 0.9690 | drop (no in-domain gain → no r_all column) |
| (gate code check, P51_NOLOG) | holdout-dense-half1 | `p51.py gate` with raw bge in both arms: difference exactly 0; bge over no-CE +0.0028 / +0.00148 = P13's frproxy; 52 s, 1.54 GB | pass |
| (sub51 chain check) | test-France | `sub51.sh bge <P13 W_FR>` vs output/sub44r: build38 blend → tpick 0.9868 → P31 post → on43 (t42 0.99) → P46 swap lens (+3,185) → splice. **2 France lines differ** (= P38's repro43 gap). `veto51.py` + splice into output/sub48: 1,159 vetoed (rescue 44 / gc 1,115) vs P48's 1,138 (41 / 1,097), so **23 France lines differ** from sub48 at set level (byte order inside lines differs; P46 appends unsorted) | pass |
| (serialize self-check) | synthetic | France '(8) R. Konrad Adenauer' → hn 8 / street 'adenauer konrad r' / région; India 'Pvt Ltd' + MH; US LLC + IL; gib brand + noaddr flags | pass |
| (token lengths, 3k train pairs) | train | bge tokens per pair, serialized vs raw: India p50 119 vs 54, p99 172 vs 88; US p50 92 vs 37, p99 124 vs 54. maxlen 160 truncates 3.3% India / 0% US (longest_first cuts the trailing raw name first) | – |
| (channel check) | holdout | Holdout rows = P13's LOCO r files' rows exactly (193,751). Train India 294,766 / US 483,754 = p13.build's selection (seed 0). Job token means: India 116 / US 90.5 | pass |
| (LOCO job logs) | holdout band | US: 483,712 pairs at 501 pairs/s on one L40S (936 s, micro 64 at L 160, 17.6 GB). India: 294,656 pairs at 120 pairs/s on one L4 (2,333 s); scoring 420–560 pairs/s on the L4 vs ~3,200 on the L40S. Mid checkpoints (50%): US→India 0.8536, India→US 0.8573 | – |

## Jobs (AWS, --profile harsh, ap-south-1, on-demand)
| job | instance | purpose | status | billable s | $ |
|---|---|---|---|---|---|
| p51-us-g6e2-0927-0421 | ml.g6e.2xlarge | LOCO: train US (1 epoch) → holdout band (mid + end) | Completed | 1,417 | 1.33 |
| p51-in-g6x-0927-0421 | ml.g6.xlarge (1× L4) | LOCO: train India (1 epoch) → holdout band (mid + end) | Completed | 3,588 | 1.35 |
| p51-us-g62-0927-0421 | ml.g6.2xlarge | US duplicate | stopped at Downloading | 41 | 0.02 |
| p51-in-g54-0927-0421 | ml.g5.4xlarge | India duplicate | stopped at Downloading | 66 | 0.04 |
| p51-in-g6e4-0927-0421 | ml.g6e.4xlarge | India duplicate | stopped while Pending | 0 | 0 |
| (not created) | ml.g5.xlarge | India duplicate | refused: quota | 0 | 0 |

Total **$2.74** (Pricing API: g6e.2xlarge $3.37/h, g6.xlarge $1.353/h). S3: `amc26-p51/data/{v1,pool}/` (~100 MB) and `amc26-p51/rall/`. The latter holds serialized texts for P45's r_all rows (holdout fold-0 argmax + 6.6M test argmax pairs, ~1 GB), built ahead of the gate and now unused. Delete it if S3 cost matters.

## What failed      (approaches tried and dropped, with the number that killed them)
- **The serialized bge as France's CE:**
  - +0.00053 / +0.00038 over raw bge. The bar was +0.0010.
  - The AUC gain (+0.03–0.04 on the unseen country, Qwen-sized) does not become Qwen-sized F0.5: Qwen gets +0.00141 / +0.00109 from a similar AUC.
  - Reading: the extra ranking quality sits away from the pairs that decide the France threshold.
- **Serialized + Qwen ensemble:** +0.00014 / +0.00002 over Qwen alone. It is redundant with Qwen.
- **Serialized + raw bge ensemble:** +0.00061 / +0.00038 over raw. Fails the bar.
- **India/US r_all column (step 4(b)):** not run. In-domain band AUC is unchanged vs raw bge, so a serialized r_all would duplicate P45's column at the cost of ~8M CE pairs.
- **Smoke job:** skipped by decision. Instead: P13's unchanged job, its in-job selfcheck, and the first-minute events of the real jobs. Nothing crashed.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- **`serialize.py` — `serialize(df[name, business_name, business_address, country], M) -> np.ndarray[str]`**
  - Output format: `name: <name_core> | legal: <canonical legal forms> | hn: <hn> | street: <street_key> | city: <city> | area: <other segments> | state: <code> | pin: <postcode> | brand | noaddr | raw: <P3 Latin name>`.
  - Empty fields are omitted; the flags appear only when true.
  - Parsers used: normalize.name_core / name_norm / _is_legal / address_fields, france.paren_hn / street_key, and P36 gib.is_gib (with the split's learned syllables).
  - No country branching: the label only picks normalize's state table. Self-check: `python serialize.py`.
- **`build51.py`**
  - `build [India,US]`: P13's train/val/holdout pairs, re-texted (~1 min per country, 1.3 GB).
  - `test <c> [src]`: a test band, chunked. France: P31's file band, 319,752 pairs in 36 s, 1.1 GB.
  - `pairs <split> <c> <src> <out> [holdout_am]`: any (s1, sx) file. 6.6M pairs in ~15 min at 1.4 GB.
  - `show`: 20 examples per country → data/work/p51/examples.txt (data rows, so not in git).
- **`launch51.py <phases> <instance> <job> [hp.json]`**
  - Runs P13's solution_p13.py + bundle on `amc26-p51/data/v1/` (or `data_s3` in the hp file).
  - Hyperparameters come from `hp/loco.json` (maxlen 160, cap 9,000 s so the full epoch runs) or `hp/pool.json` (2 epochs, France band test file).
  - Skips sm_launch's service-quotas pre-check, which is throttled when probes launch together.
- **`p51.py`**
  - `fetch <job> [tag=ser]`.
  - `gate <arm_US> <arm_India> [control_US control_India]`: P38's gate with its own ledger row. ~1 min, 1.5 GB. `P51_NOLOG=1` for code checks.
- **France chain helpers**
  - `sub51.sh <r|bge> <w> <tag> [BASE]`: ~2 min, 1.7 GB peak (on43).
  - `veto51.py POST BLEND R ANCHOR OUT [v]`.
  - `splice.py FRANCE_SUB BASE OUT [VETO]`: asserts same S1 order, only France lines differ, and matches ⊆ BASE's candidates.
- **`poll.sh <jobs…>`**: job status + log events.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licence / rules:** same model (bge-reranker-v2-m3, Apache-2.0, 568M). The serializer uses only repo parsers and label-free learned tables (fr_regions.json, P36 syllables). No external data.
- **Assumption: empty fields omitted.** This was a token-budget choice: serialized text is already 2.2–2.5× raw.
- **Assumption: the serializer was frozen at launch.** Known parse leftovers stayed in: 'N° 5' left in `area` as 'n 5 …', and zero-padded '08068' is not matched to hn 8068. Fixing them would have needed new LOCO runs, and the gate gap (−0.0005) is too large for these to close.
- **Assumption: one epoch, same recipe as P13's LOCO.** Longer serialized inputs might need more steps. The mid → final AUC trend was still rising (0.854 → 0.865 US→India), but the F0.5 gap to the bar is 2×.

## Open questions   (for the master; each with your best guess)
- **Serialization + Qwen (instead of bge)?** Best guess: not worth it now. Serialization adds nothing on top of Qwen in the ensemble, a Qwen LOCO on 2.3× longer inputs is ~3–4 h per direction, and the France CE headroom looks small.
- **Delete `amc26-p51/rall/` (~1 GB of unused texts)?** Best guess: yes, after the deadline. It costs cents.
