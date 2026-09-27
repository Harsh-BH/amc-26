# P80 stage 2: out-of-fold India/US bge CE (cross-fit of P13's `ce_bge_pool.pt`)

The shipped India/US combiner reads `l_r_all`, the logit of the P13 bge-reranker-v2-m3 CE (Apache-2.0, 568M). A combiner
trained on train folds needs that score out of fold. This stage trains two CEs with the shipped recipe:

- **CE_A** trains on folds {2,3}.
- **CE_B** trains on folds {4,5}.

Each CE scores every candidate pair of the folds the other one trained on.

## What is trained on what

| | CE_A (`p80ce-A-0927-2141`) | CE_B (`p80ce-B-0927-2149`) |
|---|---|---|
| instance | ml.g6e.2xlarge (1× L40S), on-demand, $3.37/h | ml.g6e.xlarge (1× L40S, 4 vCPU), on-demand, $3.13/h |
| trained on | band pairs of folds {2,3}: India 99,526 / US 159,655 | band pairs of folds {4,5}: India 99,939 / US 159,559 |
| out of fold for | **folds 4, 5** | **folds 2, 3** |
| also scores (neither CE saw them) | folds 0, 1 | folds 0, 1 |
| scored pairs | India 4,991,401 / US 7,588,798 | India 4,997,168 / US 7,580,265 |

- **Recipe.** This is P13's `bge_pool` phase, run unchanged through `solution_p13.py`:
  - 2 epochs, bs 128, lr 2e-5, warm-up then linear decay;
  - max_len 96, length buckets, bf16, word embeddings frozen.
- **Training set.** P13's sampling: the band pairs plus 0.5× as many out-of-band pairs.
  - The band is 0.1 < p < 0.9 of P4b's quick LightGBM (lr .08, 127 leaves, 300 rounds), plus a 5% sample of the other pairs.
  - P4b fit that LightGBM on fold 2, so it cannot label fold 2. Here it is **refit on fold 6**, a fold that is neither trained on nor scored. It then predicts folds 2–5.
  - Band size and positive rate match P4b's: India 33.3k vs 32.8k per fold, US 53.3k vs 54.0k. The two bands share 80–84% of their pairs.
  - Band files are in `data/work/p80ce/band_<c>.parquet`.
- **Scale.** Each CE sees 1/3 of the shipped CE's 778,520 training pairs, so it will be a little weaker. The numbers to compare against are below.
- **Validation (`val`).** 20k band pairs of the scored folds. It is only a monitor: P13 logs its AUC, and nothing is tuned on it.
- **Holdout (`holdout`).** P13's fold-0 band files (78,079 India / 115,672 US), which neither CE saw.
  - This gives a like-for-like AUC against the shipped CE: India 0.9710 / US 0.9564.
  - P59's folds-1–9 CE reached 0.9727 / 0.9585.
- **Scored pairs.** Every P4 X pair, i.e. P3's full candidate set, of the scored folds.
  - That is a superset of any pruned set (#6's pruner, stage C's `p >= PMIN`) and of any argmax set. So downstream can pick its own rows.
  - Folds 6–9 are **not** scored.
  - Folds 0 and 1 are scored by both CEs. Take the mean, as stage 1 does for M2.
- **Texts.** P3's cleaned `name | addr`, built on the instance from `txt/`. They are identical to P4b's `rerank.texts` and ids to `rerank.rows`: all 800 smoke pairs were checked.
- **Leakage.** No S1 of a scored fold is in its CE's training pairs. An SX text can appear in both, paired with a different S1. That is the protocol every matcher shares, stage 1 included.

## S3 layout (`s3://sagemaker-114872709824-ap-south-1/amc26/p80/ce/`)

- `data/`: the channel, uploaded once (880 MB).
  - `keys/{A,B}/{train,val,score}_<c>.parquet`: q1, qx (rows of P3's text files); y and p (train/val) or fold and y (score).
  - `txt/`: P3's train texts.
  - `holdout/`: P13's `sm_data/holdout_<c>.parquet`.
- `results/<job>/`, written as each file finishes:
  - **`oof/oof_<A|B>_<c>.parquet`**: q1, qx, fold, y, s1, sx, r, one row per score key. r = sigmoid(CE logit).
  - `r/r_bge_pool_test_<c>.parquet`: the same r as (s1, sx, r).
  - `r/r_bge_pool_holdout.parquet`: the fold-0 band.
  - `ckpt/ce_bge_pool.pt`: 624 MB, fp16, loads with `solution_p13.Job.restore`. Use it to score folds 6–9 later with P13's `bge_ckpt` phase or P45's `solution_p45c.py`.
  - `log.jsonl`: speeds, val and holdout AUC.
- `output/<job>/output/model.tar.gz`: a backup copy of the same r / oof files.

## Runtime and cost

- Measured on L40S in P59: train 804 pairs/s; score 3,400 (India) / 4,900 (US) pairs/s; tokenise ~30k pairs/s.
- Expected per job:
  - ~11 min training (518k pair-passes);
  - ~50 min scoring (12.6M pairs);
  - ~15 min text build, tokenisation, container start and model download;
  - **total ≈ 75–85 min**.
- **Expected cost ≈ $4.5 (A) + $4.7 (B, slower tokenising on 4 vCPU) ≈ $9.**
- MaxRuntime is 10,800 s (P13 deadline 9,600 s), so the worst case is 3 h × ($3.37 + $3.13) = **$19.5**. The budget is $40.
- `p80ce-B-0927-2141` (g6e.4xlarge) waited for capacity and was stopped before it started, so it was never billed. B was relaunched on g6e.xlarge as `p80ce-B-0927-2149`.
- If the deadline planner skips a country, `log.jsonl` has a `skip` event and that `oof_*` file is missing.

## Fetch and merge

```bash
B=s3://sagemaker-114872709824-ap-south-1/amc26/p80/ce/results
aws s3 cp $B/p80ce-A-0927-2141/log.jsonl - --profile harsh | grep -E 'val_auc|holdout_auc|train_done|skip|all_done'
for j in p80ce-A-0927-2141 p80ce-B-0927-2149; do
  aws s3 cp --recursive $B/$j/oof/ data/work/p80ce/oof/ --profile harsh    # ~4 files, ~150 MB each with ids
done
```

```python
import numpy as np, pandas as pd
c = "India"
o = pd.concat([pd.read_parquet(f"data/work/p80ce/oof/oof_{s}_{c}.parquet").assign(split=s) for s in "AB"])
oof = o[o.fold.between(2, 5)]                                    # each pair exactly once: A on 4-5, B on 2-3
ext = o[o.fold <= 1].groupby(["q1", "qx"], as_index=False).agg(  # folds 0-1: logit mean of A and B
    s1=("s1", "first"), sx=("sx", "first"), fold=("fold", "first"), y=("y", "first"),
    l=("r", lambda r: np.log(r.clip(1e-4, 1 - 1e-4) / (1 - r.clip(1e-4, 1 - 1e-4))).mean()))
```

- The join key to stage 1's `oof_*` (q1, qx, fold, is_true, p) is **(q1, qx)**. It is the same P4 X row space.
- The join key to P45 / #8c frames is **(s1, sx)**.
- `l_r_all` = logit(clip(r, 1e-4, 1 − 1e-4)), as in P45.
- Disk: pull one country at a time, or read straight from S3 with `pd.read_parquet("s3://…")`, which needs s3fs, or via boto3. The laptop has about 2 GB free.

## Files

- `build_ce.py`: `band <c>`, `keys` and `static` build and upload the channel; `smoke` runs the local CPU rehearsal. `smoke` uses the exact launch hyperparameters and only swaps in e5-small. It passed: the whole bge_pool path plus the oof post-process, on 400 pairs per file.
- `solution_p80ce.py`: the instance side. It builds the texts, calls `p13sol.main`, then writes the oof files.
- `launch_ce.py <A|B> <instance> [--dry-run]`: sm_launch with P13's pinned bundle. Always `--profile harsh`.
- `JOBS`: the launched jobs.

## Progress (27 Sep 22:05)

- **CE_A has finished training.** It ran the full recipe: 2 epochs, 4,048 steps, 652 s at 800 pairs/s, final loss 0.1835.
  - Out-of-fold val AUC on the fold 4–5 band: r India 0.960 / US 0.914. The band GBDT scores 0.821 / 0.793.
  - Fold-0 holdout band: AUC r **India 0.9623 / US 0.9479**, against the shipped CE's 0.9710 / 0.9564. The stack with p reaches 0.9746 / 0.963 (shipped: 0.977 / 0.967).
  - So the 1/3-data CE is weaker by about 0.009 AUC. Keep that in mind when a combiner trained on these OOF scores is applied to the shipped CE's test r.
  - Checkpoint: `results/p80ce-A-0927-2141/ckpt/ce_bge_pool.pt`. Scoring is running now.
- **CE_B has passed its build and self-check and is training.** Peak memory so far is 29% of the xlarge's 32 GB.

## Stage 2b: stage C's missing rows scored by both checkpoints (launched 22:37)

- **Keys.** They are in `s3://…/amc26/p80/ce/score_data/keys/<c>/` (built by `build_ce.py score_channel`).
  - `train69`: the `sc_<c>` frame rows of folds 6–9, with columns q1, qx, fold. India 1,488,866 / US 2,230,697.
  - `test`: every test X pair whose cross-fit M2 p is ≥ 0.01 (from the `p80-sct` files; the same set `stagec.apply_test` builds), with columns q1, qx. India 3,599,994 / US 2,966,683.
- **How a job scores its slice.** Each job takes a contiguous slice `part = [k, n]` of train69 + test.
  - It runs P13's `bge_ckpt` path once per checkpoint (A = `p80ce-A-0927-2141`, B = `p80ce-B-0927-2149`), sharing tokens between the two passes.
  - It writes `results/<job>/score/score_<c>_<k>of<n>.parquet`, with columns q1, qx, fold (−1 = test), s1, sx, r_A, r_B and r.
  - r = sigmoid of the mean logit, with each r clipped to [1e-6, 1 − 1e-6] before the logit.
  - Per-checkpoint P13 files are under `results/<job>/{A,B}/`.
- **Jobs** (the full list is in `JOBS`):
  - US halves 0of2 and 1of2 run on g5.4xlarge and g5.2xlarge (A10G).
  - India halves 0of2 and 1of2 run on g6.4xlarge and g6.2xlarge (L4). Their g6e twins never got capacity and were stopped unbilled.
  - L4 scores India at ~915 pairs/s, which puts those halves at ≈ 23:55. So India quarters 0–3of4 were launched as a hedge on g5.8xlarge, g5.16xlarge, g6.8xlarge and g6.16xlarge.
  - Use whichever complete India set lands first: {0of2, 1of2} or {0of4 … 3of4}. Both are keyed by (q1, qx).
- **Checks.** The CPU smoke test (`build_ce.py smoke_score`) passed. On it, ids = `rerank.rows` and texts = `rerank.texts` for both train and test, and r_A ≠ r_B.
