# P32 DAPT kit: domain-adaptive MLM for the bge cross-encoder

**Amazon ML Challenge 2026 data, for team Yeti. Do not share it outside the team.** The data files hold the
competition's records.

## What it does

The run starts from `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M params; downloaded from Hugging Face on the first run). It does three things:

1. **DAPT (masked-LM).** It continues masked-LM pre-training of the encoder on *unlabelled* records ("name | address") of
   one country: 2 epochs, 15% masking, max_len 64, lr 5e-5, bf16. Word embeddings stay frozen, and a fresh LM head is
   warmed up for 300 steps first.
2. **Fine-tune.** It trains the result as a cross-encoder on the labelled US/India band pairs. This is the team's
   P13 recipe exactly: BCE, bs 128, lr 2e-5, 500 warm-up steps, linear decay, max_len 96.
3. **Score.** It scores the band pairs and writes `r = sigmoid(logit)` per pair.

`bash run.sh` runs 5 independent chains, one process per GPU:

| chain | MLM on | fine-tune on | scores | purpose |
|---|---|---|---|---|
| `daptIndia_locoUS` | India | US (1 epoch) | holdout band India+US | LOCO gate, US→India |
| `daptUS_locoIndia` | US | India (1 epoch) | holdout band India+US | LOCO gate, India→US |
| `ctl_locoUS`, `ctl_locoIndia` | none | as above | holdout band | same-GPU no-DAPT controls |
| `daptFrance_pool` | France | India+US (2 epochs) | holdout band + **France band (1,447,625 pairs)** | the France model |

The gate itself (F0.5 with the label-free France threshold) needs the team's full holdout world, so it runs on
the team laptop: `probes/P32-bge-dapt/p32.py gate`. The kit only produces the scores.

## Requirements

- An NVIDIA GPU with **≥ 24 GB** and bf16 support (Ampere or newer). It is tuned for an **H100 80 GB**.
  - On cards ≥ 40 GB, the run keeps the same per-example recipe (bs 128 fine-tune, bs 256 MLM) but drops gradient
    accumulation and uses larger scoring batches (fine-tune micro 128, MLM micro 256, scoring 2,048).
  - On smaller cards it falls back to the A10G settings (64 / 128 / 512).
  - An OOM during MLM halves the micro-batch and carries on.
  - TF32 matmuls are on.
  - There is no DataLoader. Every text is tokenised once up front, and batches are cut from in-RAM arrays, so
    dataloader workers do not apply.
  - `torch.compile` is not used: the length-bucketed batches have variable shapes and would force recompiles.
- About 10 GB of CPU RAM per concurrent chain and about 8 GB of disk: the model cache (2.3 GB) plus checkpoints (0.6 GB each).
- Python 3.12, and internet access for the one-time model download.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # pick the torch wheel for your CUDA first if needed (see the file)
bash run.sh                            # all 5 chains, one process per GPU, every GPU used
bash run.sh --gpus 2                   # use GPUs 0-1 only (or: CUDA_VISIBLE_DEVICES=2,3 bash run.sh)
bash run.sh loco | bash run.sh france  # a subset
```

**Multi-GPU.** Each chain is an independent process pinned to one GPU (`CUDA_VISIBLE_DEVICES=i`).
- The GPUs come from `CUDA_VISIBLE_DEVICES`, else `--gpus N`, else `nvidia-smi -L`, else 1.
- Chains are dealt round-robin, longest first. Each GPU runs its share in order, and with 1 GPU everything runs
  sequentially. The model is downloaded once, before the processes start.
- Each process needs about 10 GB of CPU RAM.

Parameters can be overridden without editing any file, for example
`P32_EXTRA='"dapt_epochs": 1' bash run.sh france`. Any key in `solution_p32.DEF32` or `solution_p13.DEFAULTS` works.

**Progress:**
- The console prints `[gpu i] start / done <chain>` with timestamps.
- Per chain, `tail -f logs/<chain>.log` shows one JSON event per line.
- For all chains at once: `grep -h '"dapt_step"\|"step"\|chain_done' logs/*.log | tail`.
  - MLM progress is `"ev": "dapt_step", "step": k, "of": 12974`, with masked-token accuracy (~0.6 after 1k steps).
  - Fine-tune progress is `"ev": "step", "step": k, "of": N`.
- A chain is finished when its log shows `chain_done`. `run.sh` ends by checking every r file.

## Expected wall time

Speeds measured on the team's AWS runs of this exact script:

| | A10G 24 GB | L40S 48 GB |
|---|---|---|
| MLM | 598 texts/s | 1,234 (India) – 1,547 (US) texts/s |
| fine-tune | 287–367 pairs/s | 762–970 pairs/s |
| scoring | 1,370–1,940 pairs/s | 2,800–4,700 pairs/s |

The H100 column assumes ~2× L40S.

| chain | measured L40S (MLM + fine-tune + scoring) | **H100 (est.)** |
|---|---|---|
| `daptFrance_pool` | not run on L40S. A10G: MLM 90 min + fine-tune 76 min (1.56M pairs, 2 epochs) + scoring 1.68M pairs ~17 min ≈ 3.1 h | **~40 min** |
| `daptIndia_locoUS` | 44 + 8 + 1 = **53 min** | **~27 min** |
| `daptUS_locoIndia` | 35 + 6 + 1 = **43 min** | **~22 min** |
| `ctl_locoUS` / `ctl_locoIndia` | 9 / 7 min | **~5 / ~4 min** |

| H100s | what runs where | wall time (est.) |
|---|---|---|
| 1 | all 5 chains in sequence | **~1.7 h** |
| 2 | GPU0: France → daptUS → ctlIndia; GPU1: daptIndia → ctlUS | ~65 min |
| 3 | GPU0: France → ctlUS; GPU1: daptIndia → ctlIndia; GPU2: daptUS | ~45 min |
| ≥ 5 | one chain per GPU | ~40 min (bound by the France chain) |

Add ~3 min per run for the model download (2.3 GB) and tokenising.

## Outputs (`out/`)

- `out/r/r_<tag>_holdout.parquet` and `out/r/r_daptFrance_pool_test_France.parquet`, with columns `s1, sx, r` (float32).
  The format is the same as the team's `data/work/p13/r/`. `run.sh` checks every file at the end.
- `logs/<chain>.log` is the full console log. `logs/<chain>.jsonl` holds one JSON event per line: speeds, MLM
  loss/accuracy before and after, validation and holdout AUCs.
- `out/daptFrance_pool/ce_daptFrance.pt` holds the France-adapted encoder. `out/daptFrance_pool/ce_daptFrance_pool.pt` holds the fine-tuned France
  model (fp16, word embeddings omitted, in P13's format). Load either with `solution_p13.Job.restore`.

**Send back:** `out/r/` (6 parquet files, about 40 MB) and `logs/*.jsonl` (per-chain event logs, small). Send the `.pt`
checkpoints only if asked.

## Files

- `run.sh`: the one command.
- `solution_p32.py`: MLM + chain driver. Run as `python solution_p32.py DATA OUT '<json params>'`.
- `solution_p13.py`: the team's P13 job (model loading, fine-tune, scoring). It is used unchanged.
- `data/`:
  - `dapt_{France,India,US}.parquet`: column `t`, 1,622,480 unique strings each. France holds every France test
    record. India and US hold every holdout-band record plus random train records, up to the same count.
  - `train_{India,US}.parquet`, `val_*`: `ta, tb, y, p`.
  - `holdout_{India,US}.parquet`: `s1, sx, ta, tb, y, p`.
  - `test_France.parquet`: `s1, sx, ta, tb`. It is the union of every France candidate pair the team ships.
