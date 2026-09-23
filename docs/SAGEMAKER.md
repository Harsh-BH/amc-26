# SageMaker training framework

Two files. `sm_train.py` runs **inside** SageMaker and owns the filesystem contract;
`sm_launch.py` runs **on your laptop** and submits the job. You write `solution.py`.

Built on **boto3 only** — the `sagemaker` SDK depends on torch + CUDA and filled this disk once.

## Day 1, in order

```bash
python sm_launch.py --profile amc --create-role     # once. This is the usual day-1 blocker.
python sm_launch.py --profile amc --dry-run --data-s3 s3://x/   # prints + validates, spends nothing
python sm_launch.py --profile amc --data data/ --instance ml.g4dn.xlarge
python sm_launch.py --profile amc --status <job-name>
python sm_launch.py --profile amc --fetch  <job-name>
```

`--dry-run` validates the payload against botocore's own schema, so typos surface in a
second instead of after a five-minute job spin-up.

## What you write

`solution.py`, next to `sm_train.py`. Either:

```python
def fit(train_dir, params):          # train_dir is a pathlib.Path
    ...
    return model                      # pickled to SM_MODEL_DIR/model.pkl

def predict(model, test_dir):         # optional
    return df                         # written to SM_OUTPUT_DATA_DIR/submission.csv
```

or a single `main(train_dir, test_dir, model_dir, out_dir, params)` if you want full control.

Hyperparameters arrive as `--key value` and are JSON-parsed, so `--epochs 3` gives you an
`int`, `--lr 0.001` a `float`, and a bare `--flag` a `True`. Pass them with
`--hp '{"epochs": 3, "lr": 0.001}'`.

## Spot is on by default

Managed spot training bills at the spot price — typically ~70% off, so your $200 stretches
roughly 3×. SageMaker handles interruption and restart, and `CheckpointConfig` is wired to
`s3://<bucket>/<prefix>/checkpoints/`.

**If your job runs longer than about an hour, checkpoint inside `solution.py`.** An
interruption restarts the container from scratch otherwise, and you will lose the run.
Turn it off with `--no-spot` when you need a guaranteed-uninterrupted final run.

## Instances

| Instance | GPU | $/hr ap-south-1 | Hours per $200 |
|---|---|---|---|
| `ml.g4dn.xlarge` | T4 16 GB | $0.811 | 247 |
| `ml.g5.xlarge` | A10G 24 GB | $1.691 | 118 |
| `ml.m5.xlarge` | none | $0.242 | 826 |

Those are on-demand; spot is roughly a third. The launcher picks the CPU image automatically
for `ml.m*`, `ml.c*` and `ml.t*`, the GPU image otherwise.

Image: `763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:2.9-gpu-py312`
(account verified against `sagemaker-core`'s image_uri_config). Override with `--tag`.

## Traps

- **The IAM role.** Nothing works until `--create-role` has run. Do it before the 25th.
- **Billable ≠ wall clock.** With spot, check `BillableTimeInSeconds` in `--status`, not
  elapsed time — queue time is free.
- **`MaxWaitTimeInSeconds` must be ≥ `MaxRuntimeInSeconds`** when spot is on. The launcher
  enforces this, but if you raise `--max-run` past `--max-wait` it silently lifts the wait.
- **Training jobs, not notebooks.** A notebook instance bills while it sits idle overnight.
  A training job bills only while it runs. This is the single biggest credit leak.
- Data is uploaded file-by-file. For 250k images, tar them first and untar in `fit()`, or
  the upload alone will cost you an hour.
