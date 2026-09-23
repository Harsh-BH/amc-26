# SageMaker training framework

`sm_train.py` runs **inside** the container and owns the SageMaker filesystem contract.
`sm_launch.py` runs **on your laptop**: packs, uploads, submits, waits, fetches.
You write `solution.py`. boto3 submits the request; the `sagemaker` SDK only resolves the image URI.

## ⚠️ Quotas first — this blocks everything

**New accounts have every SageMaker training quota at 0**, CPU included. Verified on a real
account: 338 training quotas, all 0. AWS's own defaults in ap-south-1 are 0 for `ml.m5.large`
on-demand and **0 for every GPU**; only CPU spot defaults to 4.

A request for 1 CPU instance was **not** auto-approved — it opened a support case. Assume
hours to days. **Request on the challenge account the moment it exists.**

```bash
python sm_launch.py --check --instance ml.g4dn.xlarge            # prints the exact request command
aws service-quotas request-service-quota-increase --service-code sagemaker \
    --quota-code L-3F53BF0F --desired-value 1 --region ap-south-1   # g4dn.xlarge on-demand
aws service-quotas request-service-quota-increase --service-code sagemaker \
    --quota-code L-944F78BB --desired-value 1 --region ap-south-1   # g4dn.xlarge spot
```

Spot quotas must be requested *above* the AWS default — `--check` computes the right value.
Until quota lands, `--local` is how you run.

## Day 1, in order

```bash
python sm_launch.py --create-role                                   # once per account
python sm_launch.py --check --instance ml.g4dn.xlarge               # role, bucket, quota
python sm_launch.py --data data/train --test-data data/test --local # real container, free
python sm_launch.py --data data/train --test-data data/test --wait  # the real thing
python sm_launch.py --fetch <job>                                   # -> out/<job>/{model.pkl, submission.csv}
```

Also: `--status <job>`, `--logs <job>`, `--dry-run`, `--solution solution_v2.py`,
`--hp '{"epochs": 3}'`, `--no-spot`, `--instance ml.g5.xlarge`.

## `--local` — the free rehearsal

Runs your job in the **actual SageMaker PyTorch container** via Docker, with the same
`/opt/ml` layout and config files SageMaker writes. Same toolkit, same requirements install,
same hyperparameter path. If it passes locally it will pass on SageMaker, minus the GPU.

It always uses the **CPU** image (~1.7 GB compressed, 5 GB on disk; the GPU one is ~10 GB).
It logs in to the DLC registry and pulls on first use. `docker rmi` it when disk is tight.

## What you write

```python
def fit(train_dir, params):          # pathlib.Path, dict
    ...
    print(f'smape={score}')           # captured as a SageMaker metric
    return model                      # pickled to model.pkl (return None to save it yourself)

def predict(model, test_dir):         # optional -> submission.csv
    return df
```

or `main(train_dir, test_dir, model_dir, out_dir, params)` for full control.
`params['checkpoint_dir']` is always set — write checkpoints there, resume if present.
Other top-level `*.py` files ship too, so `from metrics import smape` works in the container.

Run the same code on your laptop with `python sm_train.py --epochs 3` — it reads
`data/train`, `data/test` and writes `out/local/` when it is not on SageMaker.

## The container

`pytorch-training:2.9.0-cpu/gpu-py312`, both verified to exist in ap-south-1.

| Preinstalled | Not preinstalled → `sm_requirements.txt` |
|---|---|
| torch **2.9**, numpy, pandas 3.0.5, scikit-learn, scipy, pillow, boto3 | lightgbm, transformers, sentence-transformers, timm |

`sm_requirements.txt` is pip-installed at job start (~20 s for those four). **Never list torch
or anything preinstalled** — re-resolving torch adds a ~2 GB download to every job start.
Your laptop runs torch **2.14**: `state_dict`s move fine, a pickled `nn.Module` may not.

## Hyperparameters are lossless

Every value is JSON-encoded, and `sm_train` reads `/opt/ml/input/config/hyperparameters.json`
directly — **not argv**. The toolkit rebuilds argv with `str()`, and that was verified to turn
`false` into `'False'` (truthy — AMP silently on), split `[128, 256]` on the space, and turn
`null` into a bare flag. Through the JSON file all of them arrive intact.

## Spot, instances, cost

Spot is on by default (~70% off). `MaxWait ≥ MaxRuntime` is enforced; checkpoints sync to
`s3://<bucket>/amc26/checkpoints/<job>/`. Over ~1 h, **checkpoint** or an interruption restarts
from zero. Use `--no-spot` for a final run that must not be interrupted.

| Instance | GPU | on-demand $/hr ap-south-1 | Hours per $200 |
|---|---|---|---|
| `ml.g4dn.xlarge` | T4 16 GB | $0.811 | 247 |
| `ml.g5.xlarge` | A10G 24 GB | $1.691 | 118 |
| `ml.m5.xlarge` | — | $0.242 | 826 |

Check `BillableTimeInSeconds` in `--status`, not wall clock — spot queue time is free.

## Traps

- **Quota** — see top. The single thing most likely to cost you day 1.
- **Training jobs, not notebook instances.** Notebooks bill while idle overnight.
- A role created seconds ago can't be assumed yet; `submit()` retries that automatically.
- Uploads run 16-way parallel, but 250k small images is still slow — tar them and untar in `fit()`.

## What was verified, and how

| Layer | How |
|---|---|
| Launcher + entrypoint logic | `python test_sm.py` — 29 offline tests |
| Container contract | `--local` in the real DLC: requirements install, hyperparameter types, shipped modules, metric lines, checkpoints dir, submission, the failure file |
| AWS | real IAM role (trust + policies), real S3 round-trip byte-identical, real `CreateTrainingJob` reaching the quota check |
| **Not yet** | a job actually running on SageMaker — waiting on quota |
