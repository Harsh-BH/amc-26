"""P43 SageMaker job: the official validator with --check-ids on a 16 GB CPU box (the laptop cap is 1.5 GB).
train channel = validate_submission.py + either Yeti_submission.zip (validated as unzipped) or the two TSVs;
test channel = test_source{1,2,3}.tsv. Prints the validator's output and validator_exit=<code>.
    .venv/bin/python sm_launch.py --profile harsh --solution probes/P43-final-package/solution_p43.py \
        --data-s3 s3://sagemaker-114872709824-ap-south-1/amc26-p43/val_small \
        --test-s3 s3://sagemaker-114872709824-ap-south-1/amc26-p43/test --instance ml.m5.xlarge --name p43-val-small
"""
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path


def main(train_dir, test_dir, model_dir, out_dir, params):
    d, work = Path(train_dir), Path("/tmp/p43")
    work.mkdir(parents=True, exist_ok=True)
    z = d / "Yeti_submission.zip"
    if z.exists():
        with zipfile.ZipFile(z) as f:
            print("zip members:", len(f.namelist()), sorted(f.namelist())[:8], "...", flush=True)
            f.extractall(work)
        out = work / "output"
    else:
        out = d
    for n in ("matching_results.tsv", "candidate_pairs.tsv"):
        print(n, hashlib.md5((out / n).read_bytes()).hexdigest(), flush=True)
    r = subprocess.run([sys.executable, str(d / "validate_submission.py"), "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(test_dir), "--check-ids"],
                       capture_output=True, text=True)
    print(r.stdout[-6000:], r.stderr[-3000:], sep="\n", flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "validate.log").write_text(r.stdout + r.stderr)
    print(f"validator_exit={r.returncode}", flush=True)
    assert r.returncode == 0, "official validator failed"
