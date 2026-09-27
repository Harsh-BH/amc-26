"""P38: the official validator with --check-ids on a 16 GB CPU box (the laptop cap is 1.6 GB), one run per
matching_<name>.tsv in the train channel against its candidate_pairs.tsv; test channel = test_source{1,2,3}.tsv
(P43's s3://…/amc26-p43/test). Prints validator_exit=<code> per file; the job fails if any file fails.
    .venv/bin/python sm_launch.py --profile harsh --solution probes/P38-qwen4b-fr/solution_val.py --no-spot \
        --data-s3 s3://…/amc26-p38/val/ --test-s3 s3://…/amc26-p43/test/ --instance ml.m5.xlarge --name p38-val-<ts>
"""
import hashlib
import subprocess
import sys
from pathlib import Path


def main(train_dir, test_dir, model_dir, out_dir, params):
    d, log, bad = Path(train_dir), [], []
    cand = d / "candidate_pairs.tsv"
    print("candidate_pairs.tsv", hashlib.md5(cand.read_bytes()).hexdigest(), flush=True)
    for m in sorted(d.glob("matching_*.tsv")):
        r = subprocess.run([sys.executable, str(d / "validate_submission.py"), "--matching", str(m), "--candidate", str(cand),
                            "--test-dir", str(test_dir), "--check-ids"], capture_output=True, text=True)
        log.append(f"== {m.name} md5 {hashlib.md5(m.read_bytes()).hexdigest()} validator_exit={r.returncode}\n"
                   f"{r.stdout[-4000:]}{r.stderr[-2000:]}")
        print(log[-1], flush=True)
        if r.returncode:
            bad.append(m.name)
    for o in (out_dir, model_dir):
        Path(o).mkdir(parents=True, exist_ok=True)
        (Path(o) / "val.txt").write_text("\n".join(log))
    assert not bad, f"official validator failed: {bad}"
