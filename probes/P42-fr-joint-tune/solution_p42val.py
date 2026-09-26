"""P42: the official validator (stdlib only) on SageMaker, where the candidate set fits in memory. Channel 'train':
validate_submission.py, test_source1.tsv, candidate_pairs.tsv, matching_<name>.tsv ... -> val.txt."""
import subprocess
import sys
from pathlib import Path


def main(train_dir, test_dir, model_dir, out_dir, params):
    d = Path(train_dir)
    log = []
    for m in sorted(d.glob("matching_*.tsv")):
        r = subprocess.run([sys.executable, str(d / "validate_submission.py"), "--matching", str(m), "--candidate",
                            str(d / "candidate_pairs.tsv"), "--test-dir", str(d)], capture_output=True, text=True)
        log.append(f"== {m.name} rc={r.returncode}\n{r.stdout}{r.stderr}")
        print(log[-1], flush=True)
    for o in (out_dir, model_dir):
        (Path(o) / "val.txt").write_text("\n".join(log))
