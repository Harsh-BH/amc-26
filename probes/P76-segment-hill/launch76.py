"""Launch P76 search jobs on SageMaker CPU (spot). Channel = s3 .../amc26-p76/ch/ (repo-relative frames, see solution_p76).
    launch76.py <name> <instance> <iu|fr> <arm:restarts:nulls>[,<arm:restarts:nulls>...]
    .venv/bin/python probes/P76-segment-hill/launch76.py p76-iu-dek-1735 ml.c6i.4xlarge iu joint_dek:24:20
Fetch: .venv/bin/python sm_launch.py --profile harsh --fetch <name>   (-> out/<name>/model.tar.gz)
"""
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "probes/P76-segment-hill"
sys.path.insert(0, str(ROOT))
import sm_launch  # noqa: E402

B = "s3://sagemaker-114872709824-ap-south-1"


def pack(_solution=None):
    files = {"sm_train.py": ROOT / "sm_train.py", "solution.py": HERE / "solution_p76.py", "hill.py": HERE / "hill.py",
             "dive.py": HERE / "dive.py", "fr_hill.py": HERE / "fr_hill.py",
             "solution_p42.py": ROOT / "probes/P42-fr-joint-tune/solution_p42.py",
             "requirements.txt": HERE / "sm_requirements.txt"}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for arc in sorted(files):
            tar.add(files[arc], arcname=arc)
    return buf.getvalue(), sorted(files)


if __name__ == "__main__":
    name, inst, kind, spec = sys.argv[1:5]
    hp = {"kind": kind, "spec": spec, "name": name}
    sm_launch.pack_source = pack
    sm_launch.main(["--profile", "harsh", "--instance", inst, "--spot", "--prefix", "amc26-p76", "--name", name,
                    "--data-s3", f"{B}/amc26-p76/ch/", "--solution", str(HERE / "solution_p76.py"), "--max-run", "5400",
                    "--max-wait", "6000", "--hp", json.dumps(hp)])
