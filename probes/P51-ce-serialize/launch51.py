"""launch51.py <phase[,phase]> <instance> <job-name> [hp.json]: P13's job + bundle on the P51 channel (data/v1).
Hyperparameters come from the JSON config (default hp/loco.json); phases + the job's S3 results path are added here.
The service-quotas pre-check is skipped (throttled when probes launch together; quota is 1 per GPU type)."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P13-aws-rerank")]
import p13
import sm_launch

B = "s3://sagemaker-114872709824-ap-south-1/amc26-p51"
phases, inst, name = sys.argv[1:4]
hp = json.loads(Path(sys.argv[4] if len(sys.argv) > 4 else HERE / "hp/loco.json").read_text())
hp.update(phases=phases.split(","), s3=f"{B}/results/{name}")
data = hp.pop("data_s3", f"{B}/data/v1/")
print(json.dumps(hp))
sm_launch.pack_source = p13._pack
sm_launch.quota_advice = lambda *a: (1, None)
sm_launch.main(["--profile", "harsh", "--instance", inst, "--no-spot", "--prefix", "amc26-p51", "--name", name,
                "--data-s3", data, "--solution", str(ROOT / "probes/P13-aws-rerank/solution_p13.py"),
                "--max-run", "14400", "--hp", json.dumps(hp)])
