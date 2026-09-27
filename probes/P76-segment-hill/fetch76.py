"""Fetch a finished P76 SageMaker job and split its results per arm into data/work/p76/{hill,frhill}_<arm>.json
(the local run format), then print the per-arm summary with the null percentiles.
    .venv/bin/python probes/P76-segment-hill/fetch76.py <job> [<job> ...]
"""
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

W = Path("data/work/p76")


def summary(kind, arm, res):
    real = sorted([r for r in res if r["perm"] is None], key=lambda r: r["fit"])
    for r in real:
        h = r["fit"]
        if kind == "iu":
            nl = [x["rep"]["lb"] for x in res if x["perm"] is not None and x["fit"] == h]
            nf = [x["fit_gain"] for x in res if x["perm"] is not None and x["fit"] == h]
            print(f"{arm:12s} fit h{h}->h{1 - h}: fit gain {r['fit_gain']:+.6f} (null p95 {np.percentile(nf, 95) if nf else float('nan'):+.6f}) "
                  f"report {r['rep']['lb']:+.6f} ± {r['rep']['se']:.6f} (null mean {np.mean(nl) if nl else float('nan'):+.6f} p95 "
                  f"{np.percentile(nl, 95) if nl else float('nan'):+.6f}, n {len(nl)}) nfree {r['nfree']}")
        else:
            iu = [x["rep_dF"]["chain|India_US"] for x in res if x["perm"] is not None and x["fit"] == h]
            ui = [x["rep_dF"]["chain|US_India"] for x in res if x["perm"] is not None and x["fit"] == h]
            d = r["rep_dF"]
            print(f"{arm:12s} fit h{h}->h{1 - h}: chain U>I {d['chain|US_India']:+.6f} I>U {d['chain|India_US']:+.6f} | m2 U>I "
                  f"{d['m2|US_India']:+.6f} I>U {d['m2|India_US']:+.6f} | null I>U p95 {np.percentile(iu, 95) if iu else float('nan'):+.6f} "
                  f"U>I p95 {np.percentile(ui, 95) if ui else float('nan'):+.6f} (n {len(iu)}) nfree {r['nfree']} {r['shifts']}")


def main(job):
    subprocess.run([".venv/bin/python", "sm_launch.py", "--profile", "harsh", "--fetch", job], check=True, capture_output=True)
    js = sorted((Path("out") / job).rglob(f"{job}.json"))
    if js:
        r = json.loads(js[0].read_text())
    else:
        with tarfile.open(Path("out") / job / "model.tar.gz") as t:
            r = json.load(t.extractfile([x for x in t.getmembers() if x.name.endswith(".json")][0]))
    kind = "fr" if "-fr" in job else "iu"
    arms = sorted({x["arm"] for x in r["res"]})
    for arm in arms:
        res = [x for x in r["res"] if x["arm"] == arm]
        out = W / (f"frhill_{arm}.json" if kind == "fr" else f"hill_{arm}.json")
        out.write_text(json.dumps(dict(res=res, runtime_s=r["runtime_s"], job=job), default=float))
        summary(kind, arm, res)


if __name__ == "__main__":
    for j in sys.argv[1:]:
        main(j)
