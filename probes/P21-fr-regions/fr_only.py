"""P21 check B driver: run.py's model stage for France alone under ER_WORK=data/work/p21/w (fr_world.py), then diff
France's rows of both output files against a staged submission.

    ER_WORK=data/work/p21/w .venv/bin/python probes/P21-fr-regions/fr_only.py sub8c output/sub8c
"""
import json
import os
import sys
import time
from pathlib import Path

assert os.environ.get("ER_WORK", "").endswith("p21/w"), "run under the isolated ER_WORK"
sys.path.insert(0, "code/business_entity_resolution/src")
import numpy as np
import run
import model as M



def main():
    recipe, ref = sys.argv[1], Path(sys.argv[2])
    if not ref.is_absolute() and not ref.exists():
        ref = Path("/home/harsh1/github-repos/amc-26") / ref
    run.use(recipe)
    t0 = time.time()
    d = run.model_parts("test", only=["France"])[0]
    t = run.thresholds([d])["France"]
    K = (d["q1"].astype(np.int64) << 32) | d["qx"].astype(np.int64)
    km = K[M.argmax_sx(K & 0xFFFFFFFF, d["p"]) & (d["p"] >= t)]
    out = Path(f"data/work/p21/out/{recipe}")
    out.mkdir(parents=True, exist_ok=True)
    ids = d["id1"]
    pos = np.arange(len(ids))
    res = {"recipe": recipe, "t": t, "secs": round(time.time() - t0)}
    for name, keys in (("matching_results.tsv", km), ("candidate_pairs.tsv", K)):
        with open(out / name, "w", encoding="utf-8") as f:
            run.write_lists(f, ids, pos[keys >> 32], d["idx"], keys & 0xFFFFFFFF)
        fr = set(ids.to_pylist())
        new = set((out / name).read_text(encoding="utf-8").splitlines())
        old = {l for l in (ref / name).read_text(encoding="utf-8").splitlines() if l.split("\t", 1)[0] in fr}
        diff = sorted({l.split("\t", 1)[0] for l in new ^ old})
        res[name] = {"france_rows_new": len(new), "france_rows_ref": len(old), "rows_differ": len(diff), "s1": diff[:20]}
    print(json.dumps(res, indent=1))
    (out / "diff.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":     # rules.feats' forkserver workers re-import this file
    main()
