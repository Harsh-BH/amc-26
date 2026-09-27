"""P50: P45's zoo members' TEST preds aligned to the rebuilt T_{c} rows (q1/qx keys), one member at a time
(download -> align -> delete; disk is tight). Parallel parts: zoo_test.py <country> <k> <n> (members k::n)
-> data/work/p50/zoo_test_{c}_{k}.parquet; `zoo_test.py merge` -> zoo_test_{c}.parquet (p_<member> float32)."""
import subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "probes/P44-meta-stack")
import ev as E  # noqa: E402

B = "s3://sagemaker-114872709824-ap-south-1"
P45J = lambda m: (Path("data/work/p45") / f"job_{m}.txt").read_text().strip()
SRC = {**{m: f"{B}/amc26-p25/results/p25-{m}-0926-1404/pred" for m in ("bag1", "bag2", "bag3", "bag4", "goss", "deep", "extra")},
       "cat": f"{B}/amc26-p22/results/p22-cat-0926-1300/pred",
       **{m: f"{B}/amc26-p45/results/{P45J(m)}/pred" for m in ("dart", "et2", "fsub", "mlp", "mlp2", "noctx", "rank_s1", "rank_sx2")}}
W = Path("data/work/p50")

if sys.argv[1] == "merge":
    for c in E.C:
        parts = sorted(W.glob(f"zoo_test_{c}_*.parquet"))
        d = pd.concat([pd.read_parquet(p) for p in parts], axis=1)
        assert sorted(d.columns) == sorted(f"p_{m}" for m in SRC), d.columns
        d.to_parquet(W / f"zoo_test_{c}.parquet", index=False)
        for p in parts:
            p.unlink()
        print(c, d.shape, flush=True)
    sys.exit()
c, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
T = pd.read_parquet(W / f"T_{c}.parquet", columns=["q1", "qx"])
key = E.keys(T.q1, T.qx)
del T
tmp = W / f"_zoo_tmp_{c}_{k}.parquet"
out = {}
for m in list(SRC)[k::n]:
    subprocess.run(["aws", "s3", "cp", "--only-show-errors", "--profile", "harsh", f"{SRC[m]}/test_{c}.parquet", str(tmp)], check=True)
    t = pq.read_table(tmp, columns=["q1", "qx", "p"])
    v = pd.Series(t.column("p").to_numpy(), index=E.keys(t.column("q1").to_numpy(), t.column("qx").to_numpy()))
    out[f"p_{m}"] = v[~v.index.duplicated()].reindex(key).to_numpy(np.float32)
    miss = float(np.isnan(out[f"p_{m}"]).mean())
    print(c, k, m, t.num_rows, "rows; missing on T", round(miss, 6), flush=True)
    assert miss == 0, (c, m)
    del t, v
    tmp.unlink()
pd.DataFrame(out).to_parquet(W / f"zoo_test_{c}_{k}.parquet", index=False)
print("PART_DONE", c, k, flush=True)
