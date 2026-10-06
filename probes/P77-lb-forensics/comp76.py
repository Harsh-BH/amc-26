"""P77: component flags for the sub66_final -> sub76 change set (P67q, P70a, t975ce, P71a adds; N54 removals).
    .venv/bin/python probes/P77-lb-forensics/comp76.py  -> work/comp76.parquet (key, comp flags)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from collect import keys  # noqa: E402
W = ROOT / "data/work"


def enc(s1, sx):
    s1, sx = pd.Series(s1, dtype=str), pd.Series(sx, dtype=str)
    return ((s1.str.slice(3).astype(np.int64).to_numpy() << 31) | ((sx.str.slice(0, 2) == "S3").to_numpy().astype(np.int64) << 30)
            | sx.str.slice(3).astype(np.int64).to_numpy())


def main():
    u = pd.read_parquet(HERE / "work/union.parquet")
    names = [n for n, _ in json.loads((HERE / "work/files.json").read_text())]
    m = u["mask"].to_numpy().astype(np.int64)
    base = u.key.to_numpy()[(m >> names.index("sub66_final")) & 1 == 1]
    comp = {}
    p67 = pd.read_parquet(W / "p67/changes.parquet")
    comp["P67q"] = enc(p67.s1[p67.qgate_keep.astype(bool)], p67.sx[p67.qgate_keep.astype(bool)])
    comp["P67all"] = enc(p67.s1, p67.sx)
    p70 = pd.read_parquet(W / "p70/changes.parquet")
    comp["P70a"] = enc(p70.s1, p70.sx)
    for nm, d in [("t975ce", "sub66_t975ce"), ("P71a", "sub71a"), ("P41", "sub41_on66")]:
        k = keys(ROOT / "output" / d / "matching_results.tsv")
        comp[nm] = np.setdiff1d(k, base)
        print(nm, len(comp[nm]), "adds;", len(np.setdiff1d(base, k)), "removes", flush=True)
        del k
    n54 = pd.read_parquet(W / "p74/n54/removed.parquet")
    comp["N54"] = enc(n54.s1, n54.sx)
    allk = np.unique(np.concatenate(list(comp.values())))
    out = pd.DataFrame({"key": allk})
    for nm, k in comp.items():
        out[nm] = np.isin(allk, k)
    out.to_parquet(HERE / "work/comp76.parquet", index=False)
    print({k: int(out[k].sum()) for k in comp})


if __name__ == "__main__":
    main()
