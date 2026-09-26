"""bge r for France band pairs that a P26 matcher moves into the band (P13 / P19 never scored them), via P19's build
(P13's bge_ckpt job, unchanged) with our pair list.
    .venv/bin/python probes/P26-france-da/bge_topup.py build full_hn full_cf    # -> data/work/p26/bge/sm_data
    then P19's launch command with --prefix amc26-p26bge --data data/work/p26/bge/sm_data (see RESULT.md)
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "probes/P19-bge-idf"))
import p19

W = ROOT / "data/work"
p19.OUT = W / "p26/bge"
p19.DATA = p19.OUT / "sm_data"
p19.log = lambda *a, **k: None                      # P19's ledger row would carry P19's name; P26 logs its own


def band_r(tags):
    have = pd.concat([pd.read_parquet(W / "p13/r/r_bge_pool_test_France.parquet"),
                      pd.read_parquet(W / "p19/r/r_bge_pool_test_France.parquet")]).drop_duplicates(["s1", "sx"])
    b = pd.concat([pd.read_parquet(W / f"p26/prob_test_France_{t}_bge.parquet", columns=["s1", "sx", "p_gbdt"])
                   .query(f"p_gbdt > {p19.LO} and p_gbdt < {p19.HI}") for t in tags]).drop_duplicates(["s1", "sx"])
    b = b.rename(columns={"p_gbdt": "p"}).merge(have, on=["s1", "sx"], how="left")
    miss = b[b.r.isna()]
    ctl = pd.read_parquet(W / "p13/r/r_bge_pool_test_France.parquet").sample(5000, random_state=1).assign(p=0.5)
    print(f"band pairs {len(b):,}, missing r {len(miss):,}", flush=True)
    return pd.concat([miss, ctl[["s1", "sx", "p", "r"]]], ignore_index=True)


if __name__ == "__main__":
    assert sys.argv[1] == "build"
    p19.OUT.mkdir(parents=True, exist_ok=True)
    tags = sys.argv[2:]
    p19.band_r = lambda: band_r(tags)
    p19.build()
