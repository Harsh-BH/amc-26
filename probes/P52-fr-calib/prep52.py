"""P52 prep: SOURCE in-domain frames, the labelled side of every calibration arm. One row per SX-argmax pair of the
source country's own fold 0 (P6 am_all), scored by the same source-only M2 as the LOCO target frame (P6 S_holdout_<src>),
r = P45's in-domain OOF CE p_r_all (the source-only CE has no source-side scores; see RESULT Risks), hn_rel as P39.

    .venv/bin/python probes/P52-fr-calib/prep52.py US | India     # -> data/work/p52/src_<src>.npz
"""
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P28-addr-anchor")]
W, OUT = ROOT / "data/work", ROOT / "data/work/p52"
CI = {"India": 0, "US": 1}
OFF1, OFF2 = 10_000_000, 100_000_000
SIB3 = np.array([3, 4, 5, 7, 9, 11, 13, 21])
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def hn_rel(h1, hx):                                   # P39 frame.hn_rel: 0 miss, 1 eq, 2 sib3, 3 +1/2, 4 -1/2, 5 other
    ok = (h1 != "") & (hx != "")
    d = np.full(len(h1), np.nan)
    d[ok] = pd.to_numeric(pd.Series(hx[ok]), errors="coerce").to_numpy(float) - pd.to_numeric(pd.Series(h1[ok]), errors="coerce").to_numpy(float)
    return np.where(~ok | np.isnan(d), 0, np.where(d == 0, 1, np.where(np.isin(d, SIB3), 2, np.where(np.isin(d, [1, 2]), 3,
                    np.where(np.isin(d, [-1, -2]), 4, 5))))).astype(np.int8)


def rf_hn(path, ids):
    t = pq.read_table(path, columns=["id", "hn"])
    t = t.filter(pc.is_in(t.column("id"), value_set=pa.array(pd.unique(ids), pa.large_string())))
    return t.to_pandas().set_index("id").hn.reindex(ids).fillna("").to_numpy()


def main(src):
    import features as F
    from ho import half
    t0, ci = time.time(), CI[src]
    S = pq.read_table(W / f"p27/p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all"],
                      filters=[("country", "=", ci), ("am_all", "=", True)]).to_pandas()
    i1, ix = F.ids("train", src)
    e1 = i1.take(pa.array(S.s1.to_numpy() - ci * OFF1)).to_numpy(zero_copy_only=False)
    ex = ix.take(pa.array(S.sx.to_numpy() - ci * OFF2)).to_numpy(zero_copy_only=False)
    del i1, ix
    fo = pd.read_parquet(W / "s1_folds.parquet")
    fo = fo[(fo.fold == 0) & (fo.country == src)].entity_id.to_numpy()
    code = pd.Index(fo).get_indexer(e1)
    assert (code >= 0).all()
    a = pd.read_parquet(W / f"p45/oof/holdout_{src}.parquet", columns=["s1", "sx", "p_r_all"])
    r = pd.DataFrame({"s1": e1, "sx": ex}).merge(a.drop_duplicates(["s1", "sx"]), on=["s1", "sx"], how="left").p_r_all.to_numpy(np.float64)
    del a
    h = hn_rel(rf_hn(W / f"rf_train_s1_{src}.parquet", e1), rf_hn(W / f"rf_train_sx_{src}.parquet", ex))
    gt = pd.read_parquet(W / "gt_pairs.parquet", columns=["s1"]).s1.value_counts()
    g = S.p.to_numpy(np.float64)
    band = (g > 0.02) & (g < 0.98)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"src_{src}.npz", s1=code.astype(np.int32), is2=pd.Series(ex).str.startswith("S2-").to_numpy(),
                        g=g, r=r, y=S.is_true.to_numpy(bool), hn_rel=h, ntrue=gt.reindex(fo).fillna(0).to_numpy(np.int16),
                        hv=half(fo).astype(np.int8))
    print(f"src {src}: {len(S):,} argmax rows, {len(fo):,} S1, y {S.is_true.mean():.4f}; r cover all {np.isfinite(r).mean():.4f} "
          f"band {np.isfinite(r[band]).mean():.4f} hi {np.isfinite(r[g >= 0.98]).mean():.4f}; hn_rel {np.bincount(h, minlength=6).tolist()}; "
          f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)



def france():
    """France hn_rel (P39 classes) for every france.npz pair, from the same hn sources as P42 prep (rf S1, P29 w_fix SX)."""
    sys.path.insert(0, str(ROOT / "probes/P42-fr-joint-tune"))
    import p42
    z = np.load(W / "p42/france.npz")
    i1, ix = p42.fr_ids()
    h1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "hn"]).set_index("id").hn.reindex(i1).fillna("").to_numpy()
    hx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "hn"]).set_index("id").hn
    hx = hx.reindex(ix).fillna("").to_numpy()
    h = hn_rel(h1[z["s1"]], hx[z["sx"]])
    assert ((h == 1) == z["hn_eq"]).all() and ((h == 2) == z["sib"]).all(), "hn_rel disagrees with P42's hn_eq / sib"
    np.save(OUT / "france_hn_rel.npy", h)
    print(f"france hn_rel {np.bincount(h, minlength=6).tolist()} peak {peak():.2f} GB")


if __name__ == "__main__":
    france() if sys.argv[1] == "france" else main(sys.argv[1])
