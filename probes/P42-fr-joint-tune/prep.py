"""P42 prep: compact numpy frames for the joint France-lever grid (small enough to ship to SageMaker).

  loco SRC TGT  P39's France chain analogue with the raw inputs kept, so the veto tau can move: one row per SX-argmax
                pair of TGT's fold 0 (P6's am_all, fixed), g = P6's source-only M2 p, r = P30's source-only bge score,
                y, hn_rel (P39's, row-aligned), + per-S1 ntrue and the report half.  -> data/work/p42/loco_SRC_TGT.npz
  france        every pair of the France chain file (1.33M, codes): p0 = fr31 without veto (tier / bxneg / cap applied),
                p05 / p10 = the veto 0.05 / 0.1 files, g (c_hn's p), r (P30 pool, then the CE cache: run.py's order),
                sib (P15 SIB3) and hn_eq (P37's definitions, w_fix SX house numbers).  -> data/work/p42/france.npz

    .venv/bin/python probes/P42-fr-joint-tune/prep.py loco US India | loco India US | france
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

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P15-research"),
                str(ROOT / "probes/P28-addr-anchor")]
W, OUT = ROOT / "data/work", ROOT / "data/work/p42"
CI = {"India": 0, "US": 1}
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def loco(src, tgt):
    from ho import half
    t0 = time.time()
    S = pd.read_parquet(W / f"p27/p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all"])
    S = S[S.country.to_numpy() == CI[tgt]].drop(columns="country").reset_index(drop=True)
    S = S.merge(pd.read_parquet(W / f"p30/r/r_loco{src}_tgt_{tgt}.parquet"), on=["s1", "sx"], how="left")
    S = S[S.am_all.to_numpy()].reset_index(drop=True)                 # P39 frame.loco's row order
    f = pd.read_parquet(W / f"p39/loco_{src}_{tgt}.parquet", columns=["s1", "is2", "g", "y", "hn_rel", "p"])
    assert len(f) == len(S) and (f.g.to_numpy() == S.p.to_numpy(np.float32)).all() and (f.y.to_numpy() == S.is_true.to_numpy()).all()
    s = pd.read_parquet(W / f"p39/s1_{src}_{tgt}.parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"loco_{src}_{tgt}.npz", s1=f.s1.to_numpy(np.int32), is2=f.is2.to_numpy(), g=S.p.to_numpy(np.float64),
                        r=S.r.to_numpy(np.float64), y=f.y.to_numpy(), hn_rel=f.hn_rel.to_numpy(np.int8), p39=f.p.to_numpy(),
                        ntrue=s.ntrue.to_numpy(np.int16), hv=half(s.id).astype(np.int8))
    print(f"loco {src}>{tgt}: {len(f):,} rows, {len(s):,} S1; {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)


def france():
    import features as F
    from sib_rule import sib_mask
    t0 = time.time()
    d = pd.read_parquet(W / "p31/fr31_t9868.parquet")
    for tau, f in (("p05", "fr31v0.05_t9868"), ("p10", "fr31v0.1_t9868")):
        e = pd.read_parquet(W / f"p31/{f}.parquet")
        assert (e.s1.to_numpy() == d.s1.to_numpy()).all() and (e.sx.to_numpy() == d.sx.to_numpy()).all()
        d[tau] = e.p.to_numpy()
        del e
    e = pd.read_parquet(W / "p31/prob_test_France_chn_fix_bge.parquet", columns=["s1", "sx", "p_gbdt"])
    assert (e.s1.to_numpy() == d.s1.to_numpy()).all() and (e.sx.to_numpy() == d.sx.to_numpy()).all()
    d["g"] = e.p_gbdt.to_numpy()
    del e
    id1, idx = F.ids("test", "France")
    q1 = pc.index_in(pa.array(d.s1.to_numpy(), pa.large_string()), value_set=id1.cast(pa.large_string())).to_numpy(zero_copy_only=False)
    qx = pc.index_in(pa.array(d.sx.to_numpy(), pa.large_string()), value_set=idx.cast(pa.large_string())).to_numpy(zero_copy_only=False)
    assert not (pd.isna(q1).any() or pd.isna(qx).any())
    k = (q1.astype(np.int64) << 32) | qx.astype(np.int64)
    rp = pd.read_parquet(W / "p30/r/r_pool_France.parquet")
    a = pc.index_in(pa.array(rp.s1.to_numpy(), pa.large_string()), value_set=id1.cast(pa.large_string())).to_numpy(zero_copy_only=False)
    b = pc.index_in(pa.array(rp.sx.to_numpy(), pa.large_string()), value_set=idx.cast(pa.large_string())).to_numpy(zero_copy_only=False)
    ok = ~(pd.isna(a) | pd.isna(b))
    rk = pd.concat([pd.DataFrame({"k": (a[ok].astype(np.int64) << 32) | b[ok].astype(np.int64), "r": rp.r.to_numpy()[ok].astype(np.float32)}),
                    pd.read_parquet(W / "p9/r_test_France_3bd0b930.parquet")]).drop_duplicates("k")
    del rp, a, b
    r = pd.Series(rk.r.to_numpy(np.float32), index=rk.k.to_numpy()).reindex(k).to_numpy(np.float32)
    del rk
    s1ids = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id"]).id.to_numpy()
    c1 = pd.Index(s1ids).get_indexer(d.s1)
    assert (c1 >= 0).all()
    h1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "hn"]).set_index("id").hn
    hx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "hn"]).set_index("id").hn
    h1, hx = h1.reindex(d.s1).fillna("").to_numpy(), hx.reindex(d.sx).fillna("").to_numpy()
    sib, hn_eq = sib_mask(h1, hx), (h1 != "") & (h1 == hx)
    del h1, hx
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "france.npz", s1=c1.astype(np.int32), sx=qx.astype(np.int32), is2=d.sx.str.startswith("S2-").to_numpy(),
                        p0=d.p.to_numpy(np.float64), p05=d.p05.to_numpy(np.float64), p10=d.p10.to_numpy(np.float64),
                        g=d.g.to_numpy(np.float64), r=r.astype(np.float64), sib=sib, hn_eq=hn_eq, n_s1=len(s1ids))
    vv = (d.g.to_numpy() >= 0.98) & (np.nan_to_num(r, nan=1.0) < 0.1)
    print(f"france: {len(d):,} pairs, {len(s1ids):,} S1; r missing on g>=0.98: {int(np.isnan(r[d.g.to_numpy() >= 0.98]).sum())}; "
          f"veto0.1 cond {int(vv.sum()):,}, p10 zeroed {int(((d.p10 == 0) & (d.p > 0)).sum()):,}, cond&kept {int((vv & (d.p10 > 0)).sum()):,}; "
          f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)


if __name__ == "__main__":
    {"loco": loco, "france": france}[sys.argv[1]](*sys.argv[2:])
