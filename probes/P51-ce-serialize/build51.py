"""P51 data channel: P13's exact pairs (train/val = #5's CE set, holdout = m2_big's band; test bands later), with
serialize() texts instead of P3's "name | addr".
    build51.py build [India,US]      -> data/work/p51/sm_data/{train,val,holdout}_<c>.parquet (same rows/order as P13)
    build51.py test <c>              -> data/work/p51/sm_data/test_<c>.parquet (m2_big test band, as P13)
    build51.py pairs <split> <c> <src> <out> [holdout_am]   -> any (s1, sx) file with serialized texts (P45 r_all rows)
    build51.py show [n=20]           -> n examples per country (train holdout + test France)
"""
import gc, resource, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "probes/P4b-rerank")]
import rerank as rr
from serialize import serialize, gib

WORK = ROOT / "data/work"
DATA = WORK / "p51/sm_data"
LO, HI = 0.02, 0.98
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
M = {s: gib.learn(s) for s in ("train", "test")}


def _raw(split: str, ids: pa.Array, srcs) -> pd.DataFrame:
    """Raw (entity_id, business_name, business_address, country) of `ids` from <split>_source<i>.parquet."""
    parts = []
    for i in srcs:
        f = pq.ParquetFile(WORK / f"{split}_source{i}.parquet")
        for rg in range(f.num_row_groups):
            t = f.read_row_group(rg)
            parts.append(t.filter(pc.is_in(t["entity_id"], value_set=ids)))
    return pa.concat_tables(parts).to_pandas().set_index("entity_id")


def _ser(split: str, c: str, txt: pa.Table, srcs) -> np.ndarray:
    """P3 text rows (entity_id, name) -> serialize() strings, in chunks."""
    raw = _raw(split, txt["entity_id"], srcs)
    d = txt.select(["entity_id", "name"]).to_pandas()
    d = d.join(raw[["business_name", "business_address", "country"]], on="entity_id")
    assert d.business_name.notna().all(), f"{split} {c}: every P3 row has its raw record"
    return np.concatenate([serialize(d.iloc[k:k + 100_000].reset_index(drop=True), M[split])
                           for k in range(0, len(d), 100_000)])


def texts(split: str, c: str, q1: np.ndarray, qx: np.ndarray):
    """rr.texts with serialize(): (S1 text of row q1, SX text of row qx)."""
    TXT = rr.TXT[split]
    u1, i1 = np.unique(q1, return_inverse=True)
    a = _ser(split, c, pq.read_table(TXT / f"{c}_s1.parquet", columns=["entity_id", "name"]).take(u1), (1,))[i1]
    ux, ix = np.unique(qx, return_inverse=True)
    parts, off = [], 0
    for p in sorted(TXT.glob(f"{c}_sx_*.parquet")):
        n = pq.ParquetFile(p).metadata.num_rows
        lo, hi = np.searchsorted(ux, [off, off + n])
        if hi > lo:
            parts.append(pq.read_table(p, columns=["entity_id", "name"]).take(ux[lo:hi] - off))
        off += n
    tx = _ser(split, c, pa.concat_tables(parts), (2, 3))
    del parts
    gc.collect()
    return a, tx[ix]


def _save(df, name):
    df.to_parquet(DATA / name, index=False, compression="zstd")
    print(f"{name}: {len(df):,} rows, {(DATA / name).stat().st_size / 1e6:.1f} MB, peak {peak():.2f} GB, "
          f"{time.time() - T0:.0f}s", flush=True)


def build(countries="India,US"):
    """p13.build's selection verbatim (seed 0), texts from serialize()."""
    DATA.mkdir(parents=True, exist_ok=True)
    for c in countries.split(","):
        b = pd.read_parquet(WORK / "p4b" / f"trainband_{c}.parquet")
        v = b[(b.fold == 3) & b.inband]
        b = b[b.fold >= 4]
        ib, ob = b[b.inband], b[~b.inband]
        b = pd.concat([ib, ob.sample(min(len(ob), int(0.5 * len(ib))), random_state=0)])
        v = v.sample(min(len(v), 20_000), random_state=0)
        for nm, d in (("val", v), ("train", b)):
            ta, tb = texts("train", c, d.q1.to_numpy(), d.qx.to_numpy())
            _save(pd.DataFrame({"ta": ta, "tb": tb, "y": d.is_true.to_numpy(np.int8), "p": d.p.to_numpy(np.float32)}),
                  f"{nm}_{c}.parquet")
            del ta, tb
        del b, v, ib, ob
        gc.collect()
        h = pd.read_parquet(WORK / "p4/prob_holdout_big.parquet", columns=["s1", "sx", "p", "country", "is_true"])
        g = h[(h.p > LO) & (h.p < HI) & (h.country == c)]
        del h
        q1, qx = rr.rows("train", c, g.s1, g.sx)
        ta, tb = texts("train", c, q1, qx)
        _save(pd.DataFrame({"s1": g.s1.to_numpy(), "sx": g.sx.to_numpy(), "ta": ta, "tb": tb,
                            "y": g.is_true.to_numpy(np.int8), "p": g.p.to_numpy(np.float32)}), f"holdout_{c}.parquet")
        del g, ta, tb
        gc.collect()


def test(c, src="", chunk=200_000):
    """Test band pairs, written in pair chunks so the text arrays never all sit in RAM. src: a (s1, sx, p_gbdt)
    parquet whose 0.02 < p_gbdt < 0.98 rows are scored (France: P31's file, the pairs build38's chain blends);
    default m2_big's band (as p13.build)."""
    DATA.mkdir(parents=True, exist_ok=True)
    if src:
        g = pd.read_parquet(src, columns=["s1", "sx", "p_gbdt"]).rename(columns={"p_gbdt": "p"})
        g = g[(g.p > LO) & (g.p < HI)].reset_index(drop=True)
    else:
        g = pq.read_table(WORK / f"p4/prob_test_{c}_big.parquet", columns=["s1", "sx", "p"],
                          filters=[("p", ">", LO), ("p", "<", HI)]).to_pandas()
    q1, qx = rr.rows("test", c, g.s1, g.sx)
    w = None
    for k in range(0, len(g), chunk):
        ta, tb = texts("test", c, q1[k:k + chunk], qx[k:k + chunk])
        t = pa.Table.from_pandas(pd.DataFrame({"s1": g.s1.to_numpy()[k:k + chunk], "sx": g.sx.to_numpy()[k:k + chunk],
                                               "ta": ta, "tb": tb, "p": g.p.to_numpy(np.float32)[k:k + chunk]}),
                                 preserve_index=False)
        w = w or pq.ParquetWriter(DATA / f"test_{c}.parquet", t.schema, compression="zstd")
        w.write_table(t)
        del ta, tb, t
        gc.collect()
        print(f"test_{c}: {min(k + chunk, len(g)):,}/{len(g):,} peak {peak():.2f} GB {time.time() - T0:.0f}s", flush=True)
    w.close()


def pairs(split, c, src, out, holdout_am="", chunk=250_000):
    """Any (s1, sx) pair file -> serialized texts parquet `out` (s1, sx, ta, tb, p=0), chunked. holdout_am: keep only
    rows with holdout & am (P45's r_all rows on the holdout: every fold-0 #8c argmax pair)."""
    cols = ["s1", "sx"] + (["holdout", "am"] if holdout_am else [])
    g = pd.read_parquet(src, columns=cols)
    if holdout_am:
        g = g[g.holdout & g.am].reset_index(drop=True)
    q1, qx = rr.rows(split, c, g.s1, g.sx)
    w = None
    for k in range(0, len(g), chunk):
        ta, tb = texts(split, c, q1[k:k + chunk], qx[k:k + chunk])
        t = pa.Table.from_pandas(pd.DataFrame({"s1": g.s1.to_numpy()[k:k + chunk], "sx": g.sx.to_numpy()[k:k + chunk],
                                               "ta": ta, "tb": tb, "p": np.zeros(len(ta), np.float32)}), preserve_index=False)
        w = w or pq.ParquetWriter(out, t.schema, compression="zstd")
        w.write_table(t)
        del ta, tb, t
        gc.collect()
        print(f"{out}: {min(k + chunk, len(g)):,}/{len(g):,} peak {peak():.2f} GB {time.time() - T0:.0f}s", flush=True)
    w.close()


def show(n=20):
    """n random pairs per country (holdout India/US, test France) -> data/work/p51/examples.txt (data: never in git)."""
    n, out = int(n), []
    for f in ("holdout_India", "holdout_US", "test_France"):
        d = pd.read_parquet(DATA / f"{f}.parquet", columns=["ta", "tb"] + (["y"] if "holdout" in f else []))
        for _, r in d.sample(n, random_state=0).iterrows():
            out.append(f"[{f}{' y=' + str(r.y) if 'y' in r else ''}]\n  S1: {r.ta}\n  SX: {r.tb}")
    (WORK / "p51/examples.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out[:3]))


if __name__ == "__main__":
    T0 = time.time()
    globals()[sys.argv[1]](*sys.argv[2:])
