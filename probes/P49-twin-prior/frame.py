"""P49: the twin frame on the dense #8c holdout (P23 dump) -- address-less SX whose candidate list holds >= 2 S1 with the
same normalised FULL name (P40 floor.key) as its #8c argmax S1.

    probes/P49-twin-prior/py.sh probes/P49-twin-prior/frame.py        # -> data/work/p49/tw_{c}.parquet, s1_{c}.parquet

tw_{c}: one row per (ambiguous SX, candidate S1 in its twin set): sx, s1, holdout (scored fold-0 S1), p (#8c), am, y_full
  (true pair, FULL train truth), src (S2/S3), sx_name, s1_name (raw), kfull (train S1 sharing the key), nset (twins that
  are candidates), exact (key(sx) == key), kd_src / kd_oth (label-free: other accepted #8c argmax SX of the S1 in the SX's
  source / the other source, addressed SX only), kt_src / kt_oth (truth: addressed true members, same / other source),
  kta_src (truth: address-less true members of the source, excluding this SX).
s1_{c}: per twin-set S1: s1, state (rf), holdout, n_rows (frame rows).
Aggregates only in logs; raw strings stay in data/work (gitignored).
"""
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
OUT = W / "p49"
T = 0.71
NA = {"", "NULL", "<NULL>", "N/A", "NA", "NONE", "-"}          # P35 / P36 address-less placeholders


def key(s: str) -> str:
    """P40 floor.key: NFKD, accents dropped, lower, & -> and, unicode word tokens."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("&", " and ")
    return " ".join(re.findall(r"\w+", s))


def noaddr(a: pd.Series) -> np.ndarray:
    return a.fillna("").str.strip().str.upper().isin(NA).to_numpy()


def sx_text(ids: set) -> pd.DataFrame:
    """entity_id, business_name, business_address of the given SX ids (streamed, both sources)."""
    va = pa.array(list(ids), pa.string())
    parts = []
    for i in (2, 3):
        for b in pq.ParquetFile(W / f"train_source{i}.parquet").iter_batches(
                columns=["entity_id", "business_name", "business_address"], batch_size=1_000_000):
            m = pc.is_in(b.column(0), value_set=va)
            parts.append(pa.Table.from_batches([b]).filter(m).to_pandas())
    return pd.concat(parts, ignore_index=True)


def build(c: str):
    t0 = time.time()
    s1 = pq.read_table(W / "train_source1.parquet", columns=["entity_id", "business_name", "country"],
                       filters=[("country", "=", c)]).to_pandas()
    s1["k"] = [key(x) for x in s1.business_name]
    s1["kfull"] = s1.groupby("k").entity_id.transform("size")
    s1 = s1.set_index("entity_id")
    d = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "p", "am"])
    d["k1"] = s1.k.reindex(d.s1).to_numpy()
    # address status of every frame SX
    tx = sx_text(set(d.sx.unique())).set_index("entity_id")
    d["na"] = noaddr(tx.business_address.reindex(d.sx))
    # twin set: candidates of an address-less SX sharing its argmax S1's key
    amk = d[d.am].set_index("sx").k1
    d["amk"] = amk.reindex(d.sx).to_numpy()
    tw = d[d.na & (d.k1 == d.amk)].copy()
    tw["nset"] = tw.groupby("sx").s1.transform("size")
    tw = tw[tw.nset >= 2].reset_index(drop=True)
    tw["kfull"] = s1.kfull.reindex(tw.s1).to_numpy()
    tw["src"] = tw.sx.str[:2]
    tw["sx_name"] = tx.business_name.reindex(tw.sx).to_numpy()
    tw["s1_name"] = s1.business_name.reindex(tw.s1).to_numpy()
    tw["exact"] = np.array([key(x) for x in tw.sx_name]) == tw.k1.to_numpy()
    # label-free counts: other accepted #8c argmax SX of the S1 (addressed only), per source
    acc = d[d.am & (d.p >= T) & ~d.na]
    cnt = acc.assign(src=acc.sx.str[:2]).groupby(["s1", "src"]).size()
    oth = {"S2": "S3", "S3": "S2"}
    tw["kd_src"] = cnt.reindex(pd.MultiIndex.from_arrays([tw.s1, tw.src])).fillna(0).to_numpy(int)
    tw["kd_oth"] = cnt.reindex(pd.MultiIndex.from_arrays([tw.s1, tw.src.map(oth)])).fillna(0).to_numpy(int)
    # truth: full train truth of the twin-set S1
    gt = pq.read_table(W / "gt_pairs.parquet", filters=[("s1", "in", list(set(tw.s1)))]).to_pandas()
    gtx = sx_text(set(gt.sx)).set_index("entity_id")
    gt["na"] = noaddr(gtx.business_address.reindex(gt.sx))
    gt["src"] = gt.sx.str[:2]
    ya = set(zip(gt.s1, gt.sx))
    tw["y_full"] = [(a, b) in ya for a, b in zip(tw.s1, tw.sx)]
    ta = gt[~gt.na].groupby(["s1", "src"]).size()
    tn = gt[gt.na].groupby(["s1", "src"]).size()
    ix = pd.MultiIndex.from_arrays([tw.s1, tw.src])
    tw["kt_src"] = ta.reindex(ix).fillna(0).to_numpy(int)
    tw["kt_oth"] = ta.reindex(pd.MultiIndex.from_arrays([tw.s1, tw.src.map(oth)])).fillna(0).to_numpy(int)
    tw["kta_src"] = tn.reindex(ix).fillna(0).to_numpy(int) - tw.y_full.astype(int)
    # frame coverage of each twin S1's addressed truth (is kd a fair label-free proxy for competitor S1?)
    fr = set(zip(d.s1, d.sx))
    gt["in_frame"] = [(a, b) in fr for a, b in zip(gt.s1, gt.sx)]
    cov = gt[~gt.na].groupby("s1").in_frame.mean()
    tw["cov"] = cov.reindex(tw.s1).to_numpy()
    tw = tw.drop(columns=["k1", "amk", "na"])
    tw.to_parquet(OUT / f"tw_{c}.parquet", index=False)
    rf = pd.read_parquet(W / f"rf_train_s1_{c}.parquet", columns=["id", "state"]).set_index("id")
    S = pd.DataFrame({"s1": sorted(set(tw.s1))})
    S["state"] = rf.state.reindex(S.s1).to_numpy()
    S["holdout"] = S.s1.isin(set(d.s1[d.holdout]))
    S.to_parquet(OUT / f"s1_{c}.parquet", index=False)
    nsx = tw.sx.nunique()
    own = tw.groupby("sx").y_full.any()
    print(f"{c}: {len(tw):,} rows, {nsx:,} ambiguous SX ({own.mean():.3f} owned inside the set), "
          f"{tw.groupby('sx').holdout.any().mean():.3f} with a holdout twin; nset {tw.groupby('sx').size().value_counts().sort_index().to_dict()}; "
          f"addressed-truth frame coverage holdout S1 {tw[tw.holdout]['cov'].mean():.3f} / competitor {tw[~tw.holdout]['cov'].mean():.3f}; "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    assert key("Café & Co., LTD") == "cafe and co ltd"
    assert noaddr(pd.Series([" null ", "<NULL>", "12 Main St", ""])).tolist() == [True, True, False, True]
    OUT.mkdir(exist_ok=True)
    for c in sys.argv[1:] or ("India", "US"):
        build(c)
