"""P49 test 4 (India): does an Indic-script SX name's script predict its owner's state?
Learn P(script | S1 state) on train true pairs of non-holdout S1 (folds 1-9, 30% sample), then among ambiguous twin SX
with an Indic name: predict the twin whose state gives the script the highest probability (unique best only).
    probes/P49-twin-prior/py.sh probes/P49-twin-prior/test4.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame import W  # noqa: E402

BLOCKS = [(0x0900, "Deva"), (0x0980, "Beng"), (0x0A00, "Guru"), (0x0A80, "Gujr"), (0x0B00, "Orya"), (0x0B80, "Taml"),
          (0x0C00, "Telu"), (0x0C80, "Knda"), (0x0D00, "Mlym"), (0x0D80, None)]


def script(s: str):
    for ch in s:
        o = ord(ch)
        if 0x0900 <= o < 0x0D80:
            for (a, nm), (b, _) in zip(BLOCKS, BLOCKS[1:]):
                if a <= o < b:
                    return nm
    return None


assert script("Café") is None and script("राम") == "Deva" and script("தமிழ்") == "Taml"

if __name__ == "__main__":

    s1 = pd.read_parquet(W / "s1_folds.parquet")
    s1 = s1[(s1.country == "India") & (s1.fold != 0)].sample(frac=0.3, random_state=0)
    gt = pq.read_table(W / "gt_pairs.parquet", filters=[("s1", "in", list(s1.entity_id))]).to_pandas()
    ids = set(gt.sx)
    parts = []
    for i in (2, 3):
        for b in pq.ParquetFile(W / f"train_source{i}.parquet").iter_batches(columns=["entity_id", "business_name", "country"],
                                                                            batch_size=1_000_000):
            d = b.to_pandas()
            d = d[(d.country == "India") & d.entity_id.isin(ids)]
            d["scr"] = [script(x) for x in d.business_name]
            parts.append(d[["entity_id", "scr"]])
    sx = pd.concat(parts).set_index("entity_id").scr
    gt["scr"] = sx.reindex(gt.sx).to_numpy()
    rf = pd.read_parquet(W / "rf_train_s1_India.parquet", columns=["id", "state"]).set_index("id").state
    gt["state"] = rf.reindex(gt.s1).to_numpy()
    g = gt[gt.scr.notna() & gt.state.notna() & (gt.state != "")]
    tab = pd.crosstab(g.state, g.scr)
    P = (tab + 1).div((tab + 1).sum(axis=1), axis=0)             # P(script | state), add-one
    P.to_parquet(W / "p49/script_state.parquet")
    print(f"train (30% of non-holdout India S1): {len(gt):,} true pairs, Indic {gt.scr.notna().mean():.3f}; "
          f"states {len(tab)}; P(max script | state) median {P.max(axis=1).median():.3f}")
    top = P.idxmax(axis=1)
    print("state -> top script share (10 largest states):",
          {s: (top[s], round(P.loc[s].max(), 2)) for s in tab.sum(axis=1).sort_values().index[-10:]})
    mi = (tab / tab.values.sum())
    pj = mi.values
    px, py = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True)
    print(f"mutual information script;state = {np.nansum(pj * np.log(pj / (px * py + 1e-12) + 1e-12)):.3f} nats "
          f"(script entropy {-np.nansum(py * np.log(py)):.3f})")

    tw = pd.read_parquet(W / "p49/tw_India.parquet", columns=["sx", "s1", "holdout", "y_full", "nset", "sx_name"])
    tw["own"] = tw.groupby("sx").y_full.transform("any")
    tw["anyho"] = tw.groupby("sx").holdout.transform("any")
    q = tw[tw.own & tw.anyho].copy()
    q["scr"] = [script(x) for x in q.sx_name]
    q = q[q.scr.notna()].copy()
    q["state"] = rf.reindex(q.s1).to_numpy()
    q["lp"] = [np.log(P.loc[s, k]) if (s in P.index and k in P.columns) else np.nan for s, k in zip(q.state, q.scr)]
    q = q[q.groupby("sx").lp.transform(lambda x: x.notna().all())]
    mx = q.groupby("sx").lp.transform("max")
    q["best"] = q.lp == mx
    nb = q.groupby("sx").best.transform("sum")
    one = q[q.best & (nb == 1)]
    r = one.y_full.mean()
    print(f"ambiguous India SX with an Indic name (owner in set): {q.sx.nunique():,}; twins' states differ enough for a unique "
          f"best: {len(one):,}; best-state twin is the owner {r:.3f} ± {np.sqrt(r * (1 - r) / max(len(one), 1)):.3f} "
          f"(uniform {(1 / one.nset).mean():.3f})")
