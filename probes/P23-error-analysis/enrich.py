"""P23: one error table per country out of dump.py's files, texts and rule features joined, for the error analysts.

    .venv/bin/python probes/P23-error-analysis/enrich.py      # after dump.py; ~2 min, < 2 GB

err_<c>.parquet, one row per error pair of a holdout S1 (+ a TP sample for contrast), kind:
  missed      true pair, not a candidate            (in_p3: P3 blocking had it, so the pruner / cut dropped it)
  fn_stolen   true pair, candidate, its SX went to another S1 (win_s1 / win_p / win_y / win_holdout)
  fn_lowp     true pair, candidate, argmax is right but p < t
  fp          predicted, not true (own_s1 = the SX's true S1 if any, own_p = that pair's p if it was a candidate)
  tp          a 4,000-row sample of correct predictions
Columns per side (s1_ / sx_ / win_): name, addr, core, web, state, city, hn, places; per S1: n_true, n_pred, n_cand, f05.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
from metric import f_beta
from worlds import WORK

D = WORK / "p23"
T = 0.71
RF = ["core", "web", "state", "city", "hn", "places"]


def texts(ids: set) -> pd.DataFrame:
    """entity_id -> name, addr (raw) + rule features, read with an id filter, one source at a time."""
    v = pa.array(sorted(ids), pa.string())
    out = []
    for i in (1, 2, 3):
        t = pq.read_table(WORK / f"train_source{i}.parquet", columns=["entity_id", "business_name", "business_address"])
        out.append(t.filter(pc.is_in(t["entity_id"], value_set=v)).to_pandas())
        del t
    tx = pd.concat(out).rename(columns={"entity_id": "id", "business_name": "name", "business_address": "addr"})
    rf = []
    for side in ("s1", "sx"):
        for c in ("India", "US"):
            t = pq.read_table(WORK / f"rf_train_{side}_{c}.parquet", columns=["id", *RF])
            rf.append(t.filter(pc.is_in(t["id"], value_set=v)).to_pandas())
            del t
    return tx.merge(pd.concat(rf).drop_duplicates("id"), on="id", how="left").set_index("id")


def side(df: pd.DataFrame, col: str, tx: pd.DataFrame, pre: str) -> pd.DataFrame:
    return df.join(tx.add_prefix(pre), on=col)


gt_all = pd.read_parquet(WORK / "gt_pairs.parquet")
for c in ("India", "US"):
    df = pd.read_parquet(D / f"pairs_{c}.parquet")
    mi = pd.read_parquet(D / f"missed_{c}.parquet")
    h = df[df.holdout]
    s1ids = pd.read_parquet(WORK / "worlds/holdout/s1.parquet", columns=["entity_id", "country"])
    s1ids = s1ids.entity_id[s1ids.country == c]
    tr = pd.read_parquet(WORK / "worlds/holdout/truth.parquet")
    tr = tr[tr.s1.isin(set(s1ids))]
    # per-S1 context
    ntrue = tr.groupby("s1").size()
    npred = h[h.pred].groupby("s1").size()
    ncand = h.groupby("s1").size()
    tset = tr.groupby("s1").sx.apply(set)
    pset = h[h.pred].groupby("s1").sx.apply(set)
    f05 = pd.Series({s: f_beta(pset.get(s, set()), tset.get(s, set())) for s in s1ids})
    ctx = pd.DataFrame({"n_true": ntrue, "n_pred": npred, "n_cand": ncand, "f05": f05}).reindex(s1ids).fillna(
        {"n_true": 0, "n_pred": 0, "n_cand": 0})
    # winners: the argmax row of every SX
    win = df[df.am][["sx", "s1", "p", "y", "holdout"]].rename(
        columns={"s1": "win_s1", "p": "win_p", "y": "win_y", "holdout": "win_holdout"})
    fn = h[h.y & ~h.pred].copy()
    fn["kind"] = np.where(fn.am, "fn_lowp", "fn_stolen")
    fn = fn.merge(win, on="sx", how="left")
    lowp = fn.kind == "fn_lowp"
    fn[["win_y", "win_holdout"]] = fn[["win_y", "win_holdout"]].astype(object)
    fn.loc[lowp, ["win_s1", "win_p", "win_y", "win_holdout"]] = None
    fp = h[h.pred & ~h.y].copy()
    fp["kind"] = "fp"
    own = gt_all[gt_all.sx.isin(set(fp.sx))].rename(columns={"s1": "own_s1"})
    fp = fp.merge(own, on="sx", how="left")
    fp = fp.merge(df[["s1", "sx", "p"]].rename(columns={"s1": "own_s1", "p": "own_p"}), on=["own_s1", "sx"], how="left")
    fp["own_holdout"] = fp.own_s1.isin(set(s1ids))
    tp = h[h.y & h.pred].sample(4000, random_state=0).assign(kind="tp")
    mi = mi.assign(kind="missed")
    p3 = pq.read_table(WORK / f"p3/cands_holdout_full_{c}.parquet",
                       columns=["s1", "sx", "block_score", "block_rank_s1", "block_rank_sx", "via"]).to_pandas()
    mi = mi.merge(p3, on=["s1", "sx"], how="left")
    mi["in_p3"] = mi.block_score.notna()
    del p3
    err = pd.concat([mi, fn, fp, tp], ignore_index=True)
    err = err.join(ctx, on="s1")
    tx = texts(set(err.s1) | set(err.sx) | set(err.win_s1.dropna()) | set(err.own_s1.dropna()))
    err = side(side(side(err, "s1", tx, "s1_"), "sx", tx, "sx_"), "win_s1", tx[["name", "addr", "hn"]], "win_")
    err = side(err, "own_s1", tx[["name", "addr", "hn"]], "own_")
    err.to_parquet(D / f"err_{c}.parquet", index=False)
    ctx.to_parquet(D / f"s1ctx_{c}.parquet")
    print(c, err.kind.value_counts().to_dict(), f"{len(err.columns)} cols", flush=True)
    del df, h, err, tx
