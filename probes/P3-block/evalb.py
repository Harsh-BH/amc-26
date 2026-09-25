"""Recall bookkeeping for P3: raw view hits -> recall@k / cands per S1 / full-recall, per country and class.

    .venv/bin/python probes/P3-block/evalb.py dev            # all views, k up to 100, on a world
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import block as B

KS = (5, 10, 20, 50, 100)
_INDIC = r"[ऀ-෿]"


def truth_df(truth: dict) -> pd.DataFrame:
    return pd.DataFrame([(a, b) for a, s in truth.items() for b in s], columns=["s1", "sx"])


def classify(tp: pd.DataFrame, s1: pd.DataFrame, sx: pd.DataFrame, cfg=B.CFG) -> pd.Series:
    """Stopgap difficulty class per true pair (P1's taxonomy replaces it when it lands). First match wins."""
    a = s1.set_index("entity_id").reindex(tp.s1)
    b = sx.set_index("entity_id").reindex(tp.sx)
    na, nb = B.clean(a.business_name, "none").to_numpy(), B.clean(b.business_name, "none").to_numpy()
    raw_b = b.business_name.fillna("").astype("string")
    cls = np.full(len(tp), "token_overlap", dtype=object)
    tok = np.array([bool(set(x.split()) & set(y.split())) for x, y in zip(na, nb)])
    cls[~tok] = "no_token_overlap"
    cls[raw_b.str.contains(r"(?i)www\.|\.com\b|^[@#]|\.in\b|\.org\b|\.net\b", regex=True).to_numpy()] = "website"
    cls[raw_b.str.contains(_INDIC, regex=True).to_numpy()] = "cross_script"
    cls[na == nb] = "exact"
    return pd.Series(cls, index=tp.index)


def hits_world(s1, sx, cfg, vecs, log=print):
    """Long hit table with entity ids: s1, sx, score, rank, view, dir."""
    out = []
    for c in sorted(set(s1.country) & set(sx.country)):
        a, b = s1[s1.country == c].reset_index(drop=True), sx[sx.country == c].reset_index(drop=True)
        t0 = time.time()
        h = B.run_views(a, b, vecs, cfg)
        h["s1"] = a.entity_id.to_numpy()[h.q1]
        h["sx"] = b.entity_id.to_numpy()[h.qx]
        out.append(h.drop(columns=["q1", "qx"]))
        log(f"  {c}: {len(a):,} x {len(b):,} views in {time.time() - t0:.0f}s")
    return pd.concat(out, ignore_index=True)


def select(h: pd.DataFrame, fwd: dict, rev: dict) -> pd.DataFrame:
    kf = h.view.map(fwd).fillna(0).to_numpy()
    kr = h.view.map(rev).fillna(0).to_numpy()
    m = np.where(h["dir"].to_numpy() == "f", h["rank"].to_numpy() < kf, h["rank"].to_numpy() < kr)
    return h.loc[m, ["s1", "sx"]].drop_duplicates()


def metrics(sel: pd.DataFrame, tp: pd.DataFrame, s1: pd.DataFrame) -> dict:
    """tp carries country + cls columns. Returns overall / per-country / per-class recall and cands/S1."""
    hit = tp.merge(sel.assign(_h=1), on=["s1", "sx"], how="left")._h.notna().to_numpy()
    m = {"pair_recall": hit.mean(), "cands_per_s1": len(sel) / len(s1)}
    full = pd.Series(hit).groupby(tp.s1.to_numpy()).all()
    m["s1_full_recall"] = full.mean()
    m["s1_any_recall"] = pd.Series(hit).groupby(tp.s1.to_numpy()).any().mean()
    cty = s1.set_index("entity_id").country
    nc = sel.s1.map(cty).value_counts()
    for c, g in tp.groupby("country"):
        m[f"recall_{c}"] = hit[g.index].mean()
        m[f"cands_{c}"] = nc.get(c, 0) / (s1.country == c).sum()
    for k, g in tp.groupby("cls"):
        m[f"recall_{k}"] = hit[g.index].mean()
    return m


def curve(h, tp, s1, view, dirn="f"):
    """recall@k of one view/direction for k in KS."""
    g = h[(h.view == view) & (h["dir"] == dirn)]
    r = tp.merge(g[["s1", "sx", "rank"]], on=["s1", "sx"], how="left")["rank"].to_numpy()
    return {k: float(np.nanmean(np.where(np.isnan(r), 0, r < k)) if len(r) else 0) for k in KS}


if __name__ == "__main__":
    from worlds import load_world
    world = sys.argv[1] if len(sys.argv) > 1 else "dev"
    s1, sx, truth = load_world(world)
    tp = truth_df(truth)
    tp["country"] = tp.s1.map(s1.set_index("entity_id").country)
    tp["cls"] = classify(tp, s1, sx)
    print(tp.cls.value_counts(normalize=True).round(3).to_dict())
    cfg = {**B.CFG, "views": {"name": 100, "addr": 100, "na": 100}, "rev": {"name": 10, "addr": 10, "na": 10}}
    vecs = B.fit_views(sx, cfg)
    h = hits_world(s1, sx, cfg, vecs)
    for v in ("name", "addr", "na"):
        print(v, "fwd", curve(h, tp, s1, v), "rev@1..", curve(h, tp, s1, v, "r"))
