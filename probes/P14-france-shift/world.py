"""Labelled France-shaped proxy: a 3-state sub-world of ONE train country (France: 259k S1 in 3 régions). Every
world S1 and world S2/S3 competes, the fold-0 S1 are scored, with #6's models (m2_big + pruner 0.03, M2 p only, no
CE -- France gets no CE either). Rebuilt end to end in a scratch ER_WORK: texts -> rules.feats -> P3 blocking -> X.

World: S1 whose parsed state is in --states; S2/S3 = every child of those S1 + the unmatched S2/S3 of those states
+ a same-share sample of the unmatched state-less ones. Variants (data / vectorizer swaps, src untouched):
  --blank F   remove the state segment(s) from the raw address of a seeded fraction F of the stated S2/S3
              (0.33: 3% -> 35% state-less, France's rate)
  --fill      then variant.py's fill: state from the city (world S1 majority), appended to the raw address
  --idf pooled|loco|loco-own   pooled = #6's train IDF (US + India); loco = IDF of the OTHER train country only
              (this country's tokens under-counted, as France's are now); loco-own = loco + the world's own S2/S3
              at --share of the pool (variant.py --idf)

    .venv/bin/python probes/P14-france-shift/world.py NAME --country India --states dl,up,ka [--blank 0.33] [--fill]
        [--idf loco-own]
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "data/work"
sys.path[:0] = [str(HERE), str(ROOT / "code/business_entity_resolution/src")]
import variant as V

SHARED = ["gt_pairs.parquet", "s1_folds.parquet", "place_freq_train.json", "p2_xlit_table.parquet",
          "p4/m2_big.txt", "p10/pruner.txt"]


def select(c: str, states: list, seed: int = 0) -> tuple:
    r1 = pd.read_parquet(BASE / f"rf_train_s1_{c}.parquet", columns=["id", "state"])
    rx = pd.read_parquet(BASE / f"rf_train_sx_{c}.parquet", columns=["id", "state"])
    gt = pd.read_parquet(BASE / "gt_pairs.parquet")
    s1 = set(r1.id[r1.state.isin(states)])
    kids = set(gt.sx[gt.s1.isin(s1)])
    free = ~rx.id.isin(set(gt.sx))
    share = len(s1) / len(r1)
    rng = np.random.default_rng(seed)
    sx = rx.id.isin(kids) | (free & rx.state.isin(states)) | (free & (rx.state == "") & (rng.random(len(rx)) < share))
    return s1, set(rx.id[sx]), share


def raw(name: str, ids: set) -> pd.DataFrame:
    """Rows of data/work/<name>.parquet whose entity_id is in ids, streamed (never the whole file in RAM)."""
    vs, out = pa.array(sorted(ids)), []
    for b in pq.ParquetFile(BASE / f"{name}.parquet").iter_batches(500_000):
        out.append(b.filter(pc.is_in(b["entity_id"], value_set=vs)))
    return pa.Table.from_batches(out).to_pandas()


def blank(d: pd.DataFrame, c: str, frac: float, seed: int = 1) -> tuple:
    """Drop the segments that ARE a state from a seeded frac of the addresses that have one."""
    import normalize as N
    table = N.STATES[N.fold(pd.Series([c])).iloc[0]]
    seg = d.business_address.str.split(",").explode()
    is_st = pd.Series(N.fold(seg.fillna("")).map(N._basic).isin(table.keys()).to_numpy(), index=seg.index)
    has = is_st.groupby(level=0).any().reindex(d.index, fill_value=False).to_numpy()
    pick = has & (np.random.default_rng(seed).random(len(d)) < frac)
    sel = pd.Series(pick, index=d.index)
    s = seg[sel.reindex(seg.index).to_numpy() & ~is_st.to_numpy()]
    new = s.groupby(level=0).agg(",".join).reindex(d.index[pick]).fillna("").str.strip(" ,")
    d = d.copy()
    d.loc[pick, "business_address"] = new.to_numpy()
    return d, set(d.entity_id[pick])


def idf(v: Path, c: str, how: str, share: float, seed: int = 0, n: int = 1_000_000):
    import pickle
    vec = pickle.loads((BASE / "p3/vecs_hash.pkl").read_bytes())
    if how == "pooled":
        (v / "p3/vecs_hash.pkl").write_bytes(pickle.dumps(vec))
        return {}
    from sklearn.feature_extraction.text import TfidfTransformer
    other = {"India": "US", "US": "India"}[c]
    files = sorted((BASE / "p3/txt").glob(f"{other}_sx_*.parquet"))
    t = pd.concat([pq.read_table(f, columns=["name", "addr"]).to_pandas() for f in files], ignore_index=True)
    t = t.sample(n=min(n, len(t)), random_state=seed)
    own = pd.concat([pq.read_table(f, columns=["name", "addr"]).to_pandas()
                     for f in sorted((v / "p3/txt").glob(f"{c}_sx_*.parquet"))], ignore_index=True)
    m = round(len(t) * share / (1 - share)) if how == "loco-own" else 0
    t = pd.concat([t, own.sample(n=min(m, len(own)), random_state=seed)], ignore_index=True)
    for view in ("name", "addr"):
        hv = vec[view].steps[0][1]
        vec[view].steps[1] = (vec[view].steps[1][0], TfidfTransformer(sublinear_tf=True).fit(hv.transform(t[view])))
    (v / "p3/vecs_hash.pkl").write_bytes(pickle.dumps(vec))
    return {"idf_rows": len(t), "own_rows": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--country", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--blank", type=float, default=0.0)
    ap.add_argument("--fill", action="store_true")
    ap.add_argument("--idf", default="pooled")
    ap.add_argument("--share", type=float, default=0.144)
    ap.add_argument("--from", dest="src", help="reuse this world's raw subset, texts and rules.feats (IDF-only variants)")
    a = ap.parse_args()
    t0, c, states = time.time(), a.country, a.states.split(",")
    v = BASE / "p14" / f"wd_{a.name}"
    (v / "p3").mkdir(parents=True, exist_ok=True)
    for rel in SHARED:
        V.link(v, rel)
    info = {"country": c, "states": states}
    if a.src:
        w = BASE / "p14" / f"wd_{a.src}"
        for f in [*(w.glob("train_source*.parquet")), *(w.glob(f"rf_train_*_{c}.parquet")), *(w / "p3/txt").glob("*")]:
            d = v / f.relative_to(w)
            d.parent.mkdir(parents=True, exist_ok=True)
            if not d.exists():
                d.symlink_to(f.resolve())
    if not (v / "train_source3.parquet").exists():
        s1, sx, share = select(c, states)
        d1 = raw("train_source1", s1)
        d1.to_parquet(v / "train_source1.parquet", index=False)
        info |= {"s1": len(s1), "sx": len(sx), "world_share": round(share, 4)}
        added, blanked = {}, set()
        parts = {i: raw(f"train_source{i}", sx) for i in (2, 3)}
        if a.blank:
            for i in (2, 3):
                parts[i], b = blank(parts[i], c, a.blank)
                blanked |= b
        if a.fill:
            import normalize as N
            r1 = pd.read_parquet(BASE / f"rf_train_s1_{c}.parquet", columns=["id", "state", "places"])
            r1 = r1[r1.id.isin(s1)]
            rx = pd.read_parquet(BASE / f"rf_train_sx_{c}.parquet", columns=["id", "state", "places"])
            rx = rx[rx.id.isin(sx)].reset_index(drop=True)
            rx.loc[rx.id.isin(blanked), "state"] = ""             # state segments never enter places
            fill = V.fill_state(rx, V.city_states(r1))
            disp = V.state_display(d1.business_address, c)
            added = dict(zip(rx.id[fill != ""], fill[fill != ""].map(disp)))
            # fill precision against the true S1's state (labels used for the report only)
            gt = pd.read_parquet(BASE / "gt_pairs.parquet")
            tst = gt[gt.sx.isin(added.keys())].merge(r1[["id", "state"]], left_on="s1", right_on="id")
            info |= {"filled": len(added), "fill_share": round(len(added) / len(sx), 4),
                     "fill_precision_on_true": round(float((tst.sx.map(fill.set_axis(rx.id)) == tst.state).mean()), 4),
                     "fill_true_n": len(tst)}
            for i in (2, 3):
                parts[i] = V.append_state(parts[i], added)
        for i in (2, 3):
            parts[i].to_parquet(v / f"train_source{i}.parquet", index=False)
        info["blanked"] = len(blanked)
        del parts
    os.environ["ER_WORK"] = str(v)
    sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
    import block as B
    import features as F
    import model as M
    import run as RUN
    from metric import score
    B.CFG["threads"] = int(os.environ.get("P14_THREADS", 8))  # ponytail: 15 threads thrash on a shared, loaded box
    from ledger import log
    RUN.prepare("train", c)
    rxs = pd.read_parquet(v / f"rf_train_sx_{c}.parquet", columns=["state"])
    info["sx_state_less"] = round(float((rxs.state == "").mean()), 4)
    if not (v / "p3/vecs_hash.pkl").exists():
        info |= idf(v, c, a.idf, a.share)
    d = M.split_probs("holdout", c)
    id1, idx = F.ids("train", c)
    fo = pd.read_parquet(BASE / "s1_folds.parquet")
    ids = set(fo.entity_id[fo.fold == 0]) & set(id1.to_pylist())
    gt = pd.read_parquet(BASE / "gt_pairs.parquet")
    truth = {s: set() for s in ids}
    for s, x in zip(gt.s1, gt.sx):
        if s in truth:
            truth[s].add(x)
    q1, qx, p = d.q1.to_numpy(), d.qx.to_numpy(), d.p.to_numpy()
    s1s = np.array(id1.take(pa.array(q1)).to_pylist(), object)
    sxs = np.array(idx.take(pa.array(qx)).to_pylist(), object)
    am = M.argmax_sx(qx, p)
    tp_all = set(zip(s1s, sxs))
    n_true = sum(len(t) for t in truth.values())
    info["cand_recall"] = round(sum(x in tp_all for s, t in truth.items() for x in [(s, y) for y in t]) / n_true, 4)
    band = am & np.isin(s1s, list(ids))
    info["band_03_09"] = round(float(((p[band] > 0.3) & (p[band] < 0.9)).mean()), 4)
    curve = {}
    for t in np.round(np.arange(0.5, 0.99, 0.01), 2):
        k = am & (p >= t)
        pred = {}
        for s, x in zip(s1s[k], sxs[k]):
            if s in truth:
                pred.setdefault(s, set()).add(x)
        curve[float(t)] = score(pred, truth)["f05"]
    cv = pd.Series(curve)
    info |= {"n_scored": len(ids), "f05_069": round(cv.loc[0.69], 5), "t_oracle": float(cv.idxmax()),
             "f05_oracle": round(cv.max(), 5), "f05_080": round(cv.loc[0.8], 5), "f05_090": round(cv.loc[0.9], 5),
             "f05_095": round(cv.loc[0.95], 5)}
    pd.DataFrame({"s1": s1s, "sx": sxs, "p": p}).to_parquet(BASE / "p14" / f"prob_world_{a.name}.parquet", index=False)
    cv.to_csv(BASE / "p14" / f"curve_{a.name}.csv")
    for f in F.xfiles("train", c):
        f.unlink(missing_ok=True)
    info["runtime_s"], info["peak_gb"] = round(time.time() - t0), round(B.peak_gb(), 2)
    (BASE / "p14" / f"world_{a.name}.json").write_text(json.dumps(info, default=str, indent=1))
    print(json.dumps(info, default=str), flush=True)
    log("exp", probe="P14-france-shift", subproblem="france-proxy", name=f"world-{a.name}",
        world=f"train-{c}-{'+'.join(states)}-fold0", metrics={k: v for k, v in info.items()
                                                                if isinstance(v, (int, float)) and v is not None},
        config={"blank": a.blank, "fill": a.fill, "idf": a.idf, "share": a.share, "booster": "m2_big", "tau": 0.03},
        hypothesis="France-shaped labelled proxy: does missing state / IDF mismatch cost F0.5, does the fix win it back",
        verdict="inconclusive", runtime_s=info["runtime_s"], peak_gb=info["peak_gb"])


if __name__ == "__main__":
    main()
