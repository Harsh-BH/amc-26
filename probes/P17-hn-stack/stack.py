"""P17: Brief A of P15 -- residual LightGBM on #7's blended p over #7's PRUNED candidate list, with P15's generator-aware
house-number columns (hn_check.hn_cls, signed delta, eq) + SX<->SX support (hn_stack.support), computed on that list.

#7's p is rebuilt from run.py's own caches (m2_big p on the tau-0.03 pruned list, the 2-epoch CE r cache, sub7 blend w),
so the base arm reproduces run.py --split holdout (half 1 0.98869) and the test files reproduce output/sub7.
Stacker = p' = sigmoid(logit p + f(X)), LightGBM 200 x 31, lr 0.05, min_data_in_leaf 200, init_score = logit p,
one model per labelled country, fit on the holdout S1 of P5's half 0 (run.half). t tuned on half 0, per country.
Report = half 1, all holdout S1 (blocking misses count: the run.py yardstick), paired per-S1 SE vs #7 at t 0.71.

    .venv/bin/python probes/P17-hn-stack/stack.py holdout [arm] [swap] [t=0.71] [l2=1]   # l2: lambda_l2; swap: fit half 1, report half 0; t=: no t tuning; arm: sup (brief A, default) | hn | brc (+ brief C)
    .venv/bin/python probes/P17-hn-stack/stack.py test [arm] [t=0.71]   # -> data/work/p17/prob_test_{India,US}.parquet
"""
import json
import resource
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P15-research"), str(ROOT)]
import features as F  # noqa: E402
import model as M  # noqa: E402
import rerank as RR  # noqa: E402
from hn_check import hn_cls  # noqa: E402
from hn_stack import f05_per_s1, support  # noqa: E402
from run import half, holdout_s1  # noqa: E402
from worlds import WORK  # noqa: E402

OUT = WORK / "p17"
W7, T7 = [0.617, 0.666, 0.261], 0.71                      # sub7 blend w and India/US t
M2 = "m2_{split}_{c}_d2b6569f_6a23c2ec_0.03.parquet"      # m2_big p on the pruned list (run.py's cache)
RC = "r_{split}_{c}_bb0c29f0.parquet"                     # ce_s2P r cache (run.py's)
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=4,
           seed=0, deterministic=True)
TS = np.round(np.arange(0.50, 0.96, 0.01), 2)
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))


def frame(split: str, c: str) -> dict:
    """#7's scored rows of one country: q1, qx, p (blended), h1 / hx (house numbers), id1 / idx."""
    d = pd.read_parquet(WORK / "p9" / M2.format(split=split, c=c))
    q1, qx, pm = d.q1.to_numpy(np.int64), d.qx.to_numpy(np.int64), d.p.to_numpy(np.float32)
    r = pd.read_parquet(WORK / "p9" / RC.format(split=split, c=c))
    rr = pd.Series(r.r.to_numpy(np.float32), index=r.k.to_numpy(np.int64)).reindex((q1 << 32) | qx).to_numpy(np.float32)
    p = RR.blend(pm, rr, W7, *RR.CFG["band"])
    xs = "train" if split == "holdout" else split
    id1, idx = F.ids(xs, c)

    def hn(ids: pa.Array, q: np.ndarray, side: str) -> np.ndarray:
        rf = pd.read_parquet(WORK / f"rf_{xs}_{side}_{c}.parquet", columns=["id", "hn"])
        pos = pc.index_in(ids, value_set=pa.array(rf.id.to_numpy())).to_numpy(zero_copy_only=False)
        return rf.hn.fillna("").to_numpy(object)[pos.astype(np.int64)][q]

    return {"q1": q1, "qx": qx, "p": p, "h1": hn(id1, q1, "s1"), "hx": hn(idx, qx, "sx"), "id1": id1, "idx": idx,
            "xs": xs, "c": c}


def X_of(d: dict, arm: str) -> pd.DataFrame:
    """P15 hn_stack.main's X1 (+ support for arm sup / brc), on the list d holds."""
    h1, hx = d["h1"], d["hx"]
    both = (h1 != "") & (hx != "")
    dd = np.full(len(h1), np.nan, np.float32)
    dd[both] = np.clip(pd.to_numeric(pd.Series(hx[both])).to_numpy(np.float64)
                       - pd.to_numeric(pd.Series(h1[both])).to_numpy(np.float64), -100, 100)
    X = pd.DataFrame({"lp": lg(d["p"]).astype(np.float32), "cls": hn_cls(h1, hx).astype(np.float32), "d": dd,
                      "eq": (both & (h1 == hx)).astype(np.float32)})
    if arm in ("sup", "brc"):
        X = pd.concat([X, support(pd.DataFrame({"q1": d["q1"], "p": d["p"]}), h1, hx)], axis=1)
    if arm == "brc":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import branch
        X = pd.concat([X, branch.cols(d)], axis=1)
    return X


def labels(d: dict, c: str, s1: pd.DataFrame, t: pd.DataFrame):
    """pos (row -> scored S1 index, -1 = a competing train S1), y (true pair), ntrue per scored S1, half per scored S1."""
    ids = pa.array(s1.entity_id[s1.country == c].to_numpy())
    code = pc.index_in(ids, value_set=d["id1"]).to_numpy(zero_copy_only=False).astype(np.int64)
    inv = np.full(len(d["id1"]), -1, np.int64)
    inv[code] = np.arange(len(ids))
    pos = inv[d["q1"]]
    tt = t[t.s1.isin(set(ids.to_pylist()))]
    ntrue = np.bincount(inv[pc.index_in(pa.array(tt.s1.to_numpy()), value_set=d["id1"]).to_numpy(
        zero_copy_only=False).astype(np.int64)], minlength=len(ids))
    k1 = pc.index_in(pa.array(tt.s1.to_numpy()), value_set=d["id1"]).to_numpy(zero_copy_only=False).astype(np.int64)
    kx = pc.index_in(pa.array(tt.sx.to_numpy()), value_set=d["idx"]).to_numpy(zero_copy_only=False)
    ok = ~pd.isna(kx)
    y = np.isin((d["q1"] << 32) | d["qx"], (k1[ok] << 32) | kx[ok].astype(np.int64))
    return pos, y, ntrue, half(ids.to_pylist())


def f05(am, s, t, pos, y, ntrue):
    """Per-scored-S1 F0.5 of run.py's decision (am = M.argmax_sx over all rows, s >= t)."""
    acc = am & (s >= t)
    m = pos >= 0
    n = len(ntrue)
    tp = np.bincount(pos[m], weights=(acc & y)[m], minlength=n)
    npred = np.bincount(pos[m], weights=acc[m], minlength=n)
    return f05_per_s1(None, tp, npred, ntrue)


def fit(X: pd.DataFrame, y: np.ndarray, tr: np.ndarray) -> lgb.Booster:
    return lgb.train(PRM, lgb.Dataset(X[tr], y[tr].astype(np.int8), init_score=X.lp.to_numpy()[tr],
                                      categorical_feature=["cls"]), 200)


def stacked(m: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return (1 / (1 + np.exp(-(X.lp.to_numpy(np.float64) + m.predict(X, raw_score=True))))).astype(np.float32)


def holdout(arm: str, fh: int = 0, fixt: float | None = None):
    """Fit + tune t on half fh (0 = the shipped model), report the other half against #7 at T7. fixt: no t tuning."""
    t0 = time.time()
    s1, t = holdout_s1()
    base, new, ctrl, res = [], [], [], {}
    mt = arm + (f"-l2{PRM['lambda_l2']}" if "lambda_l2" in PRM else "") + ("-swap" if fh else "")   # model tag
    tag = mt + (f"-t{fixt}" if fixt else "")
    for c in ("India", "US"):
        d = frame("holdout", c)
        pos, y, ntrue, hv = labels(d, c, s1, t)
        X = X_of(d, arm)
        del d["h1"], d["hx"]
        tr = (pos >= 0) & (hv[np.maximum(pos, 0)] == fh)
        m = fit(X, y, tr)
        s = stacked(m, X)
        p, qx = d["p"], d["qx"]
        a7, am = M.argmax_sx(qx, p), M.argmax_sx(qx, s)
        h0, h1_ = hv == fh, hv == 1 - fh
        tb = max(TS, key=lambda t_: f05(a7, p, t_, pos, y, ntrue)[h0].mean())         # control: #7 p, t retuned
        ts = fixt or max(TS, key=lambda t_: f05(am, s, t_, pos, y, ntrue)[h0].mean())
        fb, fn, fc = (f05(a7, p, T7, pos, y, ntrue), f05(am, s, ts, pos, y, ntrue), f05(a7, p, tb, pos, y, ntrue))
        dl = (fn - fb)[h1_]
        curve = {f"{t_:.2f}": round((f05(am, s, t_, pos, y, ntrue) - fb)[h1_].mean(), 5)
                 for t_ in np.round(np.arange(0.61, 0.86, 0.02), 2)}                  # report-half d by t (stability)
        res[c] = {"t": float(ts), "t_ctrl": float(tb), "f05_7_h1": round(fb[h1_].mean(), 5),
                  "f05_stack_h1": round(fn[h1_].mean(), 5), "d": round(dl.mean(), 5),
                  "se": round(dl.std() / np.sqrt(len(dl)), 5), "d_ctrl": round((fc - fb)[h1_].mean(), 5),
                  "f05_7_all": round(fb.mean(), 5), "n_h1": int(h1_.sum()), "up": int((dl > 0).sum()),
                  "down": int((dl < 0).sum()), "rows": len(qx), "curve_h1": curve,
                  "gain": dict(zip(m.feature_name(), m.feature_importance("gain").round(0).tolist()))}
        print(c, res[c], f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
        base.append(fb[h1_]), new.append(fn[h1_]), ctrl.append(fc[h1_])
        OUT.mkdir(parents=True, exist_ok=True)
        m.save_model(str(OUT / f"stack_{mt}_{c}.txt"))
        del X, s, m, d
    fb, fn, fc = (np.concatenate(z) for z in (base, new, ctrl))
    dl = fn - fb
    res["all"] = {"f05_7_h1": round(fb.mean(), 5), "f05_stack_h1": round(fn.mean(), 5), "d": round(dl.mean(), 5),
                  "se": round(dl.std() / np.sqrt(len(dl)), 5), "d_ctrl": round((fc - fb).mean(), 5), "n_h1": len(dl)}
    print("India+US", res["all"], f"{time.time() - t0:.0f}s", flush=True)
    (OUT / f"holdout_{tag}.json").write_text(json.dumps(res, indent=1))
    return res, time.time() - t0


def test(arm: str, tag: str):
    """Stacked p over #7's pruned test rows, t from holdout_<tag>.json -> data/work/p17/prob_test_<c>.parquet."""
    res = json.loads((OUT / f"holdout_{tag}.json").read_text())
    for c in ("India", "US"):
        d = frame("test", c)
        X = X_of(d, arm)
        s = stacked(lgb.Booster(model_file=str(OUT / f"stack_{tag.split('-t')[0]}_{c}.txt")), X)
        out = pd.DataFrame({"s1": d["id1"].take(pa.array(d["q1"])).to_numpy(zero_copy_only=False),
                            "sx": d["idx"].take(pa.array(d["qx"])).to_numpy(zero_copy_only=False), "p": s})
        am7, am = M.argmax_sx(d["qx"], d["p"]), M.argmax_sx(d["qx"], s)
        a7, a = am7 & (d["p"] >= T7), am & (s >= res[c]["t"])
        print(f"{c}: {len(s):,} rows; accepted #7 {int(a7.sum()):,} -> {int(a.sum()):,} at t {res[c]['t']}; "
              f"pairs changed {int((a7 != a).sum()):,}; S1 changed {len(np.unique(d['q1'][a7 != a])):,}; "
              f"peak {peak():.2f} GB", flush=True)
        out.to_parquet(OUT / f"prob_test_{c}.parquet", index=False)
        del d, X, s, out


if __name__ == "__main__":
    assert list(f05_per_s1(None, np.array([1, 0, 0]), np.array([1, 0, 1]), np.array([2, 0, 0]))) == \
        [1.25 * 0.5 / (0.25 + 0.5), 1.0, 0.0]
    mode, arm = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "sup")
    if mode == "test":
        test(arm, arm + "".join(f"-l2{float(a[3:])}" for a in sys.argv[3:] if a.startswith("l2="))
             + "".join(f"-t{a[2:]}" for a in sys.argv[3:] if a.startswith("t=")))
        sys.exit()
    fh = int("swap" in sys.argv[3:])
    fixt = next((float(a[2:]) for a in sys.argv[3:] if a.startswith("t=")), None)
    PRM.update({"lambda_l2": float(a[3:]) for a in sys.argv[3:] if a.startswith("l2=")})   # leaf blow-up guard
    r, rt = holdout(arm, fh, fixt)
    from ledger import log
    a = r["all"]
    log("exp", probe="P17-hn-stack", subproblem="decide", name=f"hn-stack-{arm}-on7" + (f"-l2{PRM['lambda_l2']}" if "lambda_l2" in PRM else "") + ("-swap" if fh else "") + (f"-t{fixt}" if fixt else ""),
        world=f"holdout-dense half{1 - fh} (#7 pruned list, run.py yardstick)",
        metrics={"f05_7_h1": a["f05_7_h1"], "f05_stack_h1": a["f05_stack_h1"], "d": a["d"], "se": a["se"],
                 "d_ctrl_t_retuned": a["d_ctrl"], "d_India": r["India"]["d"], "se_India": r["India"]["se"],
                 "d_US": r["US"]["d"], "se_US": r["US"]["se"], "t_India": r["India"]["t"], "t_US": r["US"]["t"]},
        config={"stacker": "lgb 200x31 lr .05 mdl 200 residual init_score=logit p, per country", "lambda_l2": PRM.get("lambda_l2", 0),
                "X": {"hn": "lp cls d eq", "sup": "lp cls d eq n_same p_same n_s1hn",
                      "brc": "sup + add1 add_in_list"}[arm], "list": "#7 pruned (tau .03)", "p": "#7 blend",
                "train": f"holdout S1 half {fh} (run.half)", "t": f"fixed {fixt}" if fixt else f"per country, tuned on half {fh}"},
        hypothesis="P15 brief A: generator hn offsets + same-hn support fix US siblings on top of the CE",
        verdict="keep" if a["d"] >= 0.0003 and a["d"] >= 3 * a["se"] and r["India"]["d"] >= -0.0001 else "drop",
        runtime_s=round(rt), peak_gb=round(peak(), 2))
