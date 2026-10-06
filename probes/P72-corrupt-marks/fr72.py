"""P72 France: corruption marks as extra columns of P55's country-invariant combiner, gated on both LOCO directions
(P65 ev65.loco's protocol: P55 bag of 3 seeds fit on the source frame, scored on the target; shipped cell = veto band
c < .6 + add free c > .95; paired per-S1 dF vs the base 20-feature combiner; gate India->US >= +0.0003, US->India >= 0).
    .venv/bin/python probes/P72-corrupt-marks/fr72.py marks        # frame_<src>_<tgt> (prep65) -> data/work/p72/mk_loco_*
    [P72_DROP=cols] [P72_TAG=x] .venv/bin/python probes/P72-corrupt-marks/fr72.py loco
"""
import json
import os
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P72-corrupt-marks"), str(ROOT / "probes/P65-byt5-corrector"), str(ROOT / "probes/P55-fr-combiner"), str(ROOT)]
import marks as M  # noqa: E402

W = ROOT / "data/work/p72"
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
DIRS = {"US->India": ("US", "India"), "India->US": ("India", "US")}


def split_txt(s: np.ndarray) -> tuple:
    """raw 'name | addr' -> (name, addr) at the LAST ' | ' (ev65.split without its normalisation)."""
    p = pd.Series(s, dtype="str").str.rpartition(" | ")
    return p[0].where(p[1] != "", p[2]).to_numpy(object), p[2].where(p[1] != "", "").to_numpy(object)


def marks():
    for src, tgt in DIRS.values():
        t0 = time.time()
        f = pd.read_parquet(ROOT / f"data/work/p65/sm/frame_{src}_{tgt}.parquet", columns=["row", "sx_txt", "s1_txt"])
        assert (f.row.to_numpy() == np.arange(len(f))).all()
        vocab = pickle.loads((W / f"vocab_train_{tgt}.pkl").read_bytes())     # the target's own S1 pool (France: test S1)
        n1, a1 = split_txt(f.s1_txt.to_numpy())
        nx, ax = split_txt(f.sx_txt.to_numpy())
        d = M.marks(n1, a1, nx, ax, vocab)
        d.to_parquet(W / f"mk_loco_{src}_{tgt}.parquet")
        print(src, tgt, len(d), f"{time.time() - t0:.0f}s", flush=True)


def loco():
    """ev65.loco with the corrector columns replaced by MK (all rows; P72_BAND=1: band rows only, as P65)."""
    import p55
    import ev65
    from ledger import log
    t0 = time.time()
    mk = [m for m in M.MK if m not in os.environ.get("P72_DROP", "").split(",")]
    band, tag = os.environ.get("P72_BAND") == "1", os.environ.get("P72_TAG", "")
    frames = {dr: p55.loco_frame(dr) for dr in p55.L.DIRS}
    cols = {dr: pd.read_parquet(W / f"mk_loco_{dr}.parquet", columns=mk) for dr in p55.L.DIRS}
    if band:
        cols = {dr: X.where(frames[dr]["X"]["region"].to_numpy()[:, None] == 1) for dr, X in cols.items()}
    print(f"frames {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    out = {}
    for name, trn, tst in p55.DIRN:
        a, b = frames[trn], frames[tst]
        fa, fb = cols[trn], cols[tst]
        assert len(fa) == len(a["X"]) and len(fb) == len(b["X"])
        R, C = out.setdefault(name, {}), {}
        rg = b["X"]["region"].to_numpy()
        for fs, extra in (("base", []), ("mk", mk)):
            Xa = pd.concat([a["X"], fa[extra]], axis=1) if extra else a["X"]
            Xb = pd.concat([b["X"], fb[extra]], axis=1) if extra else b["X"]
            bs = p55.bag(Xa, a["z"]["y"], None)
            cb = C[fs] = p55.bag_score(bs, Xb, None)
            R[fs] = {"auc_ref": p55.auc(b["z"]["y"][b["ref"]], cb[b["ref"]]), "auc_band": p55.auc(b["z"]["y"][rg == 1], cb[rg == 1])}
            if extra:
                imp = pd.Series(bs[0].feature_importance("gain"), index=Xa.columns)
                R["imp_mk"] = {k: round(float(imp[k] / imp.sum()), 4) for k in extra}
                R["imp_mk_share"] = round(float(imp[extra].sum() / imp.sum()), 4)
            print(name, fs, R[fs], f"{time.time() - t0:.0f}s", flush=True)
        hv = b["z"]["hv"]
        for k in ("ship", "veto06"):
            f0 = p55.L.F(b["z"], ev65._new(p55, b, C["base"], k))
            d_ = p55.L.F(b["z"], ev65._new(p55, b, C["mk"], k)) - f0
            R[f"mk-base_{k}"] = {"dF": round(float(d_.mean()), 6), "se": round(float(d_.std() / np.sqrt(len(d_))), 6),
                                 "dF_h0": round(float(d_[hv == 0].mean()), 6), "dF_h1": round(float(d_[hv == 1].mean()), 6)}
            print(name, k, "mk-base", R[f"mk-base_{k}"], flush=True)
    g = {n: out[n]["mk-base_ship"]["dF"] for n in out}
    ok = g["India->US"] >= 0.0003 and g["US->India"] >= 0
    out["gate"] = {"rule": "mk-base at the shipped cell: India->US >= +0.0003 and US->India >= 0 (P65's)", "pass": bool(ok)}
    out["runtime_s"], out["peak_gb"] = round(time.time() - t0), round(peak(), 2)
    (W / f"fr72{tag}.json").write_text(json.dumps(out, indent=1))
    log("exp", probe="P72", subproblem="france-combiner", name=f"p72-fr-loco-mk{tag}", world="holdout-loco (P42/P55 frames)",
        metrics={f"{n}|{k}": out[n][f"mk-base_{k}"] for n in g for k in ("ship", "veto06")} |
                {f"{n}|imp_share": out[n]["imp_mk_share"] for n in g},
        config={"cols": mk, "band_only": band, "combiner": "P55 bag 3 seeds (P65 ev65.loco protocol)"},
        hypothesis="raw-text corruption marks add France-analogue F0.5 over P55's 20-feature combiner",
        verdict="keep" if ok else "drop", runtime_s=out["runtime_s"], peak_gb=out["peak_gb"])
    print("GATE", "PASS" if ok else "FAIL", g, f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)




NUM = ["num_trunc", "num_step", "num_other"]


def cells():
    """Decompose num's LOCO gain vs production's P55 stage (band veto c < .6, no adds): veto swap, and adds at bar u on
    top of the base veto (each combiner), with add precision, per direction and half -> data/work/p72/fr72_cells.json."""
    import p55
    t0 = time.time()
    frames = {dr: p55.loco_frame(dr) for dr in p55.L.DIRS}
    cols = {dr: pd.read_parquet(W / f"mk_loco_{dr}.parquet", columns=NUM) for dr in p55.L.DIRS}
    out = {}
    for name, trn, tst in p55.DIRN:
        a, b = frames[trn], frames[tst]
        rg, y, hv = b["X"]["region"].to_numpy(), b["z"]["y"], b["z"]["hv"]
        C = {fs: p55.bag_score(p55.bag(pd.concat([a["X"], cols[trn][ex]], axis=1) if ex else a["X"], a["z"]["y"], None),
                               pd.concat([b["X"], cols[tst][ex]], axis=1) if ex else b["X"], None)
             for fs, ex in (("base", []), ("num", NUM))}
        veto = {fs: b["ref"] & ~(b["ref"] & (C[fs] < 0.6) & (rg == 1)) for fs in C}
        f0 = p55.L.F(b["z"], veto["base"])
        R = out[name] = {}

        def rec(tag, m):
            d_ = p55.L.F(b["z"], m) - f0
            add, rm = m & ~veto["base"], veto["base"] & ~m
            R[tag] = {"dF": round(float(d_.mean()), 6), "se": round(float(d_.std() / np.sqrt(len(d_))), 6),
                      "h0": round(float(d_[hv == 0].mean()), 6), "h1": round(float(d_[hv == 1].mean()), 6),
                      "adds": int(add.sum()), "add_prec": round(float(y[add].mean()), 4) if add.any() else None,
                      "rms": int(rm.sum()), "rm_false": round(float(1 - y[rm].mean()), 4) if rm.any() else None}
            print(name, tag, R[tag], flush=True)
        rec("veto_num", veto["num"])
        for fs in C:
            for u in (0.95, 0.97, 0.98, 0.99):
                rec(f"add_{fs}_{u}", p55.decide({**b, "ref": veto["base"]}, C[fs], None, u))
        for u in (0.97, 0.98, 0.99):
            rec(f"veto_num+add_num_{u}", p55.decide({**b, "ref": veto["num"]}, C["num"], None, u))
    out["runtime_s"] = round(time.time() - t0)
    (W / "fr72_cells.json").write_text(json.dumps(out, indent=1))


def frscore():
    """P55's pooled bag (3 seeds, both LOCO frames) with + without NUM, scored on P55's France frame (prep65 france's texts,
    row = X row) -> data/work/p55/cf_p72num.npy (+ cf_p72base.npy, which must reproduce cf_plain), so that
    `p55.py build <base> <out> mode=p72num,v=0,scope=band,u=..` adds with it."""
    import p55
    t0 = time.time()
    f = pd.read_parquet(ROOT / "data/work/p65/sm/frame_France.parquet", columns=["row", "sx_txt", "s1_txt"])
    assert (f.row.to_numpy() == np.arange(len(f))).all()
    n1, a1 = split_txt(f.s1_txt.to_numpy())
    nx, ax = split_txt(f.sx_txt.to_numpy())
    del f
    mf = pd.concat([M.marks(n1[i:i + 200_000], a1[i:i + 200_000], nx[i:i + 200_000], ax[i:i + 200_000], M.Counter())[NUM]
                    for i in range(0, len(n1), 200_000)], ignore_index=True)       # NUM needs no vocab
    del n1, a1, nx, ax
    mf.to_parquet(W / "mk_France_num.parquet")
    print("France marks", len(mf), mf.mean().round(4).to_dict(), f"{time.time() - t0:.0f}s", flush=True)
    fr = p55.france_frame()
    assert len(fr["X"]) == len(mf)
    for fs, ex in (("p72base", []), ("p72num", NUM)):
        Xs, ys = [], []
        for dr in p55.L.DIRS:
            z = p55.loco_frame(dr)
            mk = pd.read_parquet(W / f"mk_loco_{dr}.parquet", columns=NUM)
            Xs.append(pd.concat([z["X"], mk[ex]], axis=1) if ex else z["X"])
            ys.append(z["z"]["y"])
            del z, mk
        bs = p55.bag(pd.concat(Xs, ignore_index=True), np.concatenate(ys), None)
        del Xs, ys
        cf = p55.bag_score(bs, pd.concat([fr["X"], mf[ex]], axis=1) if ex else fr["X"], None)
        np.save(ROOT / f"data/work/p55/cf_{fs}.npy", cf)
        rg = fr["X"]["region"].to_numpy()
        print(fs, {u: int((fr["free"] & ~fr["ref"] & (cf > u)).sum()) for u in (0.95, 0.98, 0.99)},
              "band veto<.6", int((fr["ref"] & (cf < .6) & (rg == 1)).sum()), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    c0 = np.load(ROOT / "data/work/p55/cf_plain.npy")
    print("p72base vs cf_plain max |d|", float(np.abs(np.load(ROOT / "data/work/p55/cf_p72base.npy") - c0).max()), flush=True)


def calib(u=0.99):
    """LOCO calibration of the France add audit: the u-bar adds (on the base veto, as build's v=0 on production) of the
    base and num combiners -> precision, out-of-domain Qwen r >= .5 share of true / false adds, n target S1."""
    import p55
    QW = {"India_US": "r_qwen_locoIndia_holdout_US", "US_India": "r_qwen_locoUS_holdout_India"}
    frames = {dr: p55.loco_frame(dr) for dr in p55.L.DIRS}
    cols = {dr: pd.read_parquet(W / f"mk_loco_{dr}.parquet", columns=NUM) for dr in p55.L.DIRS}
    out = {}
    for name, trn, tst in p55.DIRN:
        a, b = frames[trn], frames[tst]
        rg, y = b["X"]["region"].to_numpy(), b["z"]["y"]
        ids = pd.read_parquet(ROOT / f"data/work/p65/sm/frame_{tst}.parquet", columns=["s1", "sx"])
        q = pd.read_parquet(ROOT / f"data/work/p38/r/{QW[tst]}.parquet").drop_duplicates(["s1", "sx"])
        r = ids.merge(q, on=["s1", "sx"], how="left").r.to_numpy()
        C = {fs: p55.bag_score(p55.bag(pd.concat([a["X"], cols[trn][ex]], axis=1) if ex else a["X"], a["z"]["y"], None),
                               pd.concat([b["X"], cols[tst][ex]], axis=1) if ex else b["X"], None)
             for fs, ex in (("base", []), ("num", NUM))}
        vb = b["ref"] & ~(b["ref"] & (C["base"] < 0.6) & (rg == 1))
        R = out[name] = {"n_s1": int(b["n"])}
        adds = {fs: p55.decide({**b, "ref": vb}, C[fs], None, u) & ~vb for fs in C}
        for tag, m in (("base", adds["base"]), ("num", adds["num"]), ("num_only", adds["num"] & ~adds["base"]),
                       ("both", adds["num"] & adds["base"])):
            rr, yy = r[m], y[m].astype(bool)
            cov = ~np.isnan(rr)
            R[tag] = {"n": int(m.sum()), "prec": round(float(yy.mean()), 4) if m.any() else None,
                      "qwen_cov": round(float(cov.mean()), 3) if m.any() else None,
                      "ge05": round(float((rr[cov] >= .5).mean()), 4) if cov.any() else None,
                      "ge05_true": round(float((rr[cov & yy] >= .5).mean()), 4) if (cov & yy).any() else None,
                      "ge05_false": round(float((rr[cov & ~yy] >= .5).mean()), 4) if (cov & ~yy).any() else None}
        print(name, R, flush=True)
    (W / f"fr72_calib_{u}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"marks": marks, "loco": loco, "cells": cells, "frscore": frscore, "calib": calib}[sys.argv[1]]()
