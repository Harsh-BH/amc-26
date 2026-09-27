"""P65 evaluation: corrector features on P42/P55's LOCO frames, as extra columns of P55's country-invariant combiner.

  fetch JOB            s3 results/<JOB>/dec/*.parquet -> data/work/p65/dec/<JOB>/
  feats JOB FRAME      frame_<FRAME> (prep65) + that job's decodes -> data/work/p65/f_<FRAME>_<JOB>.parquet: per frame row
                       raw sims (SX vs S1) and corrected sims (decode vs S1): name / addr ratio, full token_set_ratio,
                       correction distance, corrected-name exact. Undecoded rows stay NaN.
  loco US_JOB IN_JOB   P55's pooled-bag procedure per direction (fit on the source's frame, bag 3 seeds, score target):
                       feature sets base (P55's 20) / +raw / +raw+corr. Direction A->B uses the A-trained corrector on
                       BOTH frames ('strict': in-domain on the fit side, out-of-domain on the target, no B labels).
                       Cells: shipped (veto band c < .6, add free c > .95), veto-only band .6, and the P55 grid.
                       -> probes/P65-byt5-corrector/loco65.json + ledger row

    .venv/bin/python probes/P65-byt5-corrector/ev65.py feats p65-locoUS-0927-1259 US_India
"""
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W = ROOT / "data/work/p65"
BUCKET = "s3://sagemaker-114872709824-ap-south-1/amc26-p65"
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
CORR = ["nm_c", "ad_c", "ts_c", "cd", "ex_c", "dnm", "dad"]
RAW = ["nm_r", "ad_r", "ts_r"]
FSETS = {"base": [], "raw": RAW, "corr": RAW + CORR, "conly": CORR, "delta": ["dnm", "dad", "cd", "ex_c"]}
PRIMARY = "conly"      # pre-declared gate set (raw sims alone hurt transfer, dry run 13:20)


def band_only(X: pd.DataFrame, region: np.ndarray) -> pd.DataFrame:
    """Corrector columns only on band rows (0.02 < g < 0.98), NaN elsewhere, on every frame (LOCO and France): the
    shipped cell vetoes band pairs only and ~all its adds are band pairs; band rows are decoded first. Pre-declared 13:32."""
    X = X.copy()
    X.loc[region != 1, [c for c in X.columns if c in RAW + CORR]] = np.nan
    return X


def norm(s: pd.Series) -> pd.Series:
    """casefold, punctuation / symbols -> space (Unicode classes: Indic and accented letters survive), squeeze."""
    s = pd.Series(s.to_numpy(), dtype="str").str.casefold().str.replace(r"[\p{P}\p{S}]+", " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def split(s: pd.Series):
    """'name | addr' -> (name, addr), at the LAST ' | ' (names like 'X | www.x.com' keep their bar)."""
    p = pd.Series(s.to_numpy(), dtype="str").str.rpartition(" | ")
    nm = p[0].where(p[1] != "", p[2])
    ad = p[2].where(p[1] != "", "")
    return norm(nm), norm(ad)


def sims(a_nm, a_ad, b_nm, b_ad, a_full, b_full):
    from rapidfuzz import fuzz, process
    cp = lambda x, y, sc: process.cpdist(x.to_numpy(), y.to_numpy(), scorer=sc, workers=4).astype(np.float32) / 100
    return cp(a_nm, b_nm, fuzz.ratio), cp(a_ad, b_ad, fuzz.ratio), cp(a_full, b_full, fuzz.token_set_ratio)


def fetch(job):
    d = W / "dec" / job
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", f"{BUCKET}/results/{job}/dec/", str(d), "--profile", "harsh", "--region",
                    "ap-south-1", "--only-show-errors"], check=True)
    print(job, sorted(p.name for p in d.glob("*.parquet")))


def feat_rows(sx_txt, s1_txt, out) -> pd.DataFrame:
    """RAW + CORR columns for aligned arrays (out: decode or None/NaN = not decoded)."""
    from rapidfuzz import fuzz, process
    s1n, s1a = split(pd.Series(s1_txt))
    sxn, sxa = split(pd.Series(sx_txt))
    s1f, sxf = (s1n + " " + s1a).str.strip(), (sxn + " " + sxa).str.strip()
    r = pd.DataFrame(index=np.arange(len(s1n)))
    r["nm_r"], r["ad_r"], r["ts_r"] = sims(sxn, sxa, s1n, s1a, sxf, s1f)
    for c in CORR:
        r[c] = np.float32(np.nan)
    ok = pd.notna(out)
    if ok.any():
        on, oa = split(pd.Series(np.asarray(out, dtype=object)[ok]))
        of = (on + " " + oa).str.strip()
        i = np.flatnonzero(ok)
        take = lambda s: s.iloc[i].reset_index(drop=True)
        a, b, c = sims(on, oa, take(s1n), take(s1a), of, take(s1f))
        cd = 1 - process.cpdist(take(sxf).to_numpy(), of.to_numpy(), scorer=fuzz.ratio, workers=4).astype(np.float32) / 100
        r.loc[ok, "nm_c"], r.loc[ok, "ad_c"], r.loc[ok, "ts_c"], r.loc[ok, "cd"] = a, b, c, cd
        r.loc[ok, "ex_c"] = (on.to_numpy() == take(s1n).to_numpy()).astype(np.float32)
        r["dnm"], r["dad"] = r.nm_c - r.nm_r, r.ad_c - r.ad_r
    return r.astype(np.float32)


def feats(job, frame, chunk=250_000):
    t0 = time.time()
    fetch(job)
    f = pd.read_parquet(W / f"sm/frame_{frame}.parquet", columns=["row", "sx_txt", "s1_txt"])
    assert (f.row.to_numpy() == np.arange(len(f))).all()
    dj = W / "dec" / job                                     # main job's chunks + decode-only jobs' band chunks (B)
    parts = [pd.read_parquet(p) for p in sorted(dj.glob(f"dec_frame_{frame}_[0-9]*.parquet")) + sorted(dj.glob(f"dec_frame_{frame}B_[0-9]*.parquet"))]
    out = np.full(len(f), None, dtype=object)
    if parts:
        dd = pd.concat(parts, ignore_index=True)
        out[dd.row.to_numpy()] = dd.out.to_numpy(dtype=object)
        del dd, parts
    rs = [feat_rows(f.sx_txt.to_numpy()[s:s + chunk], f.s1_txt.to_numpy()[s:s + chunk], out[s:s + chunk])
          for s in range(0, len(f), chunk)]
    r = pd.concat(rs, ignore_index=True)
    r.insert(0, "row", np.arange(len(r), dtype=np.int32))
    r.to_parquet(W / f"f_{frame}_{job}.parquet", index=False)
    ok = r.nm_c.notna().to_numpy()
    print(f"feats {frame} {job}: {ok.sum():,}/{len(ok):,} decoded; {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    return r


def band_auc(y, g, X):
    import p55
    m = (g > 0.02) & (g < 0.98)
    return {c: p55.auc(y[m & ~np.isnan(X[c])], X[c][m & ~np.isnan(X[c])]) for c in X.columns}


def loco(job_us, job_in, tag=""):
    sys.path[:0] = [str(ROOT / "probes/P55-fr-combiner")]
    import p55
    from ledger import log
    t0 = time.time()
    frames = {dr: p55.loco_frame(dr) for dr in p55.L.DIRS}
    print(f"frames {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    JOB = {"India->US": job_in, "US->India": job_us}           # the corrector trained on the direction's source
    out = {}
    import os
    dirn = [d for d in p55.DIRN if d[0] in os.environ.get("P65_DIRS", "India->US,US->India").split(",")]
    for name, trn, tst in dirn:
        a, b = frames[trn], frames[tst]
        # P65_OOD=1: the fit frame's columns come from the OTHER corrector (out of domain on the fit side too, the France
        # build's situation); it reads the target country's corrector (trained on its non-holdout labels) -> not strict LOCO
        jf = {"India->US": job_us, "US->India": job_in}[name] if os.environ.get("P65_OOD") else JOB[name]
        fa = band_only(pd.read_parquet(W / f"f_{trn}_{jf}.parquet"), a["X"]["region"].to_numpy())
        fb = band_only(pd.read_parquet(W / f"f_{tst}_{JOB[name]}.parquet"), b["X"]["region"].to_numpy())
        assert len(fa) == len(a["X"]) and len(fb) == len(b["X"])
        R = out[name] = {"decoded_tgt_band": float(fb.nm_c.notna()[b["X"]["region"].to_numpy() == 1].mean()),
                         "decoded_src_band": float(fa.nm_c.notna()[a["X"]["region"].to_numpy() == 1].mean()),
                         "band_auc_tgt": band_auc(b["z"]["y"], b["z"]["g"], fb[RAW + CORR])}
        fref = p55.L.F(b["z"], b["ref"])
        rg = b["X"]["region"].to_numpy()
        C = {}
        for fs, cols in FSETS.items():
            Xa = pd.concat([a["X"], fa[cols]], axis=1) if cols else a["X"]
            Xb = pd.concat([b["X"], fb[cols]], axis=1) if cols else b["X"]
            bs = p55.bag(Xa, a["z"]["y"], None)
            cb = C[fs] = p55.bag_score(bs, Xb, None)
            S = R[fs] = {"auc_ref": p55.auc(b["z"]["y"][b["ref"]], cb[b["ref"]]),
                         "auc_band": p55.auc(b["z"]["y"][rg == 1], cb[rg == 1])}
            veto = lambda v: b["ref"] & ~(b["ref"] & (cb < v) & (rg == 1))
            S["ship"] = p55.L.cmp(b["z"], b["ref"], p55.decide({**b, "ref": veto(0.6)}, cb, None, 0.95), fref)
            S["veto06"] = p55.L.cmp(b["z"], b["ref"], veto(0.6), fref)
            for u in p55.U2:
                S[f"add:{u}"] = p55.L.cmp(b["z"], b["ref"], p55.decide(b, cb, None, u), fref)
            for v in p55.V2:
                S[f"B:band:{v}"] = p55.L.cmp(b["z"], b["ref"], veto(v), fref)
            imp = pd.Series(bs[0].feature_importance("gain"), index=Xa.columns)
            S["imp_new"] = {k: round(float(imp[k] / imp.sum()), 4) for k in cols}
            print(name, fs, {k: S[k] for k in ("auc_ref", "auc_band", "ship", "veto06")}, f"{time.time() - t0:.0f}s", flush=True)
        hv = b["z"]["hv"]
        for k in ("ship", "veto06"):                            # paired set - base at the shipped cells
            fb0 = p55.L.F(b["z"], _new(p55, b, C["base"], k))
            for fs in FSETS:
                if fs == "base":
                    continue
                d_ = p55.L.F(b["z"], _new(p55, b, C[fs], k)) - fb0
                R[f"{fs}-base_{k}"] = {"dF": round(float(d_.mean()), 6), "se": round(float(d_.std() / np.sqrt(len(d_))), 6),
                                       "dF_h0": round(float(d_[hv == 0].mean()), 6), "dF_h1": round(float(d_[hv == 1].mean()), 6)}
                print(name, k, f"{fs}-base", R[f"{fs}-base_{k}"], flush=True)
        R["rule"] = rule_arm(p55, a, b, fa, fb, C["base"])
        print(name, "rule", R["rule"], flush=True)
    g = {n: out[n][f"{PRIMARY}-base_ship"]["dF"] for n in out}
    ok = g.get("India->US", -1) >= 0.0003 and g.get("US->India", -1) >= 0
    out["gate"] = {"rule": f"{PRIMARY}-base at the shipped cell: India->US >= +0.0003 and US->India >= 0", "pass": bool(ok)}
    out["runtime_s"], out["peak_gb"] = round(time.time() - t0), round(peak(), 2)
    (HERE / f"loco65{tag}.json").write_text(json.dumps(out, indent=1))
    log("exp", probe="P65-byt5-corrector", subproblem="france-combiner", name=f"p65-loco-combiner{tag}", world="holdout-loco",
        metrics={f"{n}|{k}": v for n in g for k, v in out[n].items()
                 if "-base_" in k or k.startswith("decoded")} |
                {f"{n}|{fs}|{c}": out[n][fs][c]["dF"] for n in g for fs in FSETS for c in ("ship", "veto06")} |
                {f"{n}|rule": out[n]["rule"]["tgt"] for n in g},
        config={"jobs": JOB, "model": "google/byt5-small", "fsets": FSETS, "primary": PRIMARY, "combiner": "P55 bag 3 seeds, 200 rounds",
                "cells": "ship = veto band c<.6 + add free c>.95"},
        hypothesis="ByT5 corrector sims (corrected SX vs S1) add France-analogue F0.5 over P55's combiner",
        verdict="keep" if ok else "drop", runtime_s=out["runtime_s"], peak_gb=out["peak_gb"])
    print("GATE", "PASS" if ok else "FAIL", g, f"{time.time() - t0:.0f}s peak {peak():.2f} GB")


def rule_arm(p55, a, b, fa, fb, cb0):
    """No learning on strings: on top of P55's shipped cell (base combiner), veto band ref pairs whose corrected name AND
    address both disagree with S1 (nm_c < v and ad_c < v), add free band pairs whose corrected name matches S1 exactly
    and address ratio >= u. (v, u) picked on the source frame, applied to the target."""
    def run(fr, f, c0, v, u):
        z, rg = fr["z"], fr["X"]["region"].to_numpy()
        ship = _new(p55, fr, c0, "ship")
        nm, ad = f.nm_c.to_numpy(), f.ad_c.to_numpy()
        new = ship.copy()
        if v is not None:
            new &= ~(ship & (rg == 1) & (nm < v) & (ad < v))
        if u is not None:
            s1, is2 = z["s1"].astype(np.int64), z["is2"]
            n2, n3 = p55.S.counts(s1, is2, new, fr["n"])
            cand = np.flatnonzero(fr["free"] & ~new & (rg == 1) & (f.ex_c.to_numpy() == 1) & (ad >= u))
            o, rk = p55.S.rank_in(s1 * 2 + is2, np.nan_to_num(ad), cand)
            new[o[rk < np.where(is2[o], p55.CUR["cap"][0] - n2[s1[o]], p55.CUR["cap"][1] - n3[s1[o]])]] = True
        return p55.L.cmp(z, ship, new, p55.L.F(z, ship))
    ca = p55.cross_fit(a, "plain")                     # the source's ship decision, 5-fold out of fold (P55's pick)
    grid = [(v, u) for v in (None, 0.3, 0.4, 0.5) for u in (None, 0.5, 0.7, 0.9) if (v, u) != (None, None)]
    src = {f"{v}:{u}": run(a, fa, ca, v, u) for v, u in grid}
    k = max(src, key=lambda k: src[k]["dF"])
    v, u = (None if x == "None" else float(x) for x in k.split(":"))
    return {"cell": k, "src": src[k], "tgt": run(b, fb, cb0, v, u),
            "tgt_all": {kk: run(b, fb, cb0, *(None if x == "None" else float(x) for x in kk.split(":")))["dF"] for kk in src}}


def _new(p55, b, cb, k):
    rg = b["X"]["region"].to_numpy()
    v = b["ref"] & ~(b["ref"] & (cb < 0.6) & (rg == 1))
    return p55.decide({**b, "ref": v}, cb, None, 0.95) if k == "ship" else v


def frscore(job_pool, job_us, job_in, fs=None):
    """France combiner scores: P55's pooled bag (3 seeds) on both LOCO frames, P55's 20 features + FSETS[fs] corrector
    columns (each frame's columns from the OTHER country's corrector: out of domain, as on France), applied to P55's
    France frame with the pooled corrector's columns -> data/work/p65/cf_<fs>.npy, linked as data/work/p55/cf_p65<fs>.npy
    so `p55.py build <base> <out> mode=p65<fs>,v=..,scope=band,u=..` uses it."""
    sys.path[:0] = [str(ROOT / "probes/P55-fr-combiner")]
    import p55
    t0 = time.time()
    fs = fs or PRIMARY
    cols = FSETS[fs]
    job = {"US_India": job_us, "India_US": job_in}          # frame US_India = India pairs <- US-trained corrector
    Xs, ys = [], []
    for dr in p55.L.DIRS:
        fr = p55.loco_frame(dr)
        f = band_only(pd.read_parquet(W / f"f_{dr}_{job[dr]}.parquet"), fr["X"]["region"].to_numpy())
        Xs.append(pd.concat([fr["X"], f[cols]], axis=1))
        ys.append(fr["z"]["y"])
        del fr
    bs = p55.bag(pd.concat(Xs, ignore_index=True), np.concatenate(ys), None)
    del Xs, ys
    f = p55.france_frame()
    ff = band_only(pd.read_parquet(W / f"f_France_{job_pool}.parquet"), f["X"]["region"].to_numpy())
    assert len(ff) == len(f["X"])
    cf = p55.bag_score(bs, pd.concat([f["X"], ff[cols]], axis=1), None)
    np.save(W / f"cf_{fs}.npy", cf)
    ln = ROOT / f"data/work/p55/cf_p65{fs}.npy"
    ln.unlink(missing_ok=True)
    ln.symlink_to(W / f"cf_{fs}.npy")
    c0 = np.load(ROOT / "data/work/p55/cf_plain.npy")
    rg = f["X"]["region"].to_numpy()
    print(f"frscore {fs}: France {len(cf):,} pairs, decoded {ff.nm_c.notna().mean():.3f}; band veto<.6 p65 {int((f['ref'] & (cf < .6) & (rg == 1)).sum())} "
          f"vs P55 {int((f['ref'] & (c0 < .6) & (rg == 1)).sum())}; add>.95 cand p65 {int((f['free'] & ~f['ref'] & (cf > .95)).sum())} "
          f"vs P55 {int((f['free'] & ~f['ref'] & (c0 > .95)).sum())}; {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)


def _selfcheck():
    n, a = split(pd.Series(["Café Été | 12 Rue X, Paris", "रमेश स्टोर | ", "A | www.a.com | 1 Main St", "noaddr"]))
    assert n.tolist() == ["café été", "रमेश स्टोर", "a www a com", "noaddr"], n.tolist()
    assert a.tolist() == ["12 rue x paris", "", "1 main st", ""], a.tolist()
    print("selfcheck ok")


if __name__ == "__main__":
    cmd, pos = sys.argv[1], sys.argv[2:]
    {"fetch": lambda: fetch(*pos), "feats": lambda: feats(*pos), "loco": lambda: loco(*pos), "frscore": lambda: frscore(*pos), "self": _selfcheck}[cmd]()
