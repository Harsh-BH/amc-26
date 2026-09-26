"""P45 stage B (matcher): a diverse booster zoo stacked into P34's a10 residual stacker, on the #34 yardstick.

Zoo members = full-data matchers trained on train folds 2-5 or 6-9 (never 0 = holdout, never 1 = early stop), scored on
#6's pruned holdout pairs (the rows of data/work/p23/pairs_<c>.parquet, same order, asserted):
  P22 xgb, cat · P25 bag1..4, goss, deep, shallow, extra · P45 dart, rf, et, rank_sx, rank_s1 (whatever is fetched).
    zoo.py oof          # pred/<m>_holdout_<c>.parquet -> oof/holdout_<c>.parquet (+ README) for P44
    zoo.py corr         # pairwise error correlation + logloss of the members (band pairs of scored S1)
    zoo.py extra <ce_job|-> <e5_job|->   # full-coverage bge CE (p_r_all) + e5 cosines -> the OOF table
    zoo.py eval         # arms vs #34 (a10 guard, t .71), fit h0 -> report h1 and the swap, paired per-S1 SE
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P34-a10-stack"), str(ROOT)]
import a10 as A
import ho as H

W = ROOT / "data/work/p45"
PRED, OOF = W / "pred", W / "oof"
C = ("India", "US")
class _Rankers:                                 # raw lambdarank scores (rank_*), not probabilities
    __contains__ = staticmethod(lambda m: str(m).startswith("rank"))


RANKERS = _Rankers()
lg = A.lg
DIVERSE = {"bag3", "rank_s1", "rank_sx2", "rf2", "et2", "dart", "noctx", "fsub", "mlp", "mlp2", "r_all", "e5"}


def members() -> list:
    """Members already merged into the OOF table (the eval / corr source)."""
    f = OOF / "holdout_India.parquet"
    return sorted(x[2:] for x in pq.read_schema(f).names if x.startswith("p_") and x != "p_m2") if f.exists() else []


def pending() -> list:
    ms = sorted({f.name.split("_holdout_")[0] for f in PRED.glob("*_holdout_*.parquet")})
    return [m for m in ms if all((PRED / f"{m}_holdout_{c}.parquet").exists() for c in C)]


def oof():
    OOF.mkdir(exist_ok=True)
    ms = pending()
    for c in C:                                         # incremental: merged pred files are deleted (disk ~1 GB)
        f = OOF / f"holdout_{c}.parquet"
        if f.exists():
            pp = pd.read_parquet(f)
        else:
            pp = pd.read_parquet(ROOT / f"data/work/p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p_m2", "r", "p", "am"])
            pp = pp.rename(columns={"p": "p8c"})
        k = pq.read_table(ROOT / f"data/work/p9/m2_holdout_{c}_d2b6569f_6a23c2ec_0.03.parquet", columns=["q1", "qx"])
        q1, qx = k.column("q1").to_numpy(), k.column("qx").to_numpy()
        for m in ms:
            t = pq.read_table(PRED / f"{m}_holdout_{c}.parquet")
            assert (t.column("q1").to_numpy() == q1).all() and (t.column("qx").to_numpy() == qx).all(), (m, c)
            pp[f"p_{m}"] = t.column("p").to_numpy().astype(np.float32)
        pp.to_parquet(f, index=False)
        print(c, len(pp), [x[2:] for x in pp.columns if x.startswith("p_") and x != "p_m2"], flush=True)
    for m in ms:
        for c in C:
            (PRED / f"{m}_holdout_{c}.parquet").unlink()
    ms = sorted(x[2:] for x in pq.read_schema(OOF / "holdout_India.parquet").names if x.startswith("p_") and x != "p_m2")
    (OOF / "README.md").write_text(
        "# P45 zoo out-of-fold holdout predictions (for P44)\n\n"
        "`holdout_{India,US}.parquet`: one row per #6-pruned holdout pair = the rows (and order) of "
        "`data/work/p23/pairs_<c>.parquet` = `data/work/p9/m2_holdout_<c>_d2b6569f_6a23c2ec_0.03.parquet`.\n\n"
        "| column | meaning |\n|---|---|\n"
        "| s1, sx | entity ids |\n| holdout | True = the S1 is a scored (fold-0) S1; False = a competing train S1 |\n"
        "| y | pair is a true match |\n| p_m2 | m2_big p (the shipped matcher) |\n| r | bge CE logit (band only, else NaN) |\n"
        "| p8c | #8c's final p (m2 + bge blend + US P17 stack) |\n| am | #8c argmax S1 of the SX |\n"
        + "".join(f"| p_{m} | {'raw lambdarank score (NOT a probability)' if m in RANKERS else 'probability'} of zoo member `{m}` |\n" for m in ms)
        + "\nEvery member was trained on train folds 2-5 (bag3/bag4: 6-9) with early stopping on fold 1, never on fold 0, "
        "so every value here is out of fold. Test predictions (same q1/qx keys as `data/work/p9/m2_test_<c>_...0.03`): "
        "`s3://sagemaker-114872709824-ap-south-1/amc26-p25/results/p25-<m>-0926-1404/pred/test_<c>.parquet` (P25 members), "
        "`amc26-p22/results/p22-{xgb,cat}-0926-1300/pred/`, `amc26-p45/results/p45-<m>-*/pred/` (P45 members).\n")


def extra(ce_job: str = "", e5_job: str = ""):
    """Full-coverage bge CE (P13 bge_ckpt on every holdout pair -> r_bge_poolck_holdout) as p_r_all, and the e5-small
    cosines (e_name, e_addr, e_full) -> merged into the OOF table by (s1, sx)."""
    import subprocess
    B = "s3://sagemaker-114872709824-ap-south-1/amc26-p45/results"
    get = lambda src, dst: subprocess.run(["aws", "s3", "cp", "--only-show-errors", "--profile", "harsh", src, str(dst)], check=True)
    add = {}
    if ce_job:
        get(f"{B}/{ce_job}/r/r_bge_poolck_holdout.parquet", W / "r_all.parquet")
        add["ce"] = pd.read_parquet(W / "r_all.parquet").rename(columns={"r": "p_r_all"})
    for c in C:
        f = OOF / f"holdout_{c}.parquet"
        pp = pd.read_parquet(f)
        if ce_job:
            pp = pp.drop(columns=["p_r_all"], errors="ignore").merge(add["ce"], on=["s1", "sx"], how="left", validate="1:1")
            band = pp.r.notna().to_numpy()
            agree = np.corrcoef(lg(pp.p_r_all.to_numpy()[band]), lg(pp.r.to_numpy()[band]))[0, 1]
            print(f"{c}: r_all coverage {pp.p_r_all.notna().mean():.4f}; logit corr with the shipped band r {agree:.5f}", flush=True)
        if e5_job:
            get(f"{B}/{e5_job}/holdout_e5_{c}.parquet", W / f"e5_{c}.parquet")
            e = pd.read_parquet(W / f"e5_{c}.parquet")
            pp = pp.drop(columns=["e_name", "e_addr", "e_full"], errors="ignore").merge(e, on=["s1", "sx"], how="left", validate="1:1")
            (W / f"e5_{c}.parquet").unlink()
            print(f"{c}: e5 coverage {pp.e_full.notna().mean():.4f}", flush=True)
        assert len(pp) == len(pd.read_parquet(f, columns=["s1"]))
        pp.to_parquet(f, index=False)
    (W / "r_all.parquet").unlink(missing_ok=True)
    with open(OOF / "README.md", "a") as fh:
        if ce_job:
            fh.write(f"\n`p_r_all`: our fine-tuned bge CE (ce_bge_pool.pt, P13 bge_ckpt path) on EVERY holdout pair "
                     f"(sigmoid of its logit; job {ce_job}). On band pairs it equals `r` up to fp16 rounding.\n")
        if e5_job:
            fh.write(f"\n`e_name`, `e_addr`, `e_full`: cosine of frozen multilingual-e5-small (MIT, 118M; 'query: ' prefix, mean pool) "
                     f"embeddings of S1 vs SX name / address / 'name | addr'; e_addr NaN if either address is empty/NULL (job {e5_job}).\n")


def corr():
    ms, out = members(), {}
    for c in C:
        o = pd.read_parquet(OOF / f"holdout_{c}.parquet", columns=["holdout", "y", "p_m2"] + [f"p_{m}" for m in ms])
        o = o[o.holdout]
        band = (o.p_m2 > 0.02) & (o.p_m2 < 0.98)
        cols = ["m2"] + [m for m in ms if m not in RANKERS]
        from sklearn.linear_model import LogisticRegression
        bb, yb = band.to_numpy(), o.y.to_numpy()

        def cal(m):                                     # Platt on the band (diagnostic only): error corr of calibrated p
            z = o[f"p_{m}"].to_numpy()[bb] if m in RANKERS else lg(o[f"p_{m}"].to_numpy()[bb])
            return yb[bb] - LogisticRegression().fit(z[:, None], yb[bb]).predict_proba(z[:, None])[:, 1]
        E = pd.DataFrame({m: cal(m) for m in ["m2"] + ms})
        ll = {m: float(-np.mean(np.where(o.y, np.log(np.clip(o[f"p_{m}"], 1e-7, 1)), np.log(np.clip(1 - o[f"p_{m}"], 1e-7, 1)))))
              for m in cols}
        L = pd.DataFrame({m: lg(o[f"p_{m}"].to_numpy()) if m not in RANKERS else o[f"p_{m}"].to_numpy()
                          for m in ["m2"] + ms})[band.to_numpy()]
        out[c] = dict(err_corr_band_platt=E.corr().round(3).to_dict(), logit_corr_band=L.corr(method="spearman").round(3).to_dict(),
                      logloss_scored=ll, n_band=int(band.sum()))
        print(f"{c}: {band.sum():,} band pairs; logloss " + " ".join(f"{m} {v:.5f}" for m, v in ll.items()), flush=True)
        print("band error corr (Platt-calibrated per member):\n" + E.corr().round(3).to_string(), flush=True)
    json.dump(out, open(W / "zoo_corr.json", "w"), indent=1)


def corr2():
    """Platt-calibrated error correlation of every member (and e_full) with m2_big and with the shipped a10 p' (#34, h0
    fit, guard), per region: band / out of band (scored-S1 pairs), and #8c argmax pairs of scored S1 (where a10 lives)."""
    from sklearn.linear_model import LogisticRegression
    ms, out = members(), {}
    for c in C:
        o = pd.read_parquet(OOF / f"holdout_{c}.parquet")
        o = o[o.holdout].reset_index(drop=True)
        ho = pd.read_parquet(ROOT / f"data/work/p34/ho_{c}.parquet", columns=["s1", "sx"])
        ho["p_a10"] = np.load(ROOT / f"data/work/p34/ho_p_{c}_h0.npy")
        o = o.merge(ho, on=["s1", "sx"], how="left")
        score = {m: (o[f"p_{m}"].to_numpy() if m in RANKERS else lg(o[f"p_{m}"].to_numpy())) for m in ["m2"] + ms}
        score |= {e: o[e].to_numpy() for e in ("e_name", "e_addr", "e_full") if e in o}
        score["a10"] = lg(o.p_a10.to_numpy())
        y = o.y.to_numpy()
        band = ((o.p_m2 > 0.02) & (o.p_m2 < 0.98)).to_numpy()
        regions = {"band": band, "out_of_band": ~band, "argmax": o.p_a10.notna().to_numpy()}
        out[c] = {}
        for rn, rm in regions.items():
            def err(k):
                z, m = score[k][rm], ~np.isnan(score[k][rm])
                e = np.full(rm.sum(), np.nan)
                e[m] = y[rm][m] - LogisticRegression().fit(z[m, None], y[rm][m]).predict_proba(z[m, None])[:, 1]
                return e
            keys = [k for k in score if k != "a10" or rn == "argmax"]
            E = pd.DataFrame({k: err(k) for k in keys})
            cm = E.corr()
            out[c][rn] = {"n": int(rm.sum()), "vs_m2": cm["m2"].round(3).to_dict(), **({"vs_a10": cm["a10"].round(3).to_dict()} if "a10" in cm else {})}
            print(f"{c} {rn} (n={rm.sum():,}) vs m2: " + " ".join(f"{k} {v:.3f}" for k, v in cm["m2"].items() if k != "m2"), flush=True)
            if "a10" in cm:
                print(f"{c} {rn} vs a10: " + " ".join(f"{k} {v:.3f}" for k, v in cm["a10"].items() if k != "a10"), flush=True)
    json.dump(out, open(W / "zoo_corr2.json", "w"), indent=1)


# ---------------------------------------------------------------------------------------------------------------- eval
def load(ms: list) -> list:
    S = []
    for c in C:
        s = H.load(c)                                                        # asserts the #8c yardstick
        ecols = [x for x in ("e_name", "e_addr", "e_full") if x in pq.read_schema(OOF / f"holdout_{c}.parquet").names]
        o = pd.read_parquet(OOF / f"holdout_{c}.parquet", columns=["s1", "sx"] + [f"p_{m}" for m in ms] + ecols)
        a = s["a"].merge(o, on=["s1", "sx"], how="left", validate="1:1")
        assert len(a) == len(s["a"]) and not a[[f"p_{m}" for m in ms]].isna().any().any()
        for m in ms:
            a[f"l_{m}"] = a[f"p_{m}"].to_numpy() if m in RANKERS else lg(a[f"p_{m}"].to_numpy())
        Z = a[[f"l_{m}" for m in ms if m not in RANKERS]].to_numpy()
        a["z_mean"], a["z_std"], a["z_min"], a["z_max"] = Z.mean(1), Z.std(1), Z.min(1), Z.max(1)
        a["z_dm2"] = a.z_mean - a.lm2
        # P44's two label-free group columns (ctx.add3), from #8c p over every row of the holdout world
        d = pd.read_parquet(ROOT / f"data/work/p23/pairs_{c}.parquet", columns=["s1", "sx", "p"])
        kx = pd.factorize(d.sx)[0]
        a = a.merge(pd.DataFrame({"s1": d.s1, "sx": d.sx, "sx_psum": np.bincount(kx, d.p.to_numpy())[kx]}),
                    on=["s1", "sx"], how="left", validate="1:1")
        del d
        g, acc, s3 = a.g.to_numpy(), (a.p.to_numpy() >= A.T).astype(int), a.sx.str.startswith("S3-").to_numpy()
        a["n_same_src_acc"] = 0.0
        for v in (False, True):
            m = s3 == v
            a.loc[m, "n_same_src_acc"] = np.bincount(g[m], acc[m], minlength=g.max() + 1)[g[m]] - acc[m]
        s["a"] = a
        S.append(s)
    return S


def run_arm(S: list, extra: list, fh: int, init=None) -> list:
    """a10 with extra columns, fit on half fh (India+US pooled, as #34), guarded p' on every argmax pair."""
    base = list(A.COLS)
    try:
        A.COLS = base + [e for e in extra if e not in base]
        tr = [s["h"][s["a"].g] == fh for s in S]
        X = pd.concat([s["a"][m] for s, m in zip(S, tr)])
        if init is not None:
            X = X.assign(lp=X[init])
        b = A.fit(X, np.concatenate([s["a"].y.to_numpy()[m] for s, m in zip(S, tr)]))
        return [A.stacked(b, s["a"] if init is None else s["a"].assign(lp=s["a"][init]), True) for s in S], b
    finally:
        A.COLS = base


def paired(S: list, ps: list, ref: list, fh: int, t: float = A.T, tref: float = A.T) -> dict:
    rep, out = 1 - fh, {}
    for s, p, q in zip(S, ps, ref):
        d = (H.f_of(s, p >= t) - H.f_of(s, q >= tref))[s["h"] == rep]
        out[s["c"]] = dict(d=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), up=int((d > 0).sum()), down=int((d < 0).sum()))
    out["lb"] = dict(d=sum(H.LB_W[c] * out[c]["d"] for c in C), se=float(np.sqrt(sum((H.LB_W[c] * out[c]["se"]) ** 2 for c in C))))
    return out


def lr_meta(S: list, cols: list, fh: int) -> list:
    """Logistic meta-learner on logits (argmax pairs of the fit half, India+US pooled) -> p on every argmax pair."""
    from sklearn.linear_model import LogisticRegression
    tr = [s["h"][s["a"].g] == fh for s in S]
    X = np.vstack([s["a"][cols].to_numpy()[m] for s, m in zip(S, tr)])
    y = np.concatenate([s["a"].y.to_numpy()[m] for s, m in zip(S, tr)])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    lr = LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, y)
    ps = [lr.predict_proba((s["a"][cols].to_numpy() - mu) / sd)[:, 1] for s in S]
    return [np.where(s["a"].hc.to_numpy() == 1, s["a"].p.to_numpy(), p) for s, p in zip(S, ps)], dict(zip(cols, (lr.coef_[0] / sd).round(3)))


def tune_t(S: list, ps: list, fh: int) -> float:
    grid = np.round(np.arange(0.55, 0.90, 0.01), 2)
    sc = [sum(H.f_of(s, p >= t)[s["h"] == fh].sum() for s, p in zip(S, ps)) for t in grid]
    return float(grid[int(np.argmax(sc))])


def evaluate(only: str = ""):
    from ledger import log
    ms = members()
    zl = [f"l_{m}" for m in ms]
    S = load(ms)
    arms = {
        "z_all": zl,                                                    # every member's logit as a column
        "z_stats": ["z_mean", "z_std", "z_min", "z_max", "z_dm2"],      # disagreement summary
        "z_all_stats": zl + ["z_mean", "z_std", "z_min", "z_max", "z_dm2"],
    }
    arms["z_div"] = [f"l_{m}" for m in ms if m in DIVERSE]              # members with calibrated band error corr < 0.96 vs m2
    if "e_full" in S[0]["a"]:
        arms["z_e5"] = ["e_name", "e_addr", "e_full"]
        arms["z_div_e5"] = arms["z_div"] + arms["z_e5"]
    nn = [f"l_{m}" for m in ("mlp", "mlp2", "r_all") if m in ms]
    if nn:
        arms["z_nn"] = nn + (arms.get("z_e5") or [])                   # neural / raw-text members only
    for m in ms:
        arms[f"z_{m}"] = [f"l_{m}"]                                     # one member at a time
    if only:
        arms = {k: v for k, v in arms.items() if k in only.split(",")}
    import os
    base = [x for x in os.environ.get("P45_BASE", "").split(",") if x]   # e.g. P44's sx_psum,n_same_src_acc
    arms = {k: base + v for k, v in arms.items()}
    res = {}
    for fh in (0, 1):
        t0 = time.time()
        ref, _ = run_arm(S, base, fh)                                   # = #34 (guard) [+ base columns], refit here
        if base:
            r34, _ = run_arm(S, [], fh)
            o = paired(S, ref, r34, fh)
            print(f"fit h{fh}: base {base} vs #34 LB-eq {o['lb']['d']:+.6f} ± {o['lb']['se']:.6f}", flush=True)
            res[f"h{fh}/base_vs_34"] = o
        base8c = [s["a"].p.to_numpy() for s in S]
        chk = paired(S, ref, base8c, fh)
        print(f"fit h{fh}: #34 vs #8c LB-eq {chk['lb']['d']:+.6f} ± {chk['lb']['se']:.6f} (P34: h0 +0.000209, h1 +0.000175)", flush=True)
        res[f"h{fh}/#34_vs_8c"] = chk
        todo = dict(arms)
        for name, cols in todo.items():
            ps, b = run_arm(S, cols, fh)
            o = paired(S, ps, ref, fh)
            tt = tune_t(S, ps, fh)
            o["t_tuned"] = tt
            o["tuned"] = paired(S, ps, ref, fh, t=tt)
            res[f"h{fh}/{name}"] = o
            print(f"  h{fh} {name:14s} vs #34: LB-eq {o['lb']['d']:+.6f} ± {o['lb']['se']:.6f} ({o['lb']['d'] / o['lb']['se']:+.1f} SE) | "
                  f"India {o['India']['d']:+.6f} ± {o['India']['se']:.6f} | US {o['US']['d']:+.6f} ± {o['US']['se']:.6f} | "
                  f"t*={tt:.2f}: {o['tuned']['lb']['d']:+.6f}", flush=True)
        if not only or "lr" in only:
            for name, cols in {"lr_zoo": ["lp", "lm2"] + [z for z in zl if z[2:] not in RANKERS]}.items():
                ps, coef = lr_meta(S, cols, fh)
                tt = tune_t(S, ps, fh)
                o = paired(S, ps, ref, fh, t=tt)
                o["vs8c"] = paired(S, ps, base8c, fh, t=tt)
                o["t_tuned"], o["coef"] = tt, coef
                res[f"h{fh}/{name}"] = o
                print(f"  h{fh} {name:14s} (t*={tt:.2f}) vs #34: LB-eq {o['lb']['d']:+.6f} ± {o['lb']['se']:.6f} | vs #8c "
                      f"{o['vs8c']['lb']['d']:+.6f} ± {o['vs8c']['lb']['se']:.6f} | coef {coef}", flush=True)
                # two-stage: the LR meta p as a10's init (context columns on top of the combined booster)
                for s, p in zip(S, ps):
                    s["a"]["l_lr"] = lg(p)
                ps2, _ = run_arm(S, [], fh, init="l_lr")
                o2 = paired(S, ps2, ref, fh)
                res[f"h{fh}/{name}+a10"] = o2
                print(f"  h{fh} {name + '+a10':14s} vs #34: LB-eq {o2['lb']['d']:+.6f} ± {o2['lb']['se']:.6f} | India "
                      f"{o2['India']['d']:+.6f} | US {o2['US']['d']:+.6f}", flush=True)
        print(f"  fit h{fh} done {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(W / f"zoo_eval{'_base' if base else ''}{'_' + only.replace(',', '+') if only else ''}.json", "w"), indent=1, default=float)
    for name in [k.split("/", 1)[1] for k in res if k.startswith("h0/") and not k.endswith("_vs_8c")]:
        o0, o1 = res[f"h0/{name}"], res.get(f"h1/{name}", {})
        ok = (o0["lb"]["d"] >= 0.00015 and o0["lb"]["d"] >= 3 * o0["lb"]["se"] and min(o0[c]["d"] for c in C) >= -0.00005
              and o1.get("lb", {}).get("d", -1) > 0)
        log("exp", probe="P45-ens-stages", subproblem="matcher-zoo-stack", name=f"B-{name}-vs{'34+p44ctx' if base else '34'}", world="holdout",
            metrics={"lb_eq_h0h1": round(o0["lb"]["d"], 6), "se_h0h1": round(o0["lb"]["se"], 6),
                     "lb_eq_h1h0": round(o1.get("lb", {}).get("d", np.nan), 6), "se_h1h0": round(o1.get("lb", {}).get("se", np.nan), 6),
                     **{f"{c}_h0h1": round(o0[c]["d"], 6) for c in C}, **{f"{c}_se_h0h1": round(o0[c]["se"], 6) for c in C}},
            config={"members": ms, "arm": name, "base": "#34 a10 guard refit on the same half, t 0.71"},
            hypothesis="stacking diverse boosters into the a10 stacker adds >= +0.00015 LB-eq at >= 3 SE over #34",
            verdict="keep" if ok else "drop", runtime_s=None, peak_gb=None)


if __name__ == "__main__":
    {"oof": oof, "corr": corr, "corr2": corr2, "eval": lambda: evaluate(sys.argv[2] if len(sys.argv) > 2 else ""),
     "extra": lambda: extra(*[x if x != "-" else "" for x in sys.argv[2:4]])}[sys.argv[1]]()
