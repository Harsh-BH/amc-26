"""P29-fr-tiers: France rebuild (IDF world of #9ab + '(520) RUE' house numbers + R1 fillers + bge top-up) and the
label-free "key tier" / per-S1 cap post-rules.

    p29.py rf                  # data/work/p29/w_fix: P14's IDF world with rf_test_sx_France's hn re-parsed ('(n)' opener)
    ER_WORK=data/work/p29/w_fix p29.py rescore   # P20 rescore.py 'noise' (R1) there -> p29/prob_test_France_fix_noise
    p29.py build | launch <args> | fetch <job> | check | blend   # P19's bge top-up on the new band pairs
    p29.py holdout             # key tier + cap on #8c's dense US/India holdout (P23 dump): precision, count, dF0.5
    p29.py profile FILE        # France t curve (empty %, matches / S1)
    p29.py tiers FILE T OUT    # key tier (p >= FLOOR) + cap on a France file at t -> OUT (s1, sx, p) for combine.py
    p29.py diff A:t B:t        # France churn + hn lens between two decisions
    p29.py selftest
"""
import json, os, resource, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns")]
import normalize as N
import fr20
from ledger import log

WORK = ROOT / "data/work"
OUT = WORK / "p29"
W14 = WORK / "p14/w_idfx"                 # P14's IDF world (idf144) with X kept, as P20 used it
WFIX = OUT / "w_fix"
PROBE = "P29-fr-tiers"
PAREN = r"(^|,\s*)\((\d+)\)"              # P6 france._paren_hn: '(520) RUE X' -> '520 RUE X'
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def paren_hn(addr: pd.Series) -> pd.Series:
    return addr.fillna("").astype(str).str.replace(PAREN, r"\1\2", regex=True)


def rf():
    """Re-parse hn for France S2/S3 whose address opens a segment with '(n)'. Only hn can change: the segment keeps
    its digits, so it never was a place (normalize.address_fields drops digit segments from places)."""
    t0 = time.time()
    WFIX.mkdir(parents=True, exist_ok=True)
    for p in W14.iterdir():
        q = WFIX / p.name
        if p.name != "rf_test_sx_France.parquet" and not q.exists():
            q.symlink_to(p.resolve())
    r = pd.read_parquet(W14 / "rf_test_sx_France.parquet")
    a = pd.concat([pq.read_table(WORK / f"test_source{k}.parquet", columns=["entity_id", "business_address", "country"],
                                 filters=[("country", "==", "France")]).to_pandas() for k in (2, 3)], ignore_index=True)
    a = a.set_index("entity_id").reindex(r.id)
    assert a.country.notna().all()
    m = a.business_address.str.contains(PAREN, regex=True).fillna(False).to_numpy()
    sub = a[m]
    old = N.house_number(sub.business_address, sub.country).to_numpy()
    assert (old == r.hn.to_numpy()[m]).all(), "rf hn is not normalize.house_number of the raw address"
    new = N.house_number(paren_hn(sub.business_address), sub.country).to_numpy()
    r.loc[m, "hn"] = new
    r.to_parquet(WFIX / "rf_test_sx_France.parquet", index=False)
    s1 = pq.read_table(WORK / "test_source1.parquet", columns=["business_address", "country"],
                       filters=[("country", "==", "France")]).to_pandas()
    res = {"sx": len(r), "paren_rows": int(m.sum()), "hn_was_empty": int((old == "").sum()),
           "hn_changed": int((old != new).sum()), "hn_now_empty": int((new == "").sum()),
           "s1_paren_rows": int(s1.business_address.str.contains(PAREN, regex=True).sum())}
    print(json.dumps(res), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    log("exp", probe=PROBE, subproblem="france-rebuild", name="p29-hn-paren-rf", world="test-France", metrics=res,
        config={"regex": PAREN, "base": str(W14.relative_to(ROOT))}, hypothesis="P6: '(520) RUE X' leaves 1.6% of "
        "France S2/S3 without a house number", verdict="keep", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))


def rescore():
    """P20's rescore.py with rule 'noise' (R1) on WFIX (ER_WORK must point there before the import)."""
    assert Path(os.environ["ER_WORK"]).resolve() == WFIX.resolve()
    import rescore as RS
    RS.OUT = OUT
    RS.main("noise")                                  # -> p29/prob_test_France_w_fix_noise.parquet


# ---------------------------------------------------------------- bge top-up (P19's recipe, P13's job unchanged)
FIX = OUT / "prob_test_France_w_fix_noise.parquet"
R_OLD = [WORK / "p13/r/r_bge_pool_test_France.parquet", WORK / "p19/r/r_bge_pool_test_France.parquet"]
FINAL = OUT / "prob_test_France_fix_bge.parquet"
DATA = OUT / "sm_data"
LO, HI = 0.02, 0.98
W_FR = [0.9245, 0.2595, 0.0515]                        # P13's label-free out-of-domain France weights (as #9ab)
N_CTL = 2000
PREFIX = "amc26-p29"
BUCKET = "sagemaker-114872709824-ap-south-1"


def r_known() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f) for f in R_OLD], ignore_index=True).drop_duplicates(["s1", "sx"])


def build():
    """test_France = the new file's band pairs with no P13 / P19 r + N_CTL controls P13 scored; the India/US/holdout
    files are P19's control slices (the job reads every channel file)."""
    import shutil
    sys.path.insert(0, str(ROOT / "probes/P13-aws-rerank"))
    import p13                                          # puts P4b's rerank on sys.path
    import rerank as rr
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    for f in (WORK / "p19/sm_data").glob("*.parquet"):
        if f.name != "test_France.parquet":
            shutil.copy(f, DATA / f.name)
    d = pd.read_parquet(FIX)
    b = d[(d.p > LO) & (d.p < HI)][["s1", "sx", "p"]].merge(r_known(), on=["s1", "sx"], how="left")
    p13r = pd.read_parquet(R_OLD[0])[["s1", "sx"]]
    miss = b[b.r.isna()]
    have = b[b.r.notna()].merge(p13r, on=["s1", "sx"]).sample(N_CTL, random_state=0)
    g = pd.concat([miss, have], ignore_index=True)
    q1, qx = rr.rows("test", "France", g.s1, g.sx)
    ta, tb = rr.texts("test", "France", q1, qx)
    f = pd.DataFrame({"s1": g.s1.to_numpy(), "sx": g.sx.to_numpy(), "ta": ta, "tb": tb, "p": g.p.to_numpy(np.float32)})
    old = pd.read_parquet(WORK / "p13/sm_data/test_France.parquet", columns=["s1", "sx", "ta", "tb"])
    m = f.iloc[len(miss):].merge(old, on=["s1", "sx"], suffixes=("", "_p13"))
    assert len(m) == len(have) and (m.ta == m.ta_p13).all() and (m.tb == m.tb_p13).all(), "texts differ from P13's"
    f.to_parquet(DATA / "test_France.parquet", index=False)
    res = {"band": len(b), "have_r": int(b.r.notna().sum()), "missing": len(miss), "control": len(have),
           "test_France_rows": len(f), "text_check": "ok"}
    print(json.dumps(res), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    log("exp", probe=PROBE, subproblem="france-rebuild", name="p29-bge-build", world="test-France", metrics=res,
        config={"src": str(FIX.relative_to(ROOT)), "band": [LO, HI]}, hypothesis="the hn fix + R1 move pairs into "
        "the band that neither P13 nor P19 scored; score only those", verdict="keep",
        runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))


def launch(argv):
    sys.path.insert(0, str(ROOT / "probes/P13-aws-rerank"))
    import p13
    p13.launch(argv)


def fetch(job: str):
    subprocess.run(["aws", "s3", "sync", f"s3://{BUCKET}/{PREFIX}/results/{job}/", str(OUT), "--exclude", "ckpt/*",
                    "--profile", "harsh", "--region", "ap-south-1"], check=True)
    print(sorted(p.name for p in (OUT / "r").glob("*.parquet")))


def check():
    """The job's r on the control pairs vs P13's r (P19: median |dlogit| 0, p99 < 0.2)."""
    lg = lambda r: np.log(np.clip(r, 1e-6, 1 - 1e-6) / (1 - np.clip(r, 1e-6, 1 - 1e-6)))
    res = {}
    pairs = [("holdout", "r_bge_poolck_holdout")] + [(c, f"r_bge_pool_test_{c}") for c in ("India", "US", "France")]
    for k, nm in pairs:
        m = pd.read_parquet(OUT / f"r/{nm}.parquet").merge(pd.read_parquet(WORK / f"p13/r/{nm}.parquet"),
                                                          on=["s1", "sx"], suffixes=("", "_p13"))
        dz = np.abs(lg(m.r.to_numpy()) - lg(m.r_p13.to_numpy()))
        res[k] = {"n": len(m), "corr": round(float(np.corrcoef(m.r, m.r_p13)[0, 1]), 6),
                  "med_dlogit": round(float(np.median(dz)), 4), "p99_dlogit": round(float(np.quantile(dz, 0.99)), 4)}
    print(json.dumps(res), flush=True)
    ok = all(v["corr"] > 0.999 and v["p99_dlogit"] < 0.2 for v in res.values()) and res["France"]["n"] == N_CTL
    log("exp", probe=PROBE, subproblem="france-rebuild", name="p29-sm-repro", world="test+holdout", metrics=res,
        config={"job_r": str((OUT / "r").relative_to(ROOT))}, hypothesis="the job reproduces P13's r on controls",
        verdict="keep" if ok else "drop", runtime_s=0, peak_gb=round(peak(), 2))
    assert ok, res


def blend():
    """#9ab's France recipe on the rebuilt file: r = P13 / P19 / this job's r; p' = rr.blend(p, r, W_FR, LO, HI)."""
    sys.path.insert(0, str(ROOT / "probes/P13-aws-rerank"))
    import p13
    import rerank as rr
    t0 = time.time()
    d = pd.read_parquet(FIX)
    new = pd.read_parquet(OUT / "r/r_bge_pool_test_France.parquet")
    r_all = pd.concat([r_known(), new], ignore_index=True).drop_duplicates(["s1", "sx"])
    g = d.p.to_numpy(np.float32)
    band = (g > LO) & (g < HI)
    r = np.full(len(d), np.nan, np.float32)
    r[band] = d[band][["s1", "sx"]].merge(r_all, on=["s1", "sx"], how="left").r.to_numpy()
    assert not np.isnan(r[band]).any(), f"{int(np.isnan(r[band]).sum())} band pairs without r"
    p = rr.blend(g, r, W_FR, LO, HI)
    assert (p[~band] == g[~band]).all() and len(p) == len(d)
    pd.DataFrame({"s1": d.s1, "sx": d.sx, "p": p, "p_gbdt": g}).to_parquet(FINAL, index=False)
    res = {"pairs": len(d), "band": int(band.sum()), "r_new_job": len(new.merge(d[band][["s1", "sx"]])),
           "ce_gbdt_disagree": round(float(((g[band] >= 0.5) != (r[band] >= 0.5)).mean()), 4)}
    print(json.dumps(res), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    log("exp", probe=PROBE, subproblem="france-rebuild", name="p29-blend-fix-bge", world="test-France", metrics=res,
        config={"w": W_FR, "band": [LO, HI], "out": str(FINAL.relative_to(ROOT))},
        hypothesis="#9ab's bge blend on the hn-fixed + R1 France rebuild", verdict="keep",
        runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))


# ---------------------------------------------------------------- key tier + per-S1 cap (label-free, any country)
CAP = {"S2": 5, "S3": 6}
FLOOR = 0.5         # tier pairs below it are what the model rejects in-country, and they are ~0-5% true (holdout)
T_HI = 0.985        # #9ab's France t: the holdout proxy for "t above the country's optimum"                                # max matches per S1 and source in train (asserted in holdout())


def tier_mask(k1: pd.DataFrame, kx: pd.DataFrame, chain: np.ndarray, am: np.ndarray) -> np.ndarray:
    """Key tier: the SX's argmax pair with R1-cleaned core, hn and city all equal (non-empty) and the core unique
    among the country's S1 (chain size 1). k1 / kx: core, hn, city aligned to the pairs."""
    ok = lambda c: (k1[c].to_numpy() == kx[c].to_numpy()) & (k1[c].to_numpy() != "")
    return am & ok("core") & ok("hn") & ok("city") & (chain == 1)


def cap_mask(s1: np.ndarray, sx: np.ndarray, p: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Predicted pairs beyond CAP per S1 and source (lowest p first); returns the pairs to drop."""
    d = pd.DataFrame({"s1": s1, "src": pd.Series(sx).str[:2].to_numpy(), "p": p})[pred]
    rank = d.groupby(["s1", "src"]).p.rank(method="first", ascending=False).to_numpy()
    lim = d.src.map(CAP).to_numpy()
    drop = np.zeros(len(p), bool)
    drop[np.flatnonzero(pred)[rank > lim]] = True
    return drop


def keys(path: Path, ids=None) -> pd.DataFrame:
    """id -> R1-cleaned core, hn, city from a rules.feats file (optionally only ids in `ids`)."""
    parts = []
    for b in pq.ParquetFile(path).iter_batches(500_000, columns=["id", "core", "hn", "city"]):
        b = b.to_pandas()
        if ids is not None:
            b = b[b.id.isin(ids)]
        parts.append(b)
    k = pd.concat(parts, ignore_index=True)
    k["core"] = k.core.map(fr20._strip).astype("str")
    return k.set_index("id")


def f05(tp, npred, ntrue):
    pr, rc = np.divide(tp, npred, out=np.zeros(len(tp)), where=npred > 0), np.divide(tp, ntrue, out=np.zeros(len(tp)), where=ntrue > 0)
    f = np.divide(1.25 * pr * rc, 0.25 * pr + rc, out=np.zeros(len(tp)), where=(0.25 * pr + rc) > 0)
    return np.where(ntrue == 0, (npred == 0).astype(float), f)


def holdout():
    """#8c's dense holdout (P23 dump: every competing train S1, #8c's p / argmax / pred at its t): tier precision and
    count, and per-S1 F0.5 with pred | tier, pred minus cap, and both; paired vs #8c on P5's report half."""
    import run as RUN
    t0 = time.time()
    folds = pd.read_parquet(WORK / "s1_folds.parquet")
    gt = pd.read_parquet(WORK / "gt_pairs.parquet")
    per = gt.assign(src=gt.sx.str[:2]).groupby(["s1", "src"]).size().unstack(fill_value=0)
    train_cap = {c: int(per[c].max()) for c in per.columns}
    print("train max matches per S1:", train_cap, flush=True)
    res, F = {"train_max_per_s1": train_cap}, {}
    for c in ("US", "India"):
        d = pd.read_parquet(WORK / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p", "am", "pred"])
        k1 = keys(WORK / f"rf_train_s1_{c}.parquet")
        chain = k1.core.map(k1.core.value_counts()).reindex(d.s1).to_numpy()
        kx = keys(WORK / f"rf_train_sx_{c}.parquet", set(d.sx))
        tier = tier_mask(k1.reindex(d.s1), kx.reindex(d.sx).fillna(""), chain, d.am.to_numpy())
        del kx
        h = d.holdout.to_numpy()
        y, pred = d.y.to_numpy(), d.pred.to_numpy()
        new = tier & ~pred
        cap0 = cap_mask(d.s1.to_numpy(), d.sx.to_numpy(), d.p.to_numpy(), pred)
        pt = pred | tier
        cap1 = cap_mask(d.s1.to_numpy(), d.sx.to_numpy(), np.where(tier, 1.0, d.p.to_numpy()), pt)
        pp, am = d.p.to_numpy(), d.am.to_numpy()
        t5 = tier & (pp >= FLOOR)
        hi = am & (pp >= T_HI)                            # France-like: t far above the country's own optimum
        arms = {"base": pred, "tier": pt, "cap": pred & ~cap0, "tier_cap": pt & ~cap1, "tier05": pred | t5,
                "hi": hi, "hi_tier05": hi | t5, "hi_tier": hi | tier}
        ids = folds.entity_id[(folds.fold == 0) & (folds.country == c)].to_numpy()
        ntrue = gt[gt.s1.isin(set(ids))].groupby("s1").size().reindex(ids, fill_value=0).to_numpy()
        hs = d.s1.to_numpy()[h]
        for a, m in arms.items():
            g = pd.DataFrame({"s1": hs, "tp": (m & y)[h], "np": m[h]}).groupby("s1").sum()
            g = g.reindex(ids, fill_value=0)
            F[(c, a)] = f05(g.tp.to_numpy(float), g.np.to_numpy(float), ntrue)
        half = RUN.half(ids)
        r = {"tier_pairs": int((tier & h).sum()), "tier_precision": round(float(y[tier & h].mean()), 5),
             "tier_new_pairs": int((new & h).sum()), "tier_new_precision": round(float(y[new & h].mean()), 5) if (new & h).any() else None,
             "tier_already_pred": int((tier & pred & h).sum()),
             "cap_s1_touched": int(len(set(d.s1.to_numpy()[cap0 & h]))), "cap_pairs_dropped": int((cap0 & h).sum()),
             "cap_dropped_precision": round(float(y[cap0 & h].mean()), 4) if (cap0 & h).any() else None,
             "f05_base_all": round(float(F[(c, "base")].mean()), 6)}
        r |= {"tier05_band_pairs": int((t5 & ~hi & h).sum()), "tier05_band_precision": round(float(y[t5 & ~hi & h].mean()), 4),
              "tier_below_floor_precision": round(float(y[tier & (pp < FLOOR) & h].mean()), 4)}
        for a, b in (("tier", "base"), ("cap", "base"), ("tier_cap", "base"), ("tier05", "base"),
                     ("hi_tier05", "hi"), ("hi_tier", "hi")):
            dl = F[(c, a)] - F[(c, b)]
            r[f"d_{a}_half1"] = round(float(dl[half == 1].mean()), 6)
            r[f"se_{a}_half1"] = round(float(dl[half == 1].std() / np.sqrt((half == 1).sum())), 6)
            r[f"d_{a}_all"] = round(float(dl.mean()), 6)
        res[c] = r
        print(c, json.dumps(r), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
        del d, k1
    base = np.concatenate([F[(c, "base")] for c in ("US", "India")])
    res["f05_base_both"] = round(float(base.mean()), 6)
    for a in ("tier", "cap", "tier_cap", "tier05"):
        res[f"d_{a}_both_all"] = round(float(np.concatenate([F[(c, a)] for c in ("US", "India")]).mean() - base.mean()), 6)
    print(json.dumps({k: v for k, v in res.items() if k not in ("US", "India")}), flush=True)
    log("exp", probe=PROBE, subproblem="key-tier", name="p29-tier-cap-holdout", world="holdout-dense-half1(#8c)",
        metrics=res, config={"tier": "argmax & R1 core eq & hn eq & city eq & S1 core chain 1", "cap": CAP, "floor": FLOOR, "t_hi": T_HI,
                             "src": "data/work/p23/pairs_{c}.parquet (#8c p, t)"},
        hypothesis="key-equal pairs of a unique S1 core are ~99% true; train never has >5 S2 / >6 S3 per S1",
        verdict="keep", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))


N_FR = 259_452                                          # France test S1


def curve(d: pd.DataFrame, ts) -> pd.DataFrame:
    """France profile per t: empty-S1 share and matches per S1 (argmax S1 per SX, p >= t) -- combine.py's rule."""
    import model as M
    am = M.argmax_sx(pd.factorize(d.sx)[0], d.p.to_numpy())
    a = d[am]
    return pd.DataFrame([{"t": t, "empty": 1 - a.s1[a.p.to_numpy() >= t].nunique() / N_FR,
                          "mps": int((a.p.to_numpy() >= t).sum()) / N_FR} for t in ts])


def t_at(c: pd.DataFrame, empty: float) -> float:
    """Smallest t on the grid whose empty share reaches `empty` (#9ab: 0.0671 at 0.985)."""
    return float(c["t"][c.empty >= empty - 1e-9].min())


def profile(path=str(FINAL)):
    d = pd.read_parquet(path, columns=["s1", "sx", "p"])
    c = curve(d, np.round(np.arange(0.95, 0.9951, 0.0025), 4))
    print(c.round(5).to_string(index=False))
    return c


def tiers(path, t, out, floor=FLOOR):
    """France file -> key tier (p := 1 on tier pairs with p >= floor; they are their SX's argmax already) and the
    per-S1 cap at t (an SX over the cap loses every pair, so it cannot fall back to another S1). -> OUT (s1, sx, p)."""
    import model as M
    t0, t, floor = time.time(), float(t), float(floor)
    d = pd.read_parquet(path, columns=["s1", "sx", "p"])
    k1 = keys(W14 / "rf_test_s1_France.parquet")                  # S1 has no '(n)' house numbers (rf() counts 0)
    chain = k1.core.map(k1.core.value_counts()).reindex(d.s1).to_numpy()
    kx = keys(WFIX / "rf_test_sx_France.parquet")
    p = d.p.to_numpy(np.float32).copy()
    am = M.argmax_sx(pd.factorize(d.sx)[0], p)
    tier = tier_mask(k1.reindex(d.s1), kx.reindex(d.sx).fillna(""), chain, am)
    del kx
    t5 = tier & (p >= floor)
    acc0 = am & (p >= t)
    p[t5] = 1.0
    acc1 = am & (p >= t)                                            # argmax unchanged: only argmax pairs went up
    drop = cap_mask(d.s1.to_numpy(), d.sx.to_numpy(), p, acc1)
    p[d.sx.isin(set(d.sx[drop])).to_numpy()] = 0.0
    acc2 = M.argmax_sx(pd.factorize(d.sx)[0], p) & (p >= t)
    pd.DataFrame({"s1": d.s1, "sx": d.sx, "p": p}).to_parquet(out, index=False)
    ne = lambda a: d.s1[a].nunique()
    res = {"t": t, "floor": floor, "tier_pairs": int(tier.sum()), "tier_ge_floor": int(t5.sum()),
           "tier_below_floor": int((tier & ~t5).sum()), "pairs_added_by_tier": int((acc1 & ~acc0).sum()),
           "empty_s1_fixed_by_tier": ne(acc1) - ne(acc0), "cap_pairs_dropped": int(drop.sum()),
           "cap_s1_touched": int(d.s1[drop].nunique()), "empty_before": round(1 - ne(acc0) / N_FR, 5),
           "empty_after": round(1 - ne(acc2) / N_FR, 5), "mps_before": round(int(acc0.sum()) / N_FR, 4),
           "mps_after": round(int(acc2.sum()) / N_FR, 4)}
    print(json.dumps(res), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    log("exp", probe=PROBE, subproblem="key-tier", name=f"p29-tier-cap-France-t{t}", world="test-France",
        metrics=res, config={"src": str(path), "out": str(out), "cap": CAP},
        hypothesis="France's key-equal pairs of unique S1 cores sit just under its high t (0.7-0.98), the band where "
        "the holdout proxy says they are 96-97% true", verdict="keep", runtime_s=round(time.time() - t0),
        peak_gb=round(peak(), 2))


def diff(a_spec, b_spec):
    """France decisions of two 'file:t' specs: S1 changed, pairs added / dropped (b vs a) and the hn lens on them
    (share hn-equal, share sibling offset SIB3; hn = WFIX's re-parsed hn)."""
    import model as M
    hn = pd.concat([pd.read_parquet(W14 / "rf_test_s1_France.parquet", columns=["id", "hn"]),
                    pd.read_parquet(WFIX / "rf_test_sx_France.parquet", columns=["id", "hn"])]).set_index("id").hn

    def dec(spec):
        f, t = spec.rsplit(":", 1)
        d = pd.read_parquet(f, columns=["s1", "sx", "p"])
        return d[M.argmax_sx(pd.factorize(d.sx)[0], d.p.to_numpy()) & (d.p.to_numpy() >= float(t))][["s1", "sx"]]
    a, b = dec(a_spec), dec(b_spec)
    m = a.merge(b, how="outer", indicator=True)
    res = {"a_pairs": len(a), "b_pairs": len(b), "added": int((m._merge == "right_only").sum()),
           "dropped": int((m._merge == "left_only").sum())}
    for k, sel in (("added", "right_only"), ("dropped", "left_only")):
        x = m[m._merge == sel]
        h1, hx = hn.reindex(x.s1).fillna("").to_numpy(), hn.reindex(x.sx).fillna("").to_numpy()
        ok = (h1 != "") & (hx != "")
        dd = np.zeros(len(x), np.int64)
        dd[ok] = pd.to_numeric(pd.Series(hx[ok])).to_numpy() - pd.to_numeric(pd.Series(h1[ok])).to_numpy()
        res[f"{k}_hn_eq"] = round(float((ok & (h1 == hx)).mean()), 4) if len(x) else None
        res[f"{k}_sib3"] = round(float((ok & np.isin(dd, fr20.SIB3)).mean()), 4) if len(x) else None
        res[f"{k}_hn_missing"] = round(float((~ok).mean()), 4) if len(x) else None
    A, B = a.groupby("s1").sx.apply(frozenset).to_dict(), b.groupby("s1").sx.apply(frozenset).to_dict()
    res["s1_changed"] = sum(A.get(k, frozenset()) != B.get(k, frozenset()) for k in A.keys() | B.keys())
    print(json.dumps(res), flush=True)
    return res


def selftest():
    S = lambda *v: pd.Series(list(v), dtype="str")
    assert paren_hn(S("(520) RUE X, LILLE", "LILLE, (12) AV Y", "12 (bis) rue")).tolist() == \
        ["520 RUE X, LILLE", "LILLE, 12 AV Y", "12 (bis) rue"]
    k1 = pd.DataFrame({"core": ["a", "a", "a", "", "a"], "hn": ["1", "1", "1", "1", "1"], "city": ["x"] * 5})
    kx = pd.DataFrame({"core": ["a", "a", "a", "", "a"], "hn": ["1", "2", "1", "1", "1"], "city": ["x"] * 5})
    got = tier_mask(k1, kx, np.array([1, 1, 2, 1, 1]), np.array([True, True, True, True, False]))
    assert got.tolist() == [True, False, False, False, False], got
    s1, sx = np.array(["a"] * 8), np.array([f"S2-{i}" for i in range(7)] + ["S3-1"])
    assert cap_mask(s1, sx, np.linspace(0.9, 0.2, 8), np.ones(8, bool)).tolist() == [False] * 5 + [True, True, False]
    assert np.allclose(f05(np.array([2., 0, 0, 1]), np.array([3., 0, 1, 1]), np.array([2., 0, 0, 5])),
                       [0.714286, 1, 0, 0.555556], atol=1e-6)
    print("p29 self-check ok")


if __name__ == "__main__":
    cmd, pos = sys.argv[1], sys.argv[2:]
    if cmd == "launch":
        launch(pos)
    else:
        {"rf": rf, "rescore": rescore, "build": build, "fetch": fetch, "check": check, "blend": blend, "holdout": holdout, "profile": profile, "tiers": tiers, "diff": diff, "selftest": selftest}[cmd](*pos)
