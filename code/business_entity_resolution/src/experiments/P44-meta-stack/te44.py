"""P44 ship: the bagged combiner (5 residual boosters over #8c's argmax pairs, fit on holdout half 0) on test.

    py.sh te44.py fit              boosters -> data/work/p44ms/ship_s{0..4}.txt; asserts they reproduce the bench's h1
    py.sh te44.py build <c>        test frame from P34's te_feat / te_pairs / prob_test + P44 preds -> T_{c}.parquet
    py.sh te44.py probs <c>        -> prob_test_{c}_p44.parquet (s1, sx, p = p'' on argmax pairs else #34's p', p8c, am)
    py.sh te44.py profile          label-free flip gate: test vs holdout-h1 rates, split by other-S1 mass class
    py.sh te44.py combine SRC OUT  SRC's India/US lines (must equal #34's decision) -> the P44 decision
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev as E  # noqa: E402
import meta as MT  # noqa: E402
import a10  # noqa: E402
from best import HNF  # noqa: E402
from ctx import CTX2, CTX3, ctx2, add3  # noqa: E402

import os
VAR = os.environ.get("P44_VARIANT", "")                  # "" = the full model; "psumoff" = without sx_psum
FINAL = a10.COLS + ["l_m2hn", "l_hn21"] + HNF + ["lq", "tw_any"] + CTX2 + CTX3      # final.COLS minus lfreq
if VAR == "psumoff":
    FINAL = [c for c in FINAL if c != "sx_psum"]
if VAR == "r":                                          # + P45's bge CE on every pair (r_all)
    FINAL = FINAL + ["l_r_all"]
MODELS = ["m2hn", "hn21"] + (["r_all"] if VAR == "r" else [])
TAG = f"p44{VAR}"
SEEDS = (0, 1, 2, 3, 4)
PRM = lambda s: dict(MT.PRM, seed=s, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, deterministic=True)
P34 = E.W / "p34"


def combine_p(boosters, X: pd.DataFrame) -> np.ndarray:
    """best.bag's output: sigmoid of the mean logit of the guarded boosters (guard: hn class 1 keeps #8c's p)."""
    outs = []
    for b in boosters:
        s = MT.sig(b.predict(X[FINAL], raw_score=True) + X.lp.to_numpy())
        outs.append(MT.lg(np.where(X.hc.to_numpy() == 1, X.p.to_numpy(), s)))
    return MT.sig(np.mean(outs, axis=0))


def fit():
    import lightgbm as lgb
    from best import bag
    D = MT.prep(MODELS)
    tr = {c: D[c][0].half.to_numpy() == 0 for c in E.C}
    X = pd.concat([D[c][0].loc[tr[c], FINAL] for c in E.C])
    y = np.concatenate([D[c][0].y.to_numpy()[tr[c]] for c in E.C])
    base = np.concatenate([D[c][0].lp.to_numpy()[tr[c]] for c in E.C])
    bs = []
    for s in SEEDS:
        b = lgb.train(PRM(s), lgb.Dataset(X, y, init_score=base, categorical_feature=["cat"], free_raw_data=True),
                      200)                                  # = meta.gbdt default rounds
        b.save_model(str(E.OUT / f"ship{VAR}_s{s}.txt"))
        bs.append(b)
    ref = bag(FINAL)(tr, D)                    # the bench's own path
    r = {}
    for c in E.C:
        A, S = D[c]
        p = combine_p(bs, A)
        assert np.allclose(p, ref[c], atol=1e-9), f"{c}: shipped boosters != bench"
        r[c] = E.delta(A, S, p >= E.T, 1)
        np.save(E.OUT / f"ho_{TAG}_{c}.npy", p.astype(np.float32))
        print(c, f"h0 fit -> h1: {r[c][0]:+.6f} ± {r[c][1]:.6f} (+{r[c][2]}/-{r[c][3]})", flush=True)
    lb, se = E.lbeq(r)
    print(f"LB-eq {lb:+.6f} ± {se:.6f} ({lb / se:.1f} SE)", flush=True)


def build(c: str):
    import features as F
    from solution_p44 import HnTab
    cols = ["row", "p", "lp", "lm2", "r", "n_cand_sx", "n_twins", "n_atwins", "n_ctwins", "p2", "p2m", "marginm",
            "n_conf_other", "s1_n_am", "s1_max_other", "noaddr", "cat", "hc", "name_sup", "p_new"]
    A = pd.read_parquet(P34 / f"te_feat_{c}.parquet", columns=cols)
    tp = pq.read_table(P34 / f"te_pairs_{c}.parquet", columns=["s1", "sx", "p"])
    pr = pd.read_parquet(P34 / f"prob_test_{c}_a10.parquet", columns=["p", "p8c", "am"])
    assert np.array_equal(pr.p8c.to_numpy(), tp.column("p").to_numpy()), "prob file p8c != te_pairs p"
    row = A.row.to_numpy()
    assert np.allclose(A.p.to_numpy(), pr.p8c.to_numpy()[row]) and pr.am.to_numpy()[row].all()
    assert np.allclose(A.p_new.to_numpy(), pr.p.to_numpy()[row], atol=1e-6), "te_feat p_new != #34 p'"
    id1, idx = F.ids("test", c)
    q1 = pc.index_in(tp.column("s1"), value_set=id1).to_numpy(zero_copy_only=False)
    qx = pc.index_in(tp.column("sx"), value_set=idx).to_numpy(zero_copy_only=False)
    del id1, idx
    assert not np.isnan(q1.astype(float)).any() and not np.isnan(qx.astype(float)).any()
    q1, qx = q1.astype(np.int64), qx.astype(np.int64)
    s3 = pc.starts_with(tp.column("sx"), "S3-").to_numpy(zero_copy_only=False)
    del tp
    # integer codes stand in for the id strings (ctx2 / add3 only group and merge on them)
    d = pd.DataFrame({"s1": q1, "sx": qx, "p": pr.p8c.to_numpy(np.float64), "am": pr.am.to_numpy()})
    A["q1"], A["qx"], A["s1"], A["sx"] = q1[row], qx[row], q1[row], qx[row]
    A["g"] = pd.factorize(A.q1)[0]
    A["pa"] = A.p_new
    hn = HnTab(E.W / "p44ms/hn", "test")
    for k, v in hn.feats(c, A.q1.to_numpy(), A.qx.to_numpy()).items():
        A[k] = v
    assert (A.hn_cls.to_numpy() == A.hc.to_numpy()).all(), "derived hn class != P34's hc"
    k = E.keys(A.q1, A.qx)
    qt = pq.read_table(E.W / f"p10/q_test_{c}.parquet", columns=["q1", "qx", "q"])
    qs = pd.Series(qt.column("q").to_numpy(), index=E.keys(qt.column("q1").to_numpy(), qt.column("qx").to_numpy()))
    del qt
    A["q"] = qs[~qs.index.duplicated()].reindex(k).to_numpy(np.float32)
    del qs
    for m in ("m2hn", "hn21"):
        t = pq.read_table(E.OUT / f"pred_{m}/test_{c}.parquet")
        s = pd.Series(t.column("p").to_numpy(), index=E.keys(t.column("q1").to_numpy(), t.column("qx").to_numpy()))
        A[f"p_{m}"] = s.reindex(k).to_numpy(np.float32)
        del t, s
    assert A[["q", "p_m2hn", "p_hn21"]].notna().all().all(), A[["q", "p_m2hn", "p_hn21"]].isna().mean()
    hx = hn.t[c]["sx"].take(pa.array(A.qx.to_numpy())).to_numpy(zero_copy_only=False).astype(object)
    h1 = hn.t[c]["s1"].take(pa.array(A.q1.to_numpy())).to_numpy(zero_copy_only=False).astype(object)
    A = ctx2(A, d, h1, hx)
    del h1, hx
    A = add3(A, d.drop(columns="am"), s3=s3[A.row.to_numpy()])
    A.drop(columns=["s1", "sx"]).to_parquet(E.OUT / f"T_{c}.parquet")
    print(c, len(A), "test argmax pairs;", A[CTX2 + CTX3].mean().round(3).to_dict(), flush=True)


def derive(A: pd.DataFrame) -> pd.DataFrame:
    """MT.prep's derived columns that FINAL uses."""
    A["has_hn"] = A.hn_lev.notna().astype(np.float32)
    A["exact_hn"] = ((A.hn_cls == 0) & (A.has_hn > 0)).astype(np.float32)
    A["lq"] = E.lg(A.q.to_numpy())
    A["tw_any"] = (A.n_twins > 0).astype(np.float32)
    for m in MODELS:
        A[f"l_{m}"] = E.lg(A[f"p_{m}"].to_numpy())
    return A


def attach_r_all(c: str, A: pd.DataFrame) -> pd.DataFrame:
    """r_all on test = P45's CE score for the non-band argmax pairs (r_all_test_{c}: s1, sx, r) + the band pairs' own r
    (the same ce_bge_pool CE, already in te_feat). The two cover disjoint pairs; asserted complete."""
    f = pd.read_parquet(E.W / f"p45/r_all_test_{c}.parquet")
    tp = pd.read_parquet(P34 / f"te_pairs_{c}.parquet", columns=["s1", "sx"]).iloc[A.row.to_numpy()]
    m = tp.reset_index(drop=True).merge(f.drop_duplicates(["s1", "sx"]), on=["s1", "sx"], how="left").r.to_numpy(np.float64)
    band = A.r.notna().to_numpy()
    assert not (band & ~np.isnan(m)).any(), "P45's test r_all overlaps the band"
    A["p_r_all"] = np.where(band, A.r.to_numpy(), m)
    assert A.p_r_all.notna().all(), f"r_all missing for {A.p_r_all.isna().sum()} argmax pairs"
    H, _ = E.load(c)
    hb = H.r.notna().to_numpy()
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    print(c, "non-band r_all quantiles  test", np.quantile(m[~band], q).round(4), " holdout",
          np.quantile(H.p_r_all.to_numpy()[~hb], q).round(4), flush=True)
    return A


def probs(c: str):
    import lightgbm as lgb
    bs = [lgb.Booster(model_file=str(E.OUT / f"ship{VAR}_s{s}.txt")) for s in SEEDS]
    A = pd.read_parquet(E.OUT / f"T_{c}.parquet")
    if VAR == "r":
        A = attach_r_all(c, A)
    A = derive(A)
    p = combine_p(bs, A)
    pr = pd.read_parquet(P34 / f"prob_test_{c}_a10.parquet")
    out = pr.p.to_numpy().copy()
    out[A.row.to_numpy()] = p.astype(out.dtype)
    pr.assign(p=out).to_parquet(E.OUT / f"prob_test_{c}_{TAG}.parquet", index=False)
    old = A.pa.to_numpy() >= E.T
    new = p >= E.T
    print(c, f"test argmax pairs {len(A):,}: #34 accepts {old.sum():,}, P44 accepts {new.sum():,}; adds {(new & ~old).sum():,}, "
             f"drops {(old & ~new).sum():,}", flush=True)


def rates(old, new, g, cls):
    add, drop = new & ~old, old & ~new
    r = {"pairs": len(old), "added_per_acc": add.sum() / old.sum(), "added_per_unacc": add.sum() / max((~old).sum(), 1),
         "dropped_per_acc": drop.sum() / old.sum(), "flips_per_pair": (add | drop).mean(),
         "s1_changed": len(set(g[add | drop]))}
    for k in ("none", "mass"):
        m = cls == k
        r[f"flips_per_pair_{k}"] = (add | drop)[m].mean()
        r[f"share_{k}"] = m.mean()
    return r


def profile():
    rows = []
    for c in E.C:
        A, S = E.load(c)
        h1 = A.half.to_numpy() == 1
        p = np.load(E.OUT / f"ho_{TAG}_{c}.npy")
        cls = np.where(A.sx_psum.to_numpy() - A.p.to_numpy() >= 0.02, "mass", "none")
        rh = rates(A.pa.to_numpy()[h1] >= E.T, p[h1] >= E.T, A.g.to_numpy()[h1], cls[h1])
        rh["s1_changed_per_s1"] = rh.pop("s1_changed") / int((S.half == 1).sum())
        T = pd.read_parquet(E.OUT / f"T_{c}.parquet", columns=["row", "p", "pa", "g", "sx_psum"])
        pn = pd.read_parquet(E.OUT / f"prob_test_{c}_{TAG}.parquet", columns=["p"]).p.to_numpy()[T.row.to_numpy()]
        ct = np.where(T.sx_psum.to_numpy() - T.p.to_numpy() >= 0.02, "mass", "none")
        rt = rates(T.pa.to_numpy() >= E.T, pn >= E.T, T.g.to_numpy(), ct)
        n_s1 = int((pd.read_parquet(E.W / "test_source1.parquet", columns=["country"]).country == c).sum())
        rt["s1_changed_per_s1"] = rt.pop("s1_changed") / n_s1
        rows += [dict(country=c, world="holdout_h1", **rh), dict(country=c, world="test", **rt)]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    ok = True
    for c in E.C:
        h, t = (df[(df.country == c) & (df.world == w)].iloc[0] for w in ("holdout_h1", "test"))
        ks = ["added_per_acc", "added_per_unacc", "dropped_per_acc", "s1_changed_per_s1", "flips_per_pair",
              "flips_per_pair_none", "flips_per_pair_mass"]
        ratio = {k: round(t[k] / h[k], 2) for k in ks}
        ok &= max(ratio.values()) <= 1.5
        print(c, "test / holdout:", ratio, "PASS" if max(ratio.values()) <= 1.5 else "FAIL", flush=True)
    df.to_csv(E.OUT / f"profile_{TAG}.csv", index=False)
    return ok


def combine(src: Path, out: Path):
    """P34 te.combine with P44's prob files: SRC's India/US matching lines must be #34's decision exactly."""
    s1 = pd.read_parquet(E.W / "test_source1.parquet", columns=["entity_id", "country"])
    new, base, cand = {}, {}, {}
    for c in E.C:
        d = pd.read_parquet(E.OUT / f"prob_test_{c}_{TAG}.parquet", columns=["s1", "sx", "p", "am"])
        b = pd.read_parquet(P34 / f"prob_test_{c}_a10.parquet", columns=["p"]).p.to_numpy()
        acc = d[d.am & (d.p >= E.T)]
        assert acc.sx.is_unique
        new |= acc.groupby("s1").sx.agg(lambda s: ",".join(sorted(s))).to_dict()
        base |= d[d.am.to_numpy() & (b >= E.T)].groupby("s1").sx.agg(lambda s: ",".join(sorted(s))).to_dict()
        cand |= d.groupby("s1").sx.agg(lambda s: ",".join(sorted(s))).to_dict()
        del d, acc
    ids = set(s1.entity_id[s1.country.isin(E.C)])
    out.mkdir(parents=True, exist_ok=True)
    n_rep = n_chg = n_bmis = n_cmis = 0
    with open(src / "matching_results.tsv", encoding="utf-8") as f, open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write(next(f))
        for line in f:
            s, v = line.rstrip("\n").split("\t", 1)
            if s in ids:
                w = new.get(s, "")
                v = ",".join(sorted(v.split(","))) if v else ""
                n_rep += 1
                n_chg += v != w
                n_bmis += v != base.get(s, "")
                g.write(f"{s}\t{w}\n")
            else:
                g.write(line)
    with open(src / "candidate_pairs.tsv", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i:
                s, v = line.rstrip("\n").split("\t", 1)
                if s in ids:
                    n_cmis += (",".join(sorted(v.split(","))) != cand.get(s, "")) if v else (s in cand)
    import os
    (out / "candidate_pairs.tsv").unlink(missing_ok=True)
    os.link(src / "candidate_pairs.tsv", out / "candidate_pairs.tsv")   # byte-identical: a hard link, no 229 MB copy
    assert n_rep == len(ids), (n_rep, len(ids))
    print(f"{n_rep:,} India/US lines replaced, {n_chg:,} changed vs {src.name}; SRC lines != #34 decision: {n_bmis}; "
          f"candidate lines != prob files: {n_cmis}", flush=True)
    assert n_bmis == 0 and n_cmis == 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        fit()
    elif cmd == "build":
        build(sys.argv[2])
    elif cmd == "probs":
        probs(sys.argv[2])
    elif cmd == "profile":
        profile()
    elif cmd == "combine":
        combine(Path(sys.argv[2]), Path(sys.argv[3]))
