"""P63 step 2 (the gate): LOCO with the typo corrector on the TARGET country.

    score SRC TGT   P6's source-only M2 (data/work/p63/models/<SRC>.txt, = P6 models/<SRC>.txt) on every train pair of
                    TGT whose SX has a fold-0 candidate: g0 = the pair's P4 X as is (must reproduce P6's S_holdout p),
                    g1 = the same X with the string features recomputed (features.strings) on corrected SX text (name,
                    core, addr via rates.py's maps of TGT; sq / acro follow core). Context / block / hn / city / state /
                    rule features stay (blocking unchanged). -> data/work/p63/g_<SRC>_<TGT>.parquet (q1, qx, fold, g0, g1, ch)
    frame SRC TGT   re-argmax per SX over every train S1 with g1, keep fold-0 argmax rows, in P42's npz format
                    (s1 code, is2, g, r = P30 source-only bge, y, hn_rel = P39's) -> data/work/p63/loco_<SRC>_<TGT>{_c0,_c1}.npz;
                    _c0 (g0) must equal P42's frame row for row.
    eval            P42's chain + m2 analogues, recipe gc70 (= #37, the sub_iu54_frgc70 France recipe) and sub42_both,
                    thresholds (a) fixed at the uncorrected map, (b) re-mapped by density on the corrected p.

    .venv/bin/python probes/P63-fr-typo-correct/loco.py score US India | frame US India | eval
"""
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P42-fr-joint-tune")]
import typo as T  # noqa: E402

W = ROOT / "data/work"
OUT = W / "p63"
CI = {"India": 0, "US": 1}
OFF1, OFF2 = 10_000_000, 100_000_000
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
STR = None   # string feature columns recomputed (features.strings output minus indic_x)


def correct_rx(rx: pd.DataFrame, mn: pd.Series, ma: pd.Series) -> tuple:
    import normalize as N
    rx = rx.copy()
    rx["name"], a = T.apply(pd.Series(rx.name.array), mn)
    core, b = T.apply(pd.Series(rx.core.array), mn)
    rx["core"] = core.to_numpy()
    rx["addr"], c = T.apply(pd.Series(rx.addr.array), ma)
    rx["acro"] = N.acronym(rx.core).astype("str").array
    rx["sq"] = rx.core.str.replace(" ", "", regex=False)
    return rx, a | b | c


def pair_fix(sx: list, s1: list, oov: set) -> tuple:
    """Pair-wise variant: in each SX string, an OOV token t (len >= 4, no digit) becomes the unique token v of the
    PAIR's S1 string at the minimal DL distance <= 1 (<= 2 for len(t) >= 8). -> (new strings, changed mask)."""
    from rapidfuzz.distance import DamerauLevenshtein as DL
    out, ch = [], np.zeros(len(sx), bool)
    for k, (x, y) in enumerate(zip(sx, s1)):
        toks = x.split(" ")
        bad = [a for a, t in enumerate(toks) if t in oov]
        if not bad or not y:
            out.append(x)
            continue
        ys = set(y.split(" "))
        for a in bad:
            t = toks[a]
            lim = 2 if len(t) >= 8 else 1
            best, bd, amb = None, lim + 1, False
            for v in ys:
                if abs(len(v) - len(t)) > lim:
                    continue
                d = DL.distance(t, v, score_cutoff=lim)
                if d < bd:
                    best, bd, amb = v, d, False
                elif d == bd:
                    amb = True
            if best is not None and not amb:
                toks[a] = best
                ch[k] = True
        out.append(" ".join(toks))
    return out, ch


def oov_sets(tgt):
    """Globally OOV SX tokens (not in the S1 vocab of TGT, len >= 4, no digit), per family."""
    import rates as R
    s1 = R.s1_texts("train", tgt)
    vn = set(T.counts(s1.name).index) | set(T.counts(s1.core).index)
    va = set(T.counts(s1.addr).index)
    del s1
    ok = lambda t, v: t not in v and T._OK.match(t) is not None
    return (lambda t: ok(t, vn)), (lambda t: ok(t, va)), vn, va


def score(src, tgt):
    import lightgbm as lgb
    import features as F
    import model as M
    t0 = time.time()
    ci = CI[tgt]
    m = lgb.Booster(model_file=str(OUT / f"models/{src}.txt"))
    S = pd.read_parquet(W / f"p27/p6/S_holdout_{src}.parquet", columns=["sx", "country", "s1", "p"])
    S = S[S.country.to_numpy() == ci]
    need = np.unique(S.sx.to_numpy() - ci * OFF2).astype(np.int64)
    ref = pd.Series(S.p.to_numpy(np.float32), index=(S.s1.to_numpy(np.int64) - ci * OFF1) << 32 | (S.sx.to_numpy(np.int64) - ci * OFF2))
    del S
    mn = pd.read_parquet(OUT / f"map_train_{tgt}_name.parquet").set_index("tok").to
    ma = pd.read_parquet(OUT / f"map_train_{tgt}_addr.parquet").set_index("tok").to
    # S1 side: whole-pool counts (chain_*), record fields
    r1 = F.s1_table("train", tgt)
    counts = F.pool_counts(r1)
    files, offs = F.sx_files("train", tgt)
    rfs = F.pieces(W / f"rf_train_sx_{tgt}.parquet", np.diff(offs))
    parts, chk = [], []
    for k, f in enumerate(files):
        rf = next(rfs)
        xf = F.XDIR / f"train_{f.stem}.parquet"
        off = offs[k]
        lo, hi = np.searchsorted(need, [off, offs[k + 1]])
        mine = need[lo:hi] - off
        if not len(mine):
            continue
        txt = pd.read_parquet(f)
        rx = F.assemble(rf[F.RF], txt, pd.Series("", index=txt.index))
        del txt, rf
        rc, ch = correct_rx(rx, mn, ma)
        pf = pq.ParquetFile(xf)
        for bt in pf.iter_batches(200_000):
            X = bt.to_pandas()
            X = X[np.isin(X.qx.to_numpy() - off, mine)].reset_index(drop=True)
            if not len(X):
                continue
            g0 = M.predict(m, X)
            j = X.qx.to_numpy() - off
            c = ch[j]
            g1 = g0.copy()
            if c.any():
                i1 = X.q1.to_numpy()[c]
                u, inv = np.unique(i1, return_inverse=True)
                r1s = r1.iloc[u].reset_index(drop=True)
                m1 = F.mats(r1s)
                ju, jinv = np.unique(j[c], return_inverse=True)
                rxs = rc.iloc[ju].reset_index(drop=True)
                S2 = F.strings(r1s, rxs, inv, jinv, counts, m1, F.mats(rxs)).drop(columns="indic_x").round(3)
                if not chk:                              # sanity once: uncorrected recompute == P4's X
                    rxo = rx.iloc[ju].reset_index(drop=True)
                    S0 = F.strings(r1s, rxo, inv, jinv, counts, m1, F.mats(rxo)).drop(columns="indic_x").round(3)
                    x0 = X.loc[c, S0.columns].reset_index(drop=True)
                    bad = {col: int((~np.isclose(S0[col].to_numpy(), x0[col].to_numpy(), atol=1.5e-3, equal_nan=True)).sum()) for col in S0.columns}
                    chk.append({kk: v for kk, v in bad.items() if v})
                    print(f"  recompute check (uncorrected vs X, {c.sum():,} rows): mismatching cols {chk[-1]}", flush=True)
                    del S0, rxo, x0
                X1 = X.loc[c].reset_index(drop=True)
                X1[S2.columns] = S2.to_numpy()
                g1[c] = M.predict(m, X1)
                del S2, X1, m1, r1s, rxs
            parts.append(pd.DataFrame({"q1": X.q1.to_numpy(np.int32), "qx": X.qx.to_numpy(np.int32),
                                       "fold": X.fold.to_numpy(np.int8), "y": X.is_true.to_numpy(bool),
                                       "g0": g0, "g1": g1, "ch": c}))
            del X
        F.trim()
        print(f"  {f.stem}: {sum(map(len, parts)):,} rows so far, SX changed {ch[mine].mean():.3f}; "
              f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
        del rx, rc, ch
    d = pd.concat(parts, ignore_index=True)
    k = (d.q1.to_numpy(np.int64) << 32) | d.qx.to_numpy(np.int64)
    f0 = d.fold.to_numpy() == 0
    rp = ref.reindex(k[f0]).to_numpy()
    print(f"  g0 vs S_holdout p on fold 0: {np.isnan(rp).sum()} missing, max |diff| {np.nanmax(np.abs(rp - d.g0.to_numpy()[f0])):.2e}; "
          f"{len(ref):,} ref rows, {f0.sum():,} mine", flush=True)
    d.to_parquet(OUT / f"g_{src}_{tgt}.parquet", index=False)
    print(f"score {src}>{tgt}: {len(d):,} rows, changed pairs {d.ch.mean():.3f}, |g1-g0|>0.01 on {(np.abs(d.g1 - d.g0) > .01).mean():.4f}; "
          f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)


def score_pair(src, tgt):
    """g1 with pair_fix instead of the global map (g0 from g_<SRC>_<TGT>.parquet) -> g_<SRC>_<TGT>_pair.parquet."""
    import lightgbm as lgb
    import features as F
    import model as M
    import normalize as N
    t0 = time.time()
    ci = CI[tgt]
    m = lgb.Booster(model_file=str(OUT / f"models/{src}.txt"))
    base = pd.read_parquet(OUT / f"g_{src}_{tgt}.parquet", columns=["q1", "qx", "fold", "y", "g0"])
    _, _, vn, va = oov_sets(tgt)
    r1 = F.s1_table("train", tgt)
    counts = F.pool_counts(r1)
    n1, c1, a1 = r1.name.to_numpy(object), r1.core.to_numpy(object), r1.addr.to_numpy(object)
    files, offs = F.sx_files("train", tgt)
    rfs = F.pieces(W / f"rf_train_sx_{tgt}.parquet", np.diff(offs))
    qxall = base.qx.to_numpy()
    g1all, chall = base.g0.to_numpy().copy(), np.zeros(len(base), bool)
    for k, f in enumerate(files):
        rf = next(rfs)
        off = offs[k]
        rows = np.flatnonzero((qxall >= off) & (qxall < offs[k + 1]))
        if not len(rows):
            continue
        txt = pd.read_parquet(f)
        rx = F.assemble(rf[F.RF], txt, pd.Series("", index=txt.index))
        del txt, rf
        fam = {"name": vn, "core": vn, "addr": va}
        oov = {c: {t for t in set(T.counts(rx[c]).index) if t not in fam[c] and T._OK.match(t)} for c in fam}
        has = np.zeros(len(rx), bool)
        for c in fam:
            has |= np.fromiter((any(t in oov[c] for t in x.split(" ")) for x in rx[c].tolist()), bool, len(rx))
        rows = rows[has[qxall[rows] - off]]
        xf = F.XDIR / f"train_{f.stem}.parquet"
        want = pd.Series(np.arange(len(rows)), index=(base.q1.to_numpy(np.int64)[rows] << 32) | qxall[rows].astype(np.int64))
        for bt in pq.ParquetFile(xf).iter_batches(200_000):
            X = bt.to_pandas()
            kk = (X.q1.to_numpy(np.int64) << 32) | X.qx.to_numpy(np.int64)
            pos = want.reindex(kk).to_numpy()
            sel = ~np.isnan(pos)
            if not sel.any():
                continue
            X = X[sel].reset_index(drop=True)
            dst = rows[pos[sel].astype(np.int64)]
            i, j = X.q1.to_numpy(), X.qx.to_numpy() - off
            new, ch = {}, np.zeros(len(X), bool)
            for c, s1c in (("name", n1), ("core", c1), ("addr", a1)):
                new[c], cc = pair_fix(rx[c].to_numpy(object)[j].tolist(), s1c[i].tolist(), oov[c])
                ch |= cc
            if ch.any():
                w = np.flatnonzero(ch)
                rxs = rx.iloc[j[w]].reset_index(drop=True)
                for c in ("name", "core", "addr"):
                    rxs[c] = pd.array([new[c][a] for a in w], dtype=rx[c].dtype)
                rxs["acro"] = N.acronym(rxs.core).astype("str").array
                rxs["sq"] = rxs.core.str.replace(" ", "", regex=False)
                u, inv = np.unique(i[w], return_inverse=True)
                r1s = r1.iloc[u].reset_index(drop=True)
                S2 = F.strings(r1s, rxs, inv, np.arange(len(w)), counts, F.mats(r1s), F.mats(rxs)).drop(columns="indic_x").round(3)
                X1 = X.iloc[w].reset_index(drop=True)
                X1[S2.columns] = S2.to_numpy()
                g1all[dst[w]] = M.predict(m, X1)
                chall[dst[w]] = True
                del S2, X1, rxs, r1s
            del X
        F.trim()
        print(f"  {f.stem}: changed pairs so far {chall.sum():,}; {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
        del rx
    base["g1"], base["ch"] = g1all.astype(np.float32), chall
    base.to_parquet(OUT / f"g_{src}_{tgt}_pair.parquet", index=False)
    dg = base.g1 - base.g0
    print(f"score_pair {src}>{tgt}: changed {chall.mean():.3f}; y=1 up/down>0.05 {(dg[base.y] > .05).mean():.4f}/{(dg[base.y] < -.05).mean():.4f}, "
          f"y=0 {(dg[~base.y] > .05).mean():.4f}/{(dg[~base.y] < -.05).mean():.4f}; {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)


def hn_rel(h1, hx):
    """P39 frame.hn_rel: 0 miss, 1 eq, 2 SIB3, 3 d in {1,2}, 4 d in {-1,-2}, 5 other."""
    SIB3 = np.array([3, 4, 5, 7, 9, 11, 13, 21])
    ok = (h1 != "") & (hx != "")
    d = np.full(len(h1), np.nan)
    d[ok] = pd.to_numeric(pd.Series(hx[ok]), errors="coerce").to_numpy(float) - \
        pd.to_numeric(pd.Series(h1[ok]), errors="coerce").to_numpy(float)
    out = np.where(~ok | np.isnan(d), 0, np.where(d == 0, 1, np.where(np.isin(d, SIB3), 2,
                   np.where(np.isin(d, [1, 2]), 3, np.where(np.isin(d, [-1, -2]), 4, 5)))))
    return out.astype(np.int8)


def frame(src, tgt, tag=""):
    import features as F
    import pyarrow as pa
    import solution_p42 as J
    t0 = time.time()
    ci = CI[tgt]
    d = pd.read_parquet(OUT / f"g_{src}_{tgt}{tag}.parquet")
    z = dict(np.load(W / f"p42/loco_{src}_{tgt}.npz"))
    fo = pd.read_parquet(W / "s1_folds.parquet")
    fo = fo[(fo.fold == 0) & (fo.country == tgt)].entity_id.to_numpy()
    assert len(fo) == len(z["ntrue"])
    id1, idx = F.ids("train", tgt)
    rr = pd.read_parquet(W / f"p30/r/r_loco{src}_tgt_{tgt}.parquet")
    rr = pd.Series(rr.r.to_numpy(np.float64), index=(rr.s1.to_numpy(np.int64) - ci * OFF1) << 32 | (rr.sx.to_numpy(np.int64) - ci * OFF2))
    h1s = pq.read_table(W / f"rf_train_s1_{tgt}.parquet", columns=["id", "hn"]).to_pandas().set_index("id").hn
    hxs = pq.read_table(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "hn"]).to_pandas().set_index("id").hn
    qx = d.qx.to_numpy()
    for v in (0, 1):
        g = d[f"g{v}"].to_numpy(np.float64)
        am = J.argmax_sx(qx, g) & (d.fold.to_numpy() == 0)
        e = d[am]
        e1 = id1.take(pa.array(e.q1.to_numpy())).to_numpy(zero_copy_only=False)
        ex = idx.take(pa.array(e.qx.to_numpy())).to_numpy(zero_copy_only=False)
        code = pd.Index(fo).get_indexer(e1)
        assert (code >= 0).all()
        k = (e.q1.to_numpy(np.int64) << 32) | e.qx.to_numpy(np.int64)
        out = {"s1": code.astype(np.int32), "is2": pd.Series(ex).str.startswith("S2-").to_numpy(), "g": g[am],
               "r": rr.reindex(k).to_numpy(np.float64), "y": e.y.to_numpy(bool),
               "hn_rel": hn_rel(h1s.reindex(e1).fillna("").to_numpy(), hxs.reindex(ex).fillna("").to_numpy()),
               "ntrue": z["ntrue"], "hv": z["hv"], "k": k, "ch": e.ch.to_numpy()}
        if v == 0:                                     # must be P42's frame (order aside)
            o = np.lexsort((ex, out["s1"]))
            kz = np.lexsort((np.arange(len(z["s1"])), z["s1"]))
            same = len(o) == len(z["s1"]) and all(np.array_equal(np.sort(out[c]), np.sort(z[c])) for c in ("s1", "y", "hn_rel"))
            print(f"  c0 vs P42 frame: rows {len(o):,} / {len(z['s1']):,}, sorted s1/y/hn_rel equal {same}; "
                  f"g sum {out['g'].sum():.3f} / {z['g'].sum():.3f}, r nan {np.isnan(out['r']).sum()} / {np.isnan(z['r']).sum()}", flush=True)
        else:
            print(f"  c1: {len(e):,} argmax rows, {out['ch'].mean():.3f} changed SX, r nan {np.isnan(out['r']).sum()}", flush=True)
        np.savez_compressed(OUT / f"loco_{src}_{tgt}_c{v}{tag}.npz", **out)
    print(f"frame {src}>{tgt}: {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)


RECIPES = {"gc70": dict(t=0.9868, cap=(5, 6), r=0.9, gc=0.7, hn=None),
           "sub42_both": dict(t=0.99, cap=(5, 6), r=0.97, gc=0.7, hn=0.8),
           "base_only": dict(t=0.9868, cap=(5, 6), r=None, gc=None, hn=None)}


def f_s1(z, p, tm, rc):
    import solution_p42 as J
    s1, n = z["s1"].astype(np.int64), len(z["ntrue"])
    am = np.ones(len(s1), bool)
    mp = lambda x: None if x is None else tm[str(x)]
    pred = J.decide(s1, z["is2"], p, am, z["hn_rel"] == 2, z["hn_rel"] == 1, n, mp(rc["t"]), rc["cap"], mp(rc["r"]), mp(rc["gc"]), mp(rc["hn"]))
    return J.fbeta(np.bincount(s1[pred & z["y"]], minlength=n), np.bincount(s1[pred], minlength=n), z["ntrue"].astype(np.int64)), pred


def evaluate(tag=""):
    import solution_p42 as J
    from ledger import log
    t0 = time.time()
    G = json.loads((W / "p42/sm/grid.json").read_text())
    res = {}
    for src, tgt in (("US", "India"), ("India", "US")):
        dr = f"{src}_{tgt}"
        z0 = dict(np.load(W / f"p42/loco_{dr}.npz"))
        z1 = dict(np.load(OUT / f"loco_{dr}_c1{tag}.npz"))
        n = len(z0["ntrue"])
        for an in ("chain", "m2"):
            p0, p1 = J.loco_p(z0["g"], z0["r"], 0.1, an), J.loco_p(z1["g"], z1["r"], 0.1, an)
            one = lambda z: np.ones(len(z["s1"]), bool)
            tm0 = J.fr_to_loco(G, an, dr, 0.1, p0, one(z0), n)
            tm1 = J.fr_to_loco(G, an, dr, 0.1, p1, one(z1), n)
            for rn, rc in RECIPES.items():
                f0, _ = f_s1(z0, p0, tm0, rc)
                for mode, tm in (("fixed", tm0), ("remap", tm1)):
                    f1, pr = f_s1(z1, p1, tm, rc)
                    df = f1 - f0
                    key = f"{dr}|{an}|{rn}|{mode}"
                    res[key] = {"F0": round(float(f0.mean()), 6), "dF": round(float(df.mean()), 6),
                                "se": round(float(df.std() / np.sqrt(n)), 6),
                                "dF_h0": round(float(df[z0["hv"] == 0].mean()), 6), "dF_h1": round(float(df[z0["hv"] == 1].mean()), 6),
                                "npred": int(pr.sum()), "tp": int((pr & z1["y"]).sum())}
                    print(f"  {key:40s} {res[key]}", flush=True)
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "score":
        score(sys.argv[2], sys.argv[3])
    elif cmd == "score_pair":
        score_pair(sys.argv[2], sys.argv[3])
    elif cmd == "frame":
        frame(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
    elif cmd == "eval":
        r = evaluate(sys.argv[2] if len(sys.argv) > 2 else "")
        (HERE / f"loco{sys.argv[2] if len(sys.argv) > 2 else ''}.json").write_text(json.dumps(r, indent=1))
