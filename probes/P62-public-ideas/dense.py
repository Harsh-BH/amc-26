"""P62 idea 2 -- dense-retrieval rescue of owner-less SX (e5-small), on the #8c holdout world (P44 / P57 frame).
    .venv/bin/python probes/P62-public-ideas/dense.py ceiling     # what a perfect rescue of each miss bucket is worth
Buckets of a holdout S1's true SX that sub54's decision does not accept (argmax pairs, cross-fit s >= .71):
  stolen   another holdout S1 accepts it          in_cand  a post-pruner candidate row of the S1 exists (rejected /
  comp     a non-holdout S1 takes it (#8c am, p >= .71)       argmax elsewhere)
  pruned   in the blocking candidates, cut by the pruner      blocked  never a blocking candidate
"""
import os
import sys

os.environ["P44_VARIANT"] = "r"
sys.path[:0] = ["probes/P44-meta-stack", "probes/P50-idea-batch", "probes/P57-iu-transfer", "code/business_entity_resolution/src", "."]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import ev as E  # noqa: E402

OUT = E.W / "p62"


def decision(c: str) -> pd.DataFrame:
    """sub54 recipe on P57's full frame, cross-fit halves: (s1, sx, g, half, y, s, acc) of the argmax pairs."""
    import ev57 as V
    import features as F
    A = V.te44.derive(pd.read_parquet(V.OUT / f"A_full_{c}.parquet"))
    for m in V.Z.MEM:
        A[f"z_{m}"] = V.MT.lg(A[f"p_{m}"].to_numpy())
    for m in V.Z.RANK:
        A[f"z_{m}"] = A[f"p_{m}"].to_numpy(np.float64)
    cols, _ = V.ARMS["sub54"]
    s = np.empty(len(A))
    for fh in (0, 1):
        m = A.half.to_numpy() == 1 - fh
        s[m] = V.score(V.boosters("sub54", fh), A[m], cols)
    id1, idx = F.ids("train", c)
    out = pd.DataFrame({"s1": id1.take(pa.array(A.q1.to_numpy())).to_numpy(zero_copy_only=False),
                        "sx": idx.take(pa.array(A.qx.to_numpy())).to_numpy(zero_copy_only=False),
                        "g": A.g.to_numpy(), "half": A.half.to_numpy(), "y": A.y.to_numpy(), "s": s})
    out["acc"] = out.s >= E.T
    return out


def buckets(c: str, D: pd.DataFrame) -> pd.DataFrame:
    """Every true SX of a holdout S1 that D does not accept, with its bucket."""
    ids = pd.read_parquet(E.W / f"p23/s1ctx_{c}.parquet", columns=["n_true"]).index
    gt = pq.read_table(E.W / "gt_pairs.parquet", columns=["s1", "sx"])
    gt = gt.filter(pc.is_in(gt["s1"], value_set=pa.array(ids))).to_pandas()
    key = lambda d: pd.MultiIndex.from_arrays([d.s1, d.sx])
    acc = D[D.acc]
    k = gt[~key(gt).isin(key(acc))].copy()
    k["stolen"] = k.sx.isin(set(acc.sx))
    d = pd.read_parquet(E.W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "p", "am"])
    comp = set(d.sx[~d.holdout & d.am & (d.p >= E.T)])
    k["comp"] = k.sx.isin(comp) & ~k.stolen
    k["in_cand"] = key(k).isin(key(d[d.holdout]))
    b = pd.read_parquet(E.W / f"p3/cands_holdout_full_{c}.parquet", columns=["s1", "sx"])
    k["blocked_in"] = key(k).isin(key(b))
    k["bucket"] = np.select([k.stolen, k.comp, k.in_cand, k.blocked_in], ["stolen", "comp", "in_cand", "pruned"], "blocked")
    k["g"] = pd.Series(np.arange(len(ids)), index=ids).reindex(k.s1).to_numpy()
    return k


def ceiling():
    import a10
    S = {c: pd.read_parquet(E.OUT / f"S1_{c}.parquet") for c in E.C}
    res = {}
    for c in E.C:
        D = decision(c)
        D.to_parquet(OUT / f"dec54_{c}.parquet", index=False)
        k = buckets(c, D)
        k.to_parquet(OUT / f"miss_{c}.parquet", index=False)
        St = S[c]
        n, nt, h = len(St), St.n_true.to_numpy(), St.half.to_numpy()
        tp0 = np.bincount(D.g[D.acc & D.y], minlength=n)
        np0 = np.bincount(D.g[D.acc], minlength=n)
        f0 = a10.fb(tp0, np0, nt)
        r = {"miss": len(k), **{f"n_{b}": int((k.bucket == b).sum()) for b in ("stolen", "comp", "in_cand", "pruned", "blocked")}}
        for name, bs in (("pruned+blocked", ("pruned", "blocked")), ("in_cand", ("in_cand",)),
                         ("all_free", ("in_cand", "pruned", "blocked"))):
            a = np.bincount(k.g[k.bucket.isin(bs)].astype(int), minlength=n)
            d = a10.fb(tp0 + a, np0 + a, nt) - f0
            r[name] = [float(d[h == x].mean()) for x in (0, 1)]
        res[c] = r
        print(c, r, flush=True)
    for name in ("pruned+blocked", "in_cand", "all_free"):
        print(name, "LB-eq ceiling h0/h1", [round(sum(E.LB_W[c] * res[c][name][x] for c in E.C), 6) for x in (0, 1)])
    return res


N_TEST = {"India": 809_986, "US": 663_106}              # test S1 per country: the index is padded to this size
E5 = "intfloat/multilingual-e5-small"                     # MIT, 118M; frozen, P45's recipe ("query: ", mean pool, L2)


def text(d: pd.DataFrame) -> list:
    a = d.business_address.str.replace(r"<\s*NULL\s*>|\bNULL\b", "", regex=True, case=False).str.strip(" ,")
    return ("query: " + d.business_name + ", " + a).tolist()


def retrieve(c: str, k: int = 5, bs: int = 512, tag: str = ""):
    """e5 top-k S1 of every holdout-world SX of c that sub54 does not accept. Index = holdout-world S1 + non-holdout
    train S1 (confusers, no kids in the world) up to the test S1 count. -> data/work/p62/ret_<c>.parquet"""
    import time
    import torch
    from transformers import AutoModel, AutoTokenizer
    k, bs = int(k), int(bs)
    t0 = time.time()
    w1 = pd.read_parquet(E.W / "worlds/holdout/s1.parquet")
    w1 = w1[w1.country == c]
    fo = pd.read_parquet(E.W / "s1_folds.parquet")
    pool = fo[(fo.country == c) & (fo.fold != 0)].entity_id.sample(n=N_TEST[c] - len(w1), random_state=0)
    s1 = pd.read_parquet(E.W / "train_source1.parquet", filters=[("country", "==", c)])
    s1 = pd.concat([w1, s1[s1.entity_id.isin(set(pool))]], ignore_index=True)
    wx = pd.read_parquet(E.W / "worlds/holdout/sx.parquet")
    wx = wx[wx.country == c]
    D = pd.read_parquet(OUT / f"dec54_{c}.parquet", columns=["sx", "acc"])
    q = wx[~wx.entity_id.isin(set(D.sx[D.acc]))].reset_index(drop=True)
    print(c, f"index {len(s1):,} S1 ({len(w1):,} holdout), queries {len(q):,} of {len(wx):,} world SX", flush=True)
    tok = AutoTokenizer.from_pretrained(E5)
    net = AutoModel.from_pretrained(E5).to("cuda").half().eval()

    def enc(tx):
        o = np.argsort([len(x) for x in tx], kind="stable")
        out = torch.empty((len(tx), net.config.hidden_size), dtype=torch.float16, device="cuda")
        with torch.no_grad():
            for a in range(0, len(o), bs):
                j = o[a:a + bs]
                b = tok([tx[i] for i in j], padding=True, truncation=True, max_length=64, return_tensors="pt").to("cuda")
                h = net(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                e = (h * m).sum(1) / m.sum(1).clamp(min=1)
                out[torch.as_tensor(j, device="cuda")] = torch.nn.functional.normalize(e.float(), dim=1).half()
        return out
    X1 = enc(text(s1))
    print(c, f"index encoded {time.time() - t0:.0f}s", flush=True)
    tq = text(q)
    top_i, top_s = np.empty((len(q), k), np.int32), np.empty((len(q), k), np.float16)
    CH = 65536
    for a in range(0, len(q), CH):
        Q = enc(tq[a:a + CH])
        for b in range(0, len(Q), 512):                    # 512 x 810k fp16 scores = 0.8 GB
            v, i = torch.topk(Q[b:b + 512] @ X1.T, k, dim=1)
            top_i[a + b:a + b + len(v)], top_s[a + b:a + b + len(v)] = i.cpu().numpy(), v.float().cpu().numpy()
        print(c, f"queries {min(a + CH, len(q)):,} {time.time() - t0:.0f}s", flush=True)
    ids = s1.entity_id.to_numpy(object)
    out = pd.DataFrame({"sx": q.entity_id.to_numpy()})
    for r in range(k):
        out[f"s1_{r}"], out[f"cos_{r}"] = ids[top_i[:, r]], top_s[:, r].astype(np.float32)
    out.to_parquet(OUT / f"ret{tag}_{c}.parquet", index=False)
    pd.Series(ids).to_frame("s1").assign(world=np.arange(len(ids)) < len(w1)).to_parquet(OUT / f"pool_{c}.parquet", index=False)
    print(c, f"done {time.time() - t0:.0f}s", flush=True)


def feats(c: str) -> pd.DataFrame:
    """Top-K retrieval pairs of every query SX (long, integer codes) + label (train truth, any fold) + verifier features.
    Test-like distractors: the owner of a world distractor (an SX of no holdout S1) leaves the index, so every distractor
    is owner-less as on test; ranks / cosine context are re-derived on what is left. Ranks >= 5 are kept only with
    address evidence (same hn or city), to bound RAM."""
    from rapidfuzz import fuzz
    r = pd.read_parquet(OUT / f"ret{TAG}_{c}.parquet")
    pool = pd.read_parquet(OUT / f"pool_{c}.parquet")
    K = sum(1 for x in r.columns if x.startswith("s1_"))
    P, nW, nq = pd.Index(pool.s1), int(pool.world.sum()), len(r)
    assert pool.world.to_numpy()[:nW].all()
    S1c = np.stack([P.get_indexer(r[f"s1_{k}"]) for k in range(K)], 1)
    COS = np.stack([r[f"cos_{k}"].to_numpy(np.float32) for k in range(K)], 1)
    gt = pq.read_table(E.W / "gt_pairs.parquet", columns=["s1", "sx"])
    gt = gt.filter(pc.is_in(gt["sx"], value_set=pa.array(r.sx))).to_pandas().drop_duplicates("sx").set_index("sx").s1
    oc = P.get_indexer(gt.reindex(r.sx).fillna("").to_numpy(object))          # owner's pool code, -1 = none
    excl = np.unique(oc[oc >= nW])
    COS[np.isin(S1c, excl)] = -np.inf
    o = np.argsort(-COS, axis=1, kind="stable")
    S1c, COS = np.take_along_axis(S1c, o, 1), np.take_along_axis(COS, o, 1)
    print(c, f"index minus {len(excl):,} distractor owners -> {len(P) - len(excl):,} S1", flush=True)
    nxt = np.c_[COS[:, 1:], np.full(nq, np.nan, np.float32)]
    qi, rk = np.nonzero(np.isfinite(COS))
    s1c, cos = S1c[qi, rk], COS[qi, rk]
    f1 = pq.read_table(E.W / f"rf_train_s1_{c}.parquet", columns=["id", "core", "hn", "city", "state"])
    f1 = f1.filter(pc.is_in(f1["id"], value_set=pa.array(P))).to_pandas().set_index("id").reindex(P).fillna("")
    fx = pq.read_table(E.W / f"rf_train_sx_{c}.parquet", columns=["id", "core", "hn", "city", "state", "is_web"])
    fx = fx.filter(pc.is_in(fx["id"], value_set=pa.array(r.sx))).to_pandas().drop_duplicates("id").set_index("id").reindex(r.sx)
    eq = {}
    for k in ("hn", "city", "state"):
        a, b = f1[k].to_numpy(object)[s1c], fx[k].fillna("").to_numpy(object)[qi]
        eq[k] = np.where((a != "") & (b != ""), (a == b).astype(np.float32), np.nan)
    keep = (rk < 5) | (eq["hn"] == 1) | (eq["city"] == 1)
    qi, rk, s1c, cos = qi[keep], rk[keep], s1c[keep], cos[keep]
    X = pd.DataFrame({"qi": qi.astype(np.int32), "s1c": s1c.astype(np.int32), "rk": rk.astype(np.int8), "cos": cos,
                      "cos_0": COS[qi, 0], "gap0": COS[qi, 0] - cos, "gapn": cos - nxt[qi, rk],
                      "margin01": COS[qi, 0] - np.where(np.isfinite(COS[qi, 1]), COS[qi, 1], 0)})
    for k in eq:
        X[f"{k}_eq"] = eq[k][keep]
    del eq
    X["y"] = s1c == oc[qi]
    X["world"] = s1c < nW
    X["hub"] = np.bincount(s1c, minlength=len(P))[s1c]
    X["hub1"] = np.bincount(S1c[:, 0][np.isfinite(COS[:, 0])], minlength=len(P))[s1c]
    a, b = f1.core.to_numpy(object)[s1c], fx.core.fillna("").to_numpy(object)[qi]
    X["name_tsr"] = np.array([fuzz.token_set_ratio(u, v) for u, v in zip(a, b)], np.float32)
    X["name_r"] = np.array([fuzz.ratio(u, v) for u, v in zip(a, b)], np.float32)
    del a, b
    X["is_web"] = fx.is_web.fillna(False).to_numpy(np.float32)[qi]
    X["sx_s3"] = r.sx.str.startswith("S3-").to_numpy(np.float32)[qi]
    m = pd.read_parquet(OUT / f"miss_{c}.parquet", columns=["sx", "bucket"]).drop_duplicates("sx").set_index("sx").bucket
    X["bucket"] = pd.Categorical(m.reindex(r.sx).fillna("none").to_numpy()[qi])
    d = pd.read_parquet(E.W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout"])
    d = d[d.holdout]
    k1, kx = P.get_indexer(d.s1), pd.Index(r.sx).get_indexer(d.sx)
    ok = (k1 >= 0) & (kx >= 0)
    X["in_cand"] = np.isin(X.s1c.to_numpy(np.int64) * nq + X.qi.to_numpy(), k1[ok].astype(np.int64) * nq + kx[ok])
    ids = pd.read_parquet(E.W / f"p23/s1ctx_{c}.parquet", columns=["n_true"]).index
    gw = pd.Series(np.arange(len(ids)), index=ids).reindex(P[:nW]).fillna(-1).astype(int).to_numpy()
    X["g"] = np.where(X.world, gw[np.minimum(X.s1c.to_numpy(), nW - 1)], -1)
    X["sx"] = X.qi                                               # per-country SX key (best-candidate pick)
    X["s1_hash"] = pd.util.hash_pandas_object(pd.Series(P), index=False).to_numpy()[X.s1c.to_numpy()] % 2
    return X


TAG = os.environ.get("DENSE_TAG", "")                   # "" = the k=5 retrieval, "20" = k=20
VF = ["rk", "cos", "gap0", "gapn", "cos_0", "margin01", "hub", "hub1", "hn_eq", "city_eq", "state_eq", "name_tsr", "name_r",
      "is_web", "sx_s3"]


def evalret(mode: str = "half", tag: str = ""):
    """Hit rates per miss bucket; cross-fit verifier over the top-5 (LightGBM, 2 folds by S1 half); per SX the best-v
    candidate is rescued at v >= tau. Gate accounting: TP / FP on holdout S1 as measured, plus every FP that lands on a
    confuser S1 charged to a random holdout S1 of the same half (the world's distractors are sized to the holdout S1)."""
    import json
    import lightgbm as lgb
    import a10
    S = {c: pd.read_parquet(E.OUT / f"S1_{c}.parquet") for c in E.C}
    res, R = {}, {}
    for c in E.C:
        X = feats(c)
        hh = S[c].half.to_numpy()
        X["half"] = np.where(X.g >= 0, hh[np.maximum(X.g.to_numpy(), 0)], X.s1_hash)
        t = X[X.rk == 0]
        st = {"queries": int(len(t)), "top1_prec": float(t.y.mean())}
        for b in ("in_cand", "pruned", "blocked"):
            q = X[X.bucket == b]
            st[b] = {"n": int((q.rk == 0).sum()), "top1": float(q[q.rk == 0].y.mean()),
                     "top5": float(q.groupby("sx").y.any().mean())}
        print(c, st, flush=True)
        res[c] = st
        R[c] = X.assign(country=c)
    X = pd.concat(R.values(), ignore_index=True)
    del R
    X["country_in"] = (X.country == "India").astype(np.float32)   # ponytail: a 2-country flag; France would need its own look
    cols = VF + ["country_in"]
    X["v"] = np.nan
    prm = dict(objective="binary", num_leaves=63, learning_rate=0.05, min_data_in_leaf=200, verbose=-1, num_threads=8, seed=0)
    if mode == "loco":                                   # fit on the other country only (France proxy)
        cols = VF
        for c in E.C:
            tr = X.country.to_numpy() != c
            bst = lgb.train(prm, lgb.Dataset(X.loc[tr, cols], X.y[tr].astype(int)), 300)
            X.loc[~tr, "v"] = bst.predict(X.loc[~tr, cols])
    else:
        for h in (0, 1):
            tr = X.half.to_numpy() == h
            bst = lgb.train(prm, lgb.Dataset(X.loc[tr, cols], X.y[tr].astype(int)), 300)
            X.loc[~tr, "v"] = bst.predict(X.loc[~tr, cols])
    from sklearn.metrics import roc_auc_score
    res["verifier_auc"] = float(roc_auc_score(X.y, X.v))
    B = X.sort_values("v", ascending=False).drop_duplicates(["country", "sx"])   # best candidate per SX
    B = B[~B.in_cand]
    rng = np.random.default_rng(0)
    out = {}
    for tau in (0.7, 0.8, 0.85, 0.9, 0.95, 0.98):
        o = {}
        for c in E.C:
            x = B[(B.country == c) & (B.v >= tau)]
            D = pd.read_parquet(OUT / f"dec54_{c}.parquet", columns=["g", "y", "acc"])
            St = S[c]
            n, nt, hh = len(St), St.n_true.to_numpy(), St.half.to_numpy()
            tp0 = np.bincount(D.g[D.acc & D.y], minlength=n)
            np0 = np.bincount(D.g[D.acc], minlength=n)
            w = x[x.g >= 0]
            fc = x[(x.g < 0) & ~x.y]                                          # FPs on confusers -> random holdout S1
            gf = np.concatenate([rng.choice(np.flatnonzero(hh == k), (fc.half == k).sum()) for k in (0, 1)]) \
                if len(fc) else np.zeros(0, int)
            add = np.bincount(np.r_[w.g.to_numpy(), gf], minlength=n)
            tpa = np.bincount(w.g[w.y].to_numpy(), minlength=n)
            d = a10.fb(tp0 + tpa, np0 + add, nt) - a10.fb(tp0, np0, nt)
            o[c] = {"adds_all": int(len(x)), "prec_all": float(x.y.mean()) if len(x) else None,
                    "adds_holdout": int(len(w)), "tp_holdout": int(w.y.sum()), "fp_confuser_charged": int(len(fc)),
                    **{f"d_h{k}": float(d[hh == k].mean()) for k in (0, 1)},
                    **{f"se_h{k}": float(d[hh == k].std(ddof=1) / np.sqrt((hh == k).sum())) for k in (0, 1)}}
        for k in (0, 1):
            o[f"lb_h{k}"] = sum(E.LB_W[c] * o[c][f"d_h{k}"] for c in E.C)
            o[f"lbse_h{k}"] = float(np.sqrt(sum((E.LB_W[c] * o[c][f"se_h{k}"]) ** 2 for c in E.C)))
        out[tau] = o
        print("tau", tau, {c: (o[c]["adds_all"], round(o[c]["prec_all"] or 0, 3), o[c]["adds_holdout"], o[c]["tp_holdout"],
                                o[c]["fp_confuser_charged"]) for c in E.C},
              "LB-eq h0 %+.6f ± %.6f  h1 %+.6f ± %.6f" % (o["lb_h0"], o["lbse_h0"], o["lb_h1"], o["lbse_h1"]), flush=True)
    res["rescue"] = out
    print("verifier AUC", res["verifier_auc"])
    B.to_parquet(OUT / f"ret_best_{mode}{TAG}.parquet", index=False)
    (OUT / f"dense_eval_{mode}{TAG}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    {"ceiling": ceiling, "retrieve": retrieve, "eval": evalret}[sys.argv[1]](*sys.argv[2:])
