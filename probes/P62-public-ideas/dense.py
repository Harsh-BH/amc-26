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


def retrieve(c: str, k: int = 5, bs: int = 512):
    """e5 top-k S1 of every holdout-world SX of c that sub54 does not accept. Index = holdout-world S1 + non-holdout
    train S1 (confusers, no kids in the world) up to the test S1 count. -> data/work/p62/ret_<c>.parquet"""
    import time
    import torch
    from transformers import AutoModel, AutoTokenizer
    t0 = time.time()
    w1 = pd.read_parquet(E.W / "worlds/holdout/s1.parquet")
    w1 = w1[w1.country == c]
    fo = pd.read_parquet(E.W / "s1_folds.parquet")
    pool = fo[(fo.country == c) & (fo.fold != 0)].entity_id.sample(n=N_TEST[c] - len(w1), random_state=0)
    s1 = pd.read_parquet(E.W / "train_source1.parquet")
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
    out.to_parquet(OUT / f"ret_{c}.parquet", index=False)
    pd.Series(ids).to_frame("s1").assign(world=np.arange(len(ids)) < len(w1)).to_parquet(OUT / f"pool_{c}.parquet", index=False)
    print(c, f"done {time.time() - t0:.0f}s", flush=True)


def feats(c: str) -> pd.DataFrame:
    """Top-1 retrieval pair of every query SX + label (train truth, any fold) + cheap verifier features."""
    from rapidfuzz import fuzz
    r = pd.read_parquet(OUT / f"ret_{c}.parquet")
    pool = pd.read_parquet(OUT / f"pool_{c}.parquet")
    world = set(pool.s1[pool.world])
    gt = pq.read_table(E.W / "gt_pairs.parquet", columns=["s1", "sx"])
    gt = gt.filter(pc.is_in(gt["sx"], value_set=pa.array(r.sx))).to_pandas().drop_duplicates("sx").set_index("sx").s1
    r["owner"] = gt.reindex(r.sx).to_numpy()
    r["owner_in_pool"] = r.owner.isin(set(pool.s1))
    r["y"] = r.s1_0.to_numpy() == r.owner.to_numpy()
    r["hit5"] = np.any(np.stack([r[f"s1_{i}"].to_numpy() == r.owner.to_numpy() for i in range(5)]), axis=0)
    r["world"] = r.s1_0.isin(world)
    r["margin"] = r.cos_0 - r.cos_1
    r["hub"] = r.groupby("s1_0").sx.transform("size")          # query SX whose top-1 is this S1
    r["rank1"] = r.groupby("s1_0").cos_0.rank(ascending=False, method="first")
    m = pd.read_parquet(OUT / f"miss_{c}.parquet", columns=["sx", "bucket"]).drop_duplicates("sx").set_index("sx").bucket
    r["bucket"] = m.reindex(r.sx).fillna("none").to_numpy()
    d = pd.read_parquet(E.W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout"])
    key = lambda a, b: pd.MultiIndex.from_arrays([a, b])
    r["in_cand"] = key(r.s1_0, r.sx).isin(key(d.s1[d.holdout], d.sx[d.holdout]))
    del d
    f1 = pq.read_table(E.W / f"rf_train_s1_{c}.parquet", columns=["id", "core", "hn", "city", "state"])
    f1 = f1.filter(pc.is_in(f1["id"], value_set=pa.array(r.s1_0.unique()))).to_pandas().set_index("id").reindex(r.s1_0)
    fx = pq.read_table(E.W / f"rf_train_sx_{c}.parquet", columns=["id", "core", "hn", "city", "state", "is_web"])
    fx = fx.filter(pc.is_in(fx["id"], value_set=pa.array(r.sx))).to_pandas().drop_duplicates("id").set_index("id").reindex(r.sx)
    for k in ("hn", "city", "state"):
        a, b = f1[k].fillna("").to_numpy(object), fx[k].fillna("").to_numpy(object)
        r[f"{k}_eq"] = np.where((a != "") & (b != ""), (a == b).astype(np.float32), np.nan)
    a, b = f1.core.fillna("").tolist(), fx.core.fillna("").tolist()
    r["name_tsr"] = [fuzz.token_set_ratio(u, v) for u, v in zip(a, b)]
    r["name_r"] = [fuzz.ratio(u, v) for u, v in zip(a, b)]
    r["is_web"] = fx.is_web.fillna(False).to_numpy(np.float32)
    r["sx_s3"] = r.sx.str.startswith("S3-").to_numpy(np.float32)
    return r


VF = ["cos_0", "cos_1", "margin", "hub", "rank1", "hn_eq", "city_eq", "state_eq", "name_tsr", "name_r", "is_web", "sx_s3"]


def evalret():
    """Hit rates per miss bucket; cross-fit verifier (LightGBM, 2 folds by S1 half); rescue gate on the holdout halves."""
    import json
    import lightgbm as lgb
    import a10
    S = {c: pd.read_parquet(E.OUT / f"S1_{c}.parquet") for c in E.C}
    res = {}
    R = {}
    for c in E.C:
        r = feats(c)
        ids = pd.read_parquet(E.W / f"p23/s1ctx_{c}.parquet", columns=["n_true"]).index
        g = pd.Series(np.arange(len(ids)), index=ids)
        r["g"] = g.reindex(r.s1_0).fillna(-1).astype(int).to_numpy()
        hh = S[c].half.to_numpy()
        # S1 half: holdout S1 use P44's half; confusers a hash parity (only for the verifier's folds)
        r["half"] = np.where(r.g >= 0, hh[np.maximum(r.g, 0)], pd.util.hash_pandas_object(r.s1_0, index=False).to_numpy() % 2)
        st = {}
        for b in ("in_cand", "pruned", "blocked"):
            x = r[r.bucket == b]
            st[b] = {"n": len(x), "top1": float(x.y.mean()) if len(x) else None, "top5": float(x.hit5.mean()) if len(x) else None}
        st["queries"], st["owner_none"] = len(r), int(r.owner.isna().sum())
        st["top1_prec_all"] = float(r.y.mean())
        print(c, st, flush=True)
        res[c] = st
        R[c] = r
    X = pd.concat([R[c].assign(country=c) for c in E.C], ignore_index=True)
    X["country_in"] = (X.country == "India").astype(np.float32)   # ponytail: a 2-country flag; France would need its own look
    cols = VF + ["country_in"]
    X["v"] = np.nan
    prm = dict(objective="binary", num_leaves=31, learning_rate=0.05, min_data_in_leaf=100, verbose=-1, num_threads=8, seed=0)
    for h in (0, 1):
        tr = X.half.to_numpy() == h
        bst = lgb.train(prm, lgb.Dataset(X.loc[tr, cols], X.y[tr].astype(int)), 300)
        X.loc[~tr, "v"] = bst.predict(X.loc[~tr, cols])
    from sklearn.metrics import roc_auc_score
    res["verifier_auc"] = float(roc_auc_score(X.y, X.v))
    # rescue: top-1 pair not already a post-pruner candidate, onto a world (holdout) S1, verifier >= tau
    out = {}
    for tau in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
        o = {}
        for c in E.C:
            x = X[(X.country == c) & ~X.in_cand & (X.g >= 0) & (X.v >= tau)]
            D = pd.read_parquet(OUT / f"dec54_{c}.parquet", columns=["g", "y", "acc"])
            St = S[c]
            n, nt, hh = len(St), St.n_true.to_numpy(), St.half.to_numpy()
            tp0 = np.bincount(D.g[D.acc & D.y], minlength=n)
            np0 = np.bincount(D.g[D.acc], minlength=n)
            ga = x.g.to_numpy()
            d = a10.fb(tp0 + np.bincount(ga[x.y.to_numpy()], minlength=n), np0 + np.bincount(ga, minlength=n), nt) - a10.fb(tp0, np0, nt)
            o[c] = {"adds": len(x), "prec": float(x.y.mean()) if len(x) else None,
                    **{f"d_h{k}": float(d[hh == k].mean()) for k in (0, 1)},
                    **{f"se_h{k}": float(d[hh == k].std(ddof=1) / np.sqrt((hh == k).sum())) for k in (0, 1)}}
        for k in (0, 1):
            o[f"lb_h{k}"] = sum(E.LB_W[c] * o[c][f"d_h{k}"] for c in E.C)
            o[f"lbse_h{k}"] = float(np.sqrt(sum((E.LB_W[c] * o[c][f"se_h{k}"]) ** 2 for c in E.C)))
        out[tau] = o
        print("tau", tau, {c: (o[c]["adds"], round(o[c]["prec"] or 0, 3)) for c in E.C},
              "LB-eq h0 %+.6f ± %.6f  h1 %+.6f ± %.6f" % (o["lb_h0"], o["lbse_h0"], o["lb_h1"], o["lbse_h1"]), flush=True)
    res["rescue"] = out
    print("verifier AUC", res["verifier_auc"])
    X.drop(columns=[f"s1_{i}" for i in range(1, 5)]).to_parquet(OUT / "ret_eval.parquet", index=False)
    (OUT / "dense_eval.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    {"ceiling": ceiling, "retrieve": retrieve, "eval": evalret}[sys.argv[1]](*sys.argv[2:])
