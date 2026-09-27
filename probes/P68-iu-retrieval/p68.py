"""P68: India/US retrieval misses -- addressed SX whose true S1 P3 never lists, recovered as an ADD-ONLY tier.

P3 keeps an addressed SX's top-2 S1 of one name+address search (50/50). P23's tags on the holdout's blocking misses:
the true S1 sits at rank 2-4 of that search (chains: US 310 / India 320 pairs), or the SX was renamed (synthetic brand,
website, other name) so the name half scores 0 while an address-only search finds it (US 60% within top-5).
New candidates for addressed SX:
  R1  P3's own "na" view, SX rank 2-4          R2  address-only view, SX rank 0-4
They get the exact 65 features of features.feat_step (S1 / SX context over existing P3 pairs + the new pairs), P10's
pruner q and m2_big's p. Decision (add-only; the current decision is never touched): an SX that nothing accepted
joins its best new pair's S1 when p >= t_add (caps 5 S2 / 6 S3). t_add is picked on holdout half 0, reported on half 1.

    .venv/bin/python probes/P68-iu-retrieval/p68.py selftest
    .venv/bin/python probes/P68-iu-retrieval/p68.py search train US     # holdout (fold 0) S1 only -> hits
    .venv/bin/python probes/P68-iu-retrieval/p68.py feats train US      # new pairs -> features, q, p
    .venv/bin/python probes/P68-iu-retrieval/p68.py gate US India
    .venv/bin/python probes/P68-iu-retrieval/p68.py search test US ; ... feats test US ; build BASE OUT T
"""
import csv
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import block as B  # noqa: E402
import features as F  # noqa: E402
import model as M  # noqa: E402
import prune as PR  # noqa: E402

W = ROOT / "data/work"
OUT = W / "p68"
PROBE = "P68-iu-retrieval"   # was P67-iu-retrieval until 27 Sep 14:20 (ledger rows before that)
CFG = dict(B.CFG, rev={"na": 5, "addr": 5}, k_noaddr=0, k_nostate=5, threads=int(os.environ.get("P68_THREADS", 15)))
T_BASE, CAP = 0.71, {"S2": 5, "S3": 6}
SHARE = {"India": 0.468, "US": 0.383}                     # test S1 shares (LB-eq)
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
key = lambda a, b: (np.asarray(a, np.int64) << 32) | np.asarray(b, np.int64)


def holdout_ids(c):
    s = pd.read_parquet(W / "worlds/holdout/s1.parquet", columns=["entity_id", "country"])
    return set(s.entity_id[s.country == c])


# ---------------------------------------------------------------- 1. search (addressed SX only)
def search(split, c):
    t0 = time.time()
    txt = B.TXT[split]
    s1 = pd.read_parquet(txt / f"{c}_s1.parquet", columns=["entity_id", "name", "addr", "st"])
    keep_rev = s1.entity_id.isin(holdout_ids(c)).to_numpy() if split == "train" and not os.environ.get("P68_ALL_S1") else None
    only_qx = np.sort(pd.read_parquet(os.environ["P68_ONLY_QX"]).qx.to_numpy()) if os.environ.get("P68_ONLY_QX") else None
    vecs = B.vecs()
    ix = B.index(s1[["name", "addr", "st"]], vecs, CFG)
    del s1
    files, offs = F.sx_files(split, c)
    out = []
    owned_ids = None                                      # test: only SX that BASE assigns to nobody (the add-only tier)
    if os.environ.get("P68_FREE_BASE"):
        own = set()
        with open(Path(os.environ["P68_FREE_BASE"]) / "matching_results.tsv", encoding="utf-8") as fh:
            next(fh)
            for line in fh:
                v = line.rstrip("\n").split("\t", 1)[1]
                if v:
                    own.update(v.split(","))
        owned_ids = own
    pick = os.environ.get("P68_FILES")                    # holdout gate on a file sample: "0,2,4" = those SX files
    for k_, (f, off) in enumerate(zip(files, offs)):
        if pick and str(k_) not in pick.split(","):
            continue
        tb = pd.read_parquet(f, columns=["entity_id", "name", "addr", "st"])
        ok = (tb.addr != "").to_numpy()
        if owned_ids is not None:
            ok &= ~tb.entity_id.isin(owned_ids).to_numpy()   # skip SX the base already assigns
        tb = tb.drop(columns="entity_id")
        if only_qx is not None:                            # gate pass 2: only these SX (global rows)
            ok &= np.isin(np.arange(off, off + len(tb)), only_qx)
        rows = np.flatnonzero(ok)
        if os.environ.get("P68_SMOKE"):                     # smoke: the first N addressed SX of the first file only
            rows = rows[:int(os.environ["P68_SMOKE"])]
        for a in range(0, len(rows), CFG["shard"]):
            r = rows[a:a + CFG["shard"]]
            h = B.search(ix, tb.iloc[r].reset_index(drop=True), vecs, CFG, keep_rev)
            h = h[(h["dir"] == "r").to_numpy()]
            v, rk = h.view.astype(str).to_numpy(), h["rank"].to_numpy()
            m = ((v == "na") & (rk >= 2)) | ((v == "addr") & (rk <= 4))
            out.append(pd.DataFrame({"q1": h.q1.to_numpy(np.int32)[m], "qx": (off + r[h.qx.to_numpy()[m]]).astype(np.int32),
                                     "score": h.score.to_numpy(np.float32)[m], "rank": rk[m].astype(np.int8),
                                     "view": pd.Categorical(v[m], ["na", "addr"])}))
        print(f"  {f.name}: {len(rows):,} addressed SX, {sum(map(len, out)):,} hits so far, {time.time() - t0:.0f}s, "
              f"peak {peak():.2f} GB", flush=True)
        if os.environ.get("P68_SMOKE"):
            break
    h = pd.concat(out, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    h.to_parquet(OUT / f"hits_{split}{os.environ.get('P68_TAG', '')}_{c}.parquet", index=False)
    return {"hits": len(h), "by_view": h.view.value_counts().to_dict(), "runtime_s": round(time.time() - t0), "peak_gb": round(peak(), 2)}


# ---------------------------------------------------------------- 2. features of the new pairs
def _texts_rows(split, c, q1, qx):
    """P3 texts (name, addr) of S1 rows q1 and SX rows qx (global P3 row numbers), row-aligned."""
    txt = B.TXT[split]
    s1 = pd.read_parquet(txt / f"{c}_s1.parquet", columns=["name", "addr", "st"]).iloc[q1].reset_index(drop=True)
    files, offs = F.sx_files(split, c)
    parts = []
    for f, a, b in zip(files, offs[:-1], offs[1:]):
        m = (qx >= a) & (qx < b)
        if m.any():
            t = pd.read_parquet(f, columns=["name", "addr", "st"])
            parts.append(pd.DataFrame({"pos": np.flatnonzero(m)}).join(t.iloc[qx[m] - a].reset_index(drop=True)))
    sx = pd.concat(parts).sort_values("pos").reset_index(drop=True).drop(columns="pos")
    return s1, sx


def feats(split, c):
    t0 = time.time()
    tag = os.environ.get("P68_TAG", "")
    h = pd.read_parquet(OUT / f"hits_{split}{tag}_{c}.parquet")
    k = key(h.q1, h.qx)
    src = pd.DataFrame({"k": k, "r1": (h.view == "na").to_numpy(), "r2": (h.view == "addr").to_numpy()}).groupby("k").max()
    # existing pairs of the X files (P3 blocked + rule-only): the new pairs are the rest
    ex_k, ex_blk = [], []
    q1set = np.unique(h.q1.to_numpy())
    for p in F.xfiles(split, c):
        t = pq.read_table(p, columns=["q1", "qx", "block_score", "blocked"])
        kk = key(t.column("q1").to_numpy(), t.column("qx").to_numpy())
        ex_k.append(kk)
        blk = t.column("blocked").to_numpy() == 1
        keep = blk & np.isin(t.column("q1").to_numpy(), q1set)          # S1 context rows
        ex_blk.append(pd.DataFrame({"q1": t.column("q1").to_numpy()[keep], "qx": t.column("qx").to_numpy()[keep],
                                    "block_score": t.column("block_score").to_numpy()[keep]}))
        del t, kk
    ex_k = np.concatenate(ex_k)
    new = src.index.to_numpy()[~np.isin(src.index.to_numpy(), ex_k)]
    n = pd.DataFrame({"q1": (new >> 32).astype(np.int32), "qx": (new & 0xFFFFFFFF).astype(np.int32)})
    n = n.join(src.loc[new].reset_index(drop=True))
    print(f"{split} {c}: hits {len(h):,}, unique pairs {len(src):,}, NEW (not in X) {len(n):,} "
          f"(R1 {int(n.r1.sum()):,} / R2 {int(n.r2.sum()):,}), {time.time() - t0:.0f}s", flush=True)
    # block_score of a new pair = its P3 "na" cosine (the score P3 would have given it)
    vecs = B.vecs()
    t1, tx = _texts_rows(split, c, n.q1.to_numpy(), n.qx.to_numpy())
    A1, AX = B.mats(t1, {"na"}, vecs, CFG)["na"], B.mats(tx, {"na"}, vecs, CFG)["na"]
    n["block_score"] = np.asarray(A1.multiply(AX).sum(1)).ravel().astype(np.float32)
    del A1, AX
    # S1 / SX context over existing blocked pairs + the new pairs (features.feat_step's definitions)
    qxset = np.unique(n.qx.to_numpy())
    ex_sx = []
    for p in F.xfiles(split, c):
        t = pq.read_table(p, columns=["q1", "qx", "block_score", "blocked"])
        m = (t.column("blocked").to_numpy() == 1) & np.isin(t.column("qx").to_numpy(), qxset)
        ex_sx.append(pd.DataFrame({"q1": t.column("q1").to_numpy()[m], "qx": t.column("qx").to_numpy()[m],
                                   "block_score": t.column("block_score").to_numpy()[m]}))
    ctx = {}
    for side, ex in (("s1", pd.concat(ex_blk, ignore_index=True)), ("sx", pd.concat(ex_sx, ignore_index=True))):
        g = "q1" if side == "s1" else "qx"
        u = pd.concat([ex.assign(new=False), n[["q1", "qx", "block_score"]].assign(new=True)], ignore_index=True)
        cx = F._group_ctx(u[g].to_numpy(), np.nan_to_num(u.block_score.to_numpy(np.float32)), side)
        o = np.lexsort((-u.block_score.to_numpy(), u[g].to_numpy()))
        rr = np.empty(len(u), np.int32)
        rr[o] = B.ranks(u[g].to_numpy()[o])
        nm = u.new.to_numpy()
        ctx |= {kk: v[nm] for kk, v in cx.items()}
        ctx[f"block_rank_{side}"] = rr[nm]
    X = pd.DataFrame({"q1": n.q1, "qx": n.qx, **ctx, "block_score": n.block_score})
    X["via_r"], X["via_f"], X["blocked"] = np.float32(1), np.float32(0), np.float32(1)
    # string features: the S1 pool and the involved SX as features.rec tables
    r1 = F.s1_table(split, c)
    m1, counts = F.mats(r1), F.pool_counts(r1)
    idx = F.ids(split, c)[1]
    sx_ids = idx.take(pa.array(qxset))
    rf = pq.read_table(W / f"rf_{split}_sx_{c}.parquet")
    rf = rf.filter(pc.is_in(rf["id"], value_set=sx_ids)).to_pandas().drop_duplicates("id").set_index("id").reindex(sx_ids.to_pylist()).reset_index()
    _, txs = _texts_rows(split, c, np.zeros(0, np.int64), qxset)
    raw = []
    for i in (2, 3):
        t = pq.read_table(W / f"{split}_source{i}.parquet", columns=["entity_id", "business_name"])
        raw.append(t.filter(pc.is_in(t["entity_id"], value_set=sx_ids)).to_pandas())
    raw = pd.concat(raw).drop_duplicates("entity_id").set_index("entity_id").business_name.reindex(sx_ids.to_pylist())
    rx = F.assemble(rf[F.RF], txs, raw.reset_index(drop=True))
    mx = F.mats(rx)
    j = pd.Index(qxset).get_indexer(n.qx.to_numpy())
    S = pd.concat([F.strings(r1, rx, n.q1.to_numpy()[a:a + 200_000], j[a:a + 200_000], counts, m1, mx)
                   for a in range(0, len(n), 200_000)], ignore_index=True)
    S["rule_acc"] = np.float32(0)
    X = pd.concat([X.reset_index(drop=True), S.round(3)], axis=1)
    for col in X.columns:
        if col not in ("q1", "qx"):
            X[col] = X[col].astype(np.float32).round(3)
    X["q"] = PR.score(PR.load_model(), X)
    X["p"] = M.predict(M.load(), X)
    X["r1"], X["r2"] = n.r1.to_numpy(), n.r2.to_numpy()
    if split == "train":
        ids1 = F.ids(split, c)[0]
        gt = pq.read_table(W / "gt_pairs.parquet")
        gk = key(pc.index_in(gt["s1"], value_set=ids1).to_numpy(zero_copy_only=False).astype(np.float64),
                 pc.index_in(gt["sx"], value_set=idx).to_numpy(zero_copy_only=False).astype(np.float64))
        X["is_true"] = np.isin(key(X.q1, X.qx), gk)
    keep = ["q1", "qx", "q", "p", "r1", "r2", "block_score", "block_rank_sx", "name_cos", "addr_cos", "hn_eq", "core_eq"] + (
        ["is_true"] if split == "train" else [])
    X[keep].to_parquet(OUT / f"new_{split}{tag}_{c}.parquet", index=False)
    res = {"new": len(X), "r1": int(X.r1.sum()), "r2": int(X.r2.sum()), "q_ge_tau": int((X.q >= PR.TAU).sum()),
           "p_ge_071": int((X.p >= T_BASE).sum()), "runtime_s": round(time.time() - t0), "peak_gb": round(peak(), 2)}
    if split == "train":
        res |= {"true": int(X.is_true.sum()), "true_p_ge_071": int((X.is_true & (X.p >= T_BASE)).sum())}
    print(json.dumps(res), flush=True)
    return res


def keep_hits(h: pd.DataFrame) -> np.ndarray:
    """P68_KEEP='na:2-3,addr:0-2' restricts hits by view and SX rank (the gate picks it); default: every hit."""
    spec = os.environ.get("P68_KEEP")
    if not spec:
        return np.ones(len(h), bool)
    v, rk, m = h.view.astype(str).to_numpy(), h["rank"].to_numpy(), np.zeros(len(h), bool)
    for part in spec.split(","):
        name, rng = part.split(":")
        lo, hi = map(int, rng.split("-"))
        m |= (v == name) & (rk >= lo) & (rk <= hi)
    return m


def feats_c(split, c, chunk=300_000):
    """feats() with bounded RAM for the (larger) test side: block_score in chunks, context over the union, then string
    features + q + p per chunk of SX. Same columns and definitions as feats()."""
    t0 = time.time()
    h = pd.read_parquet(OUT / f"hits_{split}_{c}.parquet")
    h = h[keep_hits(h)]
    k = key(h.q1, h.qx)
    src = pd.DataFrame({"k": k, "r1": (h.view == "na").to_numpy(), "r2": (h.view == "addr").to_numpy()}).groupby("k").max()
    del h, k
    ks = src.index.to_numpy()
    q1set = np.unique((ks >> 32).astype(np.int32))
    qxset = np.unique((ks & 0xFFFFFFFF).astype(np.int32))
    exist, ex1, exx = np.zeros(len(ks), bool), [], []
    for p in F.xfiles(split, c):
        t = pq.read_table(p, columns=["q1", "qx", "block_score", "blocked"])
        a1, ax = t.column("q1").to_numpy(), t.column("qx").to_numpy()
        kk = key(a1, ax)
        exist |= np.isin(ks, kk)
        blk = t.column("blocked").to_numpy() == 1
        bs = t.column("block_score").to_numpy()
        m1, mx_ = blk & np.isin(a1, q1set), blk & np.isin(ax, qxset)
        ex1.append(pd.DataFrame({"q1": a1[m1], "qx": ax[m1], "block_score": bs[m1]}))
        exx.append(pd.DataFrame({"q1": a1[mx_], "qx": ax[mx_], "block_score": bs[mx_]}))
        del t, kk, a1, ax
    new = ks[~exist]
    n = pd.DataFrame({"q1": (new >> 32).astype(np.int32), "qx": (new & 0xFFFFFFFF).astype(np.int32)})
    n = n.join(src.loc[new].reset_index(drop=True)).sort_values(["qx", "q1"]).reset_index(drop=True)
    del src, ks, exist
    print(f"{split} {c}: NEW pairs {len(n):,} (R1 {int(n.r1.sum()):,} / R2 {int(n.r2.sum()):,}), {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)
    # pass 1: block_score = P3 "na" cosine
    vecs, txt = B.vecs(), B.TXT[split]
    s1t = pd.read_parquet(txt / f"{c}_s1.parquet", columns=["name", "addr", "st"])
    files, offs = F.sx_files(split, c)
    bs = np.zeros(len(n), np.float32)
    for f, a, b in zip(files, offs[:-1], offs[1:]):
        sel = np.flatnonzero((n.qx.to_numpy() >= a) & (n.qx.to_numpy() < b))
        if not len(sel):
            continue
        tx_all = pd.read_parquet(f, columns=["name", "addr", "st"])
        for i in range(0, len(sel), 400_000):
            s = sel[i:i + 400_000]
            A1 = B.mats(s1t.iloc[n.q1.to_numpy()[s]].reset_index(drop=True), {"na"}, vecs, CFG)["na"]
            AX = B.mats(tx_all.iloc[n.qx.to_numpy()[s] - a].reset_index(drop=True), {"na"}, vecs, CFG)["na"]
            bs[s] = np.asarray(A1.multiply(AX).sum(1)).ravel()
            del A1, AX
        del tx_all
    n["block_score"] = bs
    del s1t, bs
    # pass 2: context over existing blocked pairs + new pairs
    ctx = {}
    for side, ex in (("s1", pd.concat(ex1, ignore_index=True)), ("sx", pd.concat(exx, ignore_index=True))):
        g = "q1" if side == "s1" else "qx"
        u = pd.concat([ex.assign(new=False), n[["q1", "qx", "block_score"]].assign(new=True)], ignore_index=True)
        cx = F._group_ctx(u[g].to_numpy(), np.nan_to_num(u.block_score.to_numpy(np.float32)), side)
        o = np.lexsort((-u.block_score.to_numpy(), u[g].to_numpy()))
        rr = np.empty(len(u), np.int32)
        rr[o] = B.ranks(u[g].to_numpy()[o])
        nm = u.new.to_numpy()
        ctx |= {kk: v[nm] for kk, v in cx.items()}
        ctx[f"block_rank_{side}"] = rr[nm]
        del u, cx, o, rr
    del ex1, exx
    base = pd.DataFrame({"q1": n.q1, "qx": n.qx, **ctx, "block_score": n.block_score})
    base["via_r"], base["via_f"], base["blocked"] = np.float32(1), np.float32(0), np.float32(1)
    del ctx
    # pass 3: string features per chunk of SX, then q and p
    r1 = F.s1_table(split, c)
    m1, counts = F.mats(r1), F.pool_counts(r1)
    idx = F.ids(split, c)[1]
    rf_all = pq.read_table(W / f"rf_{split}_sx_{c}.parquet")
    raw_t = [pq.read_table(W / f"{split}_source{i}.parquet", columns=["entity_id", "business_name"]) for i in (2, 3)]
    mdl, pmd, outs = M.load(), PR.load_model(), []
    uqx = np.unique(n.qx.to_numpy())
    for a in range(0, len(uqx), chunk):
        qs = uqx[a:a + chunk]
        sel = np.flatnonzero(np.isin(n.qx.to_numpy(), qs))
        ids_ = idx.take(pa.array(qs))
        rf = rf_all.filter(pc.is_in(rf_all["id"], value_set=ids_)).to_pandas().drop_duplicates("id").set_index("id").reindex(ids_.to_pylist()).reset_index()
        _, txs = _texts_rows(split, c, np.zeros(0, np.int64), qs)
        raw = pd.concat([t.filter(pc.is_in(t["entity_id"], value_set=ids_)).to_pandas() for t in raw_t]).drop_duplicates("entity_id")
        raw = raw.set_index("entity_id").business_name.reindex(ids_.to_pylist())
        rx = F.assemble(rf[F.RF], txs, raw.reset_index(drop=True))
        mx = F.mats(rx)
        j = pd.Index(qs).get_indexer(n.qx.to_numpy()[sel])
        S = pd.concat([F.strings(r1, rx, n.q1.to_numpy()[sel][b:b + 200_000], j[b:b + 200_000], counts, m1, mx)
                       for b in range(0, len(sel), 200_000)], ignore_index=True)
        S["rule_acc"] = np.float32(0)
        X = pd.concat([base.iloc[sel].reset_index(drop=True), S.round(3)], axis=1)
        for col in X.columns:
            if col not in ("q1", "qx"):
                X[col] = X[col].astype(np.float32).round(3)
        X["q"] = PR.score(pmd, X)
        X["p"] = M.predict(mdl, X)
        X["r1"], X["r2"] = n.r1.to_numpy()[sel], n.r2.to_numpy()[sel]
        outs.append(X[["q1", "qx", "q", "p", "r1", "r2", "block_score", "block_rank_sx", "name_cos", "addr_cos", "hn_eq", "core_eq"]])
        print(f"  SX chunk {a:,}: {len(sel):,} pairs, {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)
        del rf, txs, raw, rx, mx, S, X
        F.trim()
    X = pd.concat(outs, ignore_index=True)
    X.to_parquet(OUT / f"new_{split}_{c}.parquet", index=False)
    res = {"new": len(X), "r1": int(X.r1.sum()), "r2": int(X.r2.sum()), "q_ge_tau": int((X.q >= PR.TAU).sum()),
           "p_ge_071": int((X.p >= T_BASE).sum()), "keep": os.environ.get("P68_KEEP", "all"),
           "runtime_s": round(time.time() - t0), "peak_gb": round(peak(), 2)}
    print(json.dumps(res), flush=True)
    return res


# ---------------------------------------------------------------- 3. holdout gate (add-only on free SX)
def fbeta(tp, npred, ntrue):
    P = np.where(npred > 0, tp / np.maximum(npred, 1), 0.0)
    R = np.where(ntrue > 0, tp / np.maximum(ntrue, 1), 0.0)
    f = np.where(P + R > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0)
    return np.where((ntrue == 0) & (npred == 0), 1.0, f)


def add_only(new_q1, new_qx, p, free_qx, n2, n3, is2, t):
    """Indices of new pairs to add: SX free, its best new pair, p >= t, S1 within the caps (counts n2 / n3 per q1)."""
    o = np.lexsort((-p, new_qx))
    first = np.r_[True, new_qx[o][1:] != new_qx[o][:-1]] if len(o) else np.zeros(0, bool)
    cand = o[first]
    cand = cand[(p[cand] >= t) & np.isin(new_qx[cand], free_qx)]
    cand = cand[np.argsort(-p[cand], kind="stable")]
    c2, c3, out = dict(n2), dict(n3), []
    for i in cand:
        s, two = new_q1[i], is2[i]
        cur = c2 if two else c3
        if cur.get(s, 0) < (CAP["S2"] if two else CAP["S3"]):
            cur[s] = cur.get(s, 0) + 1
            out.append(i)
    return np.array(out, np.int64)


def gate(c):
    sys.path.insert(0, str(ROOT / "probes/P28-addr-anchor"))
    from ho import half
    t0 = time.time()
    n = pd.read_parquet(OUT / f"new_train_{c}.parquet")
    id1, idx = F.ids("train", c)
    ho = holdout_ids(c)
    n = n[pc.is_in(id1.take(pa.array(n.q1.to_numpy())), value_set=pa.array(list(ho))).to_numpy(zero_copy_only=False)].reset_index(drop=True)
    d = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "pred"])
    assigned_sx = set(d.sx[d.pred])
    # SX outside the dump: status from their existing pairs' m2 p (X files), argmax >= T_BASE = assigned (approximation)
    sx_ids = idx.take(pa.array(n.qx.to_numpy())).to_numpy(zero_copy_only=False)
    in_dump = pd.Series(sx_ids).isin(set(d.sx)).to_numpy()
    need = np.unique(n.qx.to_numpy()[~in_dump])
    mdl, best = M.load(), {}
    for p in F.xfiles("train", c):
        X = pd.read_parquet(p)
        X = X[np.isin(X.qx.to_numpy(), need)]
        if len(X):
            pp = M.predict(mdl, X)
            s = pd.Series(pp).groupby(X.qx.to_numpy()).max()
            best.update(s.to_dict())
    out_assigned = {q for q, v in best.items() if v >= T_BASE}
    free = np.array([(not dm and q not in out_assigned) or (dm and s not in assigned_sx)
                     for q, s, dm in zip(n.qx.to_numpy(), sx_ids, in_dump)], bool)
    # base predicted sets of holdout S1 (#8c's decision, P23 dump) and truth
    tr = pd.read_parquet(W / "worlds/holdout/truth.parquet")
    tr = tr[tr.s1.isin(ho)]
    ntrue = tr.s1.value_counts()
    pr = d[d.pred & d.holdout]
    ho_ids = np.array(sorted(ho), object)
    tp0 = pr[pr.y].s1.value_counts().reindex(ho_ids).fillna(0).to_numpy()
    np0 = pr.s1.value_counts().reindex(ho_ids).fillna(0).to_numpy()
    nt = ntrue.reindex(ho_ids).fillna(0).to_numpy()
    f0 = fbeta(tp0, np0, nt)
    hv = half(ho_ids)
    n2 = pr[pr.sx.str.startswith("S2")].s1.value_counts().to_dict()
    n3 = pr[pr.sx.str.startswith("S3")].s1.value_counts().to_dict()
    s1n = id1.take(pa.array(n.q1.to_numpy())).to_numpy(zero_copy_only=False)
    is2 = pd.Series(sx_ids).str.startswith("S2").to_numpy()
    pos = pd.Index(ho_ids).get_indexer(s1n)
    y, p = n.is_true.to_numpy(), n.p.to_numpy()
    res = {"new_holdout_pairs": len(n), "true": int(y.sum()), "free_sx_pairs": int(free.sum()),
           "true_free": int((y & free).sum()), "sx_outside_dump": int((~in_dump).sum())}
    grid = {}
    for arm, mask in (("all", np.ones(len(n), bool)), ("pruned", n.q.to_numpy() >= PR.TAU)):
        for t in (0.5, 0.6, 0.71, 0.8, 0.9, 0.95):
            ok = mask & free
            sel = add_only(s1n[ok], n.qx.to_numpy()[ok], p[ok], n.qx.to_numpy()[ok], n2, n3, is2[ok], t)
            ii = np.flatnonzero(ok)[sel]
            tp = tp0 + np.bincount(pos[ii][y[ii]], minlength=len(ho_ids))
            npd = np0 + np.bincount(pos[ii], minlength=len(ho_ids))
            dl = fbeta(tp, npd, nt) - f0
            r = {f"d_h{h}": float(dl[hv == h].mean()) for h in (0, 1)} | {f"se_h{h}": float(dl[hv == h].std() / np.sqrt((hv == h).sum())) for h in (0, 1)}
            r |= {"adds": len(ii), "prec": float(y[ii].mean()) if len(ii) else None, "r1": int(n.r1.to_numpy()[ii].sum()),
                  "r2": int(n.r2.to_numpy()[ii].sum()), "lb_eq_h1": float(dl[hv == 1].mean()) * SHARE[c]}
            grid[f"{arm}@{t}"] = r
            print(f"{c} {arm:6s} t {t}: dF h0 {r['d_h0']:+.5f} ± {r['se_h0']:.5f} | h1 {r['d_h1']:+.5f} ± {r['se_h1']:.5f} | "
                  f"adds {r['adds']:,} prec {r['prec']} (R1 {r['r1']}, R2 {r['r2']}) | LB-eq h1 {r['lb_eq_h1']:+.6f}", flush=True)
    # which slices carry the gain (for the test-side P68_KEEP pre-filter): search view / rank of each new pair
    hh = pd.read_parquet(OUT / f"hits_train_{c}.parquet")
    hk = key(hh.q1, hh.qx)
    rk_na = pd.Series(np.where(hh.view.astype(str) == "na", hh["rank"], 99)).groupby(hk).min()
    rk_ad = pd.Series(np.where(hh.view.astype(str) == "addr", hh["rank"], 99)).groupby(hk).min()
    nk = key(n.q1, n.qx)
    rna, rad = rk_na.reindex(nk).fillna(99).to_numpy(), rk_ad.reindex(nk).fillna(99).to_numpy()
    slices = {"na:2-3": rna <= 3, "na:2-4": rna <= 4, "addr:0-1": rad <= 1, "addr:0-2": rad <= 2, "addr:0-4": rad <= 4,
              "na:2-3|addr:0-2": (rna <= 3) | (rad <= 2), "hn_eq": n.hn_eq.to_numpy() == 1}
    t_pick = float(max(grid, key=lambda g: grid[g]["d_h0"]).split("@")[1])
    for name, sm in slices.items():
        ok = sm & free & (n.q.to_numpy() >= PR.TAU)
        sel = add_only(s1n[ok], n.qx.to_numpy()[ok], p[ok], n.qx.to_numpy()[ok], n2, n3, is2[ok], t_pick)
        ii = np.flatnonzero(ok)[sel]
        tp = tp0 + np.bincount(pos[ii][y[ii]], minlength=len(ho_ids))
        npd = np0 + np.bincount(pos[ii], minlength=len(ho_ids))
        dl = fbeta(tp, npd, nt) - f0
        grid[f"slice {name}@{t_pick}"] = {"d_all": float(dl.mean()), "adds": len(ii), "prec": float(y[ii].mean()) if len(ii) else None,
                                          "pairs_in_slice": int(sm.sum())}
        print(f"{c} slice {name:16s} (pruned, t {t_pick}): dF all {dl.mean():+.5f} | adds {len(ii):,} prec "
              f"{grid[f'slice {name}@{t_pick}']['prec']} | new pairs in slice {int(sm.sum()):,}", flush=True)
    # pass-2 input: every free SX whose best holdout new pair clears the lowest grid t (re-searched against ALL S1)
    okq = free & (n.q.to_numpy() >= PR.TAU) & (p >= 0.5)
    pd.DataFrame({"qx": np.unique(n.qx.to_numpy()[okq])}).to_parquet(OUT / f"gate2_qx_{c}.parquet", index=False)
    best_arm = max((g for g in grid if not g.startswith("slice")), key=lambda g: grid[g]["d_h0"])
    res |= {"grid": grid, "pick_on_h0": best_arm, "h1_at_pick": grid[best_arm]["d_h1"], "lb_eq_h1_at_pick": grid[best_arm]["lb_eq_h1"],
            "runtime_s": round(time.time() - t0)}
    print(f"{c}: pick on h0 = {best_arm}; h1 dF {grid[best_arm]['d_h1']:+.5f} ± {grid[best_arm]['se_h1']:.5f}; "
          f"LB-eq {grid[best_arm]['lb_eq_h1']:+.6f}", flush=True)
    return res


def gate_final(c):
    """Pass 2: the pass-1 candidate SX (gate2_qx) re-searched against ALL train S1 (P68_ALL_S1=1 P68_TAG=_all), so an
    SX's best new pair is picked among every S1 as on test. Adds are scored on the holdout S1 they land on."""
    sys.path.insert(0, str(ROOT / "probes/P28-addr-anchor"))
    from ho import half
    t0 = time.time()
    n = pd.read_parquet(OUT / f"new_train_all_{c}.parquet")
    cq = pd.read_parquet(OUT / f"gate2_qx_{c}.parquet").qx.to_numpy()
    n = n[np.isin(n.qx.to_numpy(), cq)].reset_index(drop=True)
    id1, idx = F.ids("train", c)
    ho = holdout_ids(c)
    ho_ids = np.array(sorted(ho), object)
    d = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "pred"])
    tr = pd.read_parquet(W / "worlds/holdout/truth.parquet")
    nt = tr[tr.s1.isin(ho)].s1.value_counts().reindex(ho_ids).fillna(0).to_numpy()
    pr = d[d.pred & d.holdout]
    tp0 = pr[pr.y].s1.value_counts().reindex(ho_ids).fillna(0).to_numpy()
    np0 = pr.s1.value_counts().reindex(ho_ids).fillna(0).to_numpy()
    f0, hv = fbeta(tp0, np0, nt), half(ho_ids)
    n2 = pr[pr.sx.str.startswith("S2")].s1.value_counts().to_dict()
    n3 = pr[pr.sx.str.startswith("S3")].s1.value_counts().to_dict()
    s1n = id1.take(pa.array(n.q1.to_numpy())).to_numpy(zero_copy_only=False)
    sxn = idx.take(pa.array(n.qx.to_numpy())).to_numpy(zero_copy_only=False)
    is2 = pd.Series(sxn).str.startswith("S2").to_numpy()
    pos = pd.Index(ho_ids).get_indexer(s1n)                  # -1: a non-holdout S1 (its F0.5 is not measured)
    y, p = n.is_true.to_numpy(), n.p.to_numpy()
    res, grid = {"candidate_sx": len(cq), "pairs_all_s1": len(n), "runtime_s": 0}, {}
    for t in (0.5, 0.6, 0.71, 0.8, 0.9, 0.95):
        ok = n.q.to_numpy() >= PR.TAU
        sel = add_only(s1n[ok], n.qx.to_numpy()[ok], p[ok], cq, n2, n3, is2[ok], t)
        ii = np.flatnonzero(ok)[sel]
        hi = ii[pos[ii] >= 0]
        tp = tp0 + np.bincount(pos[hi][y[hi]], minlength=len(ho_ids))
        npd = np0 + np.bincount(pos[hi], minlength=len(ho_ids))
        dl = fbeta(tp, npd, nt) - f0
        r = {f"d_h{h}": float(dl[hv == h].mean()) for h in (0, 1)} | {f"se_h{h}": float(dl[hv == h].std() / np.sqrt((hv == h).sum())) for h in (0, 1)}
        r |= {"adds_all_s1": len(ii), "adds_holdout": len(hi), "prec_holdout": float(y[hi].mean()) if len(hi) else None,
              "r1": int(n.r1.to_numpy()[hi].sum()), "r2": int(n.r2.to_numpy()[hi].sum()), "lb_eq_h1": float(dl[hv == 1].mean()) * SHARE[c]}
        grid[f"pruned@{t}"] = r
        print(f"{c} FINAL t {t}: dF h0 {r['d_h0']:+.6f} ± {r['se_h0']:.6f} | h1 {r['d_h1']:+.6f} ± {r['se_h1']:.6f} | adds on holdout "
              f"{r['adds_holdout']:,} prec {r['prec_holdout']} (R1 {r['r1']}, R2 {r['r2']}) | LB-eq h1 {r['lb_eq_h1']:+.7f}", flush=True)
    pick = max(grid, key=lambda g: grid[g]["d_h0"])
    res |= {"grid": grid, "pick_on_h0": pick, "h1_at_pick": grid[pick]["d_h1"], "se_h1_at_pick": grid[pick]["se_h1"],
            "lb_eq_h1_at_pick": grid[pick]["lb_eq_h1"], "runtime_s": round(time.time() - t0)}
    print(f"{c}: FINAL pick on h0 = {pick}; h1 {grid[pick]['d_h1']:+.6f} ± {grid[pick]['se_h1']:.6f}; LB-eq (sample) {grid[pick]['lb_eq_h1']:+.7f}", flush=True)
    return res


# ---------------------------------------------------------------- 4. test build: BASE + add-only tier (India / US lines)
RD = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def build(base, out, t, arm="pruned"):
    """BASE's matching_results.tsv + the tier's adds (free SX, best new pair, p >= t, caps 5/6), India / US lines only.
    candidate_pairs.tsv = BASE's + every new pair the tier scored (arm 'pruned': q >= TAU, the pruner's cascade)."""
    t0, base, out, t = time.time(), Path(base), Path(out), float(t)
    sub = pd.read_csv(base / "matching_results.tsv", **RD)
    pred = {s: (m.split(",") if m else []) for s, m in zip(sub.source1_entity_id, sub.matched_entity_ids)}
    owned = {x for v in pred.values() for x in v}
    adds, cand_new, info = {}, {}, {"base": base.name, "t": t, "arm": arm}
    for c in ("India", "US"):
        n = pd.read_parquet(OUT / f"new_test_{c}.parquet")
        if arm == "pruned":
            n = n[n.q.to_numpy() >= PR.TAU].reset_index(drop=True)
        id1, idx = F.ids("test", c)
        s1n = id1.take(pa.array(n.q1.to_numpy())).to_numpy(zero_copy_only=False)
        sxn = idx.take(pa.array(n.qx.to_numpy())).to_numpy(zero_copy_only=False)
        for a, b in zip(s1n, sxn):
            cand_new.setdefault(a, []).append(b)
        free = ~pd.Series(sxn).isin(owned).to_numpy()
        n2 = {s: sum(x.startswith("S2") for x in v) for s, v in pred.items() if v}
        n3 = {s: sum(x.startswith("S3") for x in v) for s, v in pred.items() if v}
        is2 = pd.Series(sxn).str.startswith("S2").to_numpy()
        ok = np.flatnonzero(free)
        sel = ok[add_only(s1n[ok], n.qx.to_numpy()[ok], n.p.to_numpy()[ok], n.qx.to_numpy()[ok], n2, n3, is2[ok], t)]
        for i in sel:
            adds.setdefault(s1n[i], []).append(sxn[i])
        info[c] = {"new_pairs": len(n), "free": int(free.sum()), "added": len(sel), "s1_changed": len(set(s1n[sel])),
                   "empties_filled": sum(1 for s in set(s1n[sel]) if not pred.get(s)), "r1": int(n.r1.to_numpy()[sel].sum()),
                   "r2": int(n.r2.to_numpy()[sel].sum()), "p_q": np.quantile(n.p.to_numpy()[sel], [0, .25, .5, .75]).round(3).tolist() if len(sel) else []}
    out.mkdir(parents=True, exist_ok=True)
    for name, extra in (("matching_results.tsv", adds), ("candidate_pairs.tsv", None)):
        src = base / name
        with open(src, encoding="utf-8") as f, open(out / name, "w", encoding="utf-8") as g:
            g.write(next(f))
            for line in f:
                s, v = line.rstrip("\n").split("\t", 1)
                more = (extra if extra is not None else cand_new).get(s)
                if more:
                    cur = v.split(",") if v else []
                    v = ",".join(cur + [x for x in more if x not in set(cur)])
                    line = f"{s}\t{v}\n"
                g.write(line)
    if not (out / "code.zip").exists() and (base / "code.zip").exists():
        os.link(base / "code.zip", out / "code.zip")
    info["cand_pairs_added"] = sum(map(len, cand_new.values()))
    info["runtime_s"], info["peak_gb"] = round(time.time() - t0), round(peak(), 2)
    (out / "p68_info.json").write_text(json.dumps(info, default=str))
    print(json.dumps(info, default=str), flush=True)
    return info


def selftest():
    sel = add_only(np.array([1, 1, 2, 2, 3]), np.array([10, 10, 11, 12, 13]), np.array([0.9, 0.95, 0.8, 0.7, 0.99]),
                   np.array([10, 11, 13]), {1: 5}, {}, np.array([True, True, True, True, False]), 0.75)
    assert list(sel) == [4, 2], sel          # SX 10: best new pair (0.95) -> S1 1 is capped (5 S2); SX 12 not free
    f = fbeta(np.array([1, 0, 0]), np.array([1, 0, 1]), np.array([2, 0, 0]))
    assert abs(f[0] - 1.25 * 0.5 / 0.75) < 1e-9 and f[1] == 1.0 and f[2] == 0.0
    print("p68 self-check ok")


if __name__ == "__main__":
    from ledger import log
    cmd, args = sys.argv[1], sys.argv[2:]
    cfg = {"rev": CFG["rev"], "keep": "R1 na rank 2-4 | R2 addr rank 0-4, addressed SX", "t_base": T_BASE, "caps": CAP}
    if cmd == "selftest":
        selftest()
    elif cmd in ("search", "feats", "feats_c"):
        r = {"search": search, "feats": feats, "feats_c": feats_c}[cmd](*args)
        log("exp", probe=PROBE, subproblem="iu-retrieval", name=f"{cmd}-{args[0]}-{args[1]}", world=args[0], metrics=r,
            config=cfg, verdict="inconclusive", runtime_s=r["runtime_s"], peak_gb=r["peak_gb"])
    elif cmd == "build":
        r = build(*args)
        log("exp", probe=PROBE, subproblem="iu-retrieval", name=f"build-{Path(args[1]).name}", world="test", metrics=r,
            config=cfg, verdict="inconclusive", runtime_s=r["runtime_s"], peak_gb=r["peak_gb"])
    elif cmd == "gate_final":
        for c in args:
            r = gate_final(c)
            log("exp", probe=PROBE, subproblem="iu-retrieval", name=f"gate-final-{c}", world="holdout (fold 0, sampled SX files; best new pair over ALL S1)",
                metrics=r, config=cfg, hypothesis="add-only new candidates for free addressed SX lift holdout F0.5",
                verdict="keep" if r["h1_at_pick"] > 2 * r["se_h1_at_pick"] else "drop", runtime_s=r["runtime_s"])
    elif cmd == "gate":
        for c in args:
            r = gate(c)
            log("exp", probe=PROBE, subproblem="iu-retrieval", name=f"gate-{c}", world="holdout (fold 0, #8c P23 dump base)",
                metrics=r, config=cfg, hypothesis="add-only new candidates for free addressed SX lift holdout F0.5",
                verdict="keep" if r["h1_at_pick"] > 0 else "drop", runtime_s=r["runtime_s"])
