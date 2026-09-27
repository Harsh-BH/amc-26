"""#34's India / US stage (P34 port of P23's a10 'twin_p2mask' residual stacker; LB: #34 = #31 + this, India +0.0002 /
US +0.0004 holdout half 1). run.py applies it after the CE blend and P17's US stack when the recipe names a booster.

A 200 x 31 LightGBM residual (init_score = logit p) on #8c's argmax pairs: each S2/S3 keeps its #8c argmax S1, the
stacker only re-scores that pair, and the pair is kept when sigma(raw + logit p) >= t. Columns = SX context (runner-up
p2 / margin, masked where the runner-up carries the holdout-only e5 blend), S1 context, twin counts (other candidate
S1 of the SX with the argmax S1's exact name / address / legal-stripped core), name-only flag, error category.
The booster (data/work/p34/a10_h0.txt) is fit on the India + US holdout half 0 (probes/P34-a10-stack/ho.py: 2 s,
bit-identical refit).

features() is the one builder for holdout and test, so the two cannot drift. Everything is keyed on integer codes:
q1 / qx = rows of an S1 / SX text table (positional).
    python code/business_entity_resolution/src/a10.py         # self-check
"""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worlds import WORK as W
T = 0.71
LEGAL = set("ltd limited pvt private inc incorporated llc corp corporation co company pc pllc plc llp lp the of and "
            "pvtltd opc".split())
INDIC = re.compile("[ऀ-෿]")
CATS = ["indic", "web", "noaddr_exact", "noaddr_legal", "noaddr_subset", "noaddr_other", "name_replaced", "hn_diff",
        "addr_fmt_name_same", "addr_name_variant"]
COLS = ["lp", "lm2", "r", "n_cand_sx", "n_conf_other", "s1_max_other", "s1_n_am", "p2m", "marginm",   # a13 MASK
        "n_twins", "n_ctwins", "n_atwins", "noaddr", "cat", "name_sup"]                              # a10 TWIN
PRM = dict(objective="binary", num_leaves=31, learning_rate=0.05, min_data_in_leaf=200, lambda_l2=1.0, verbose=-1,
           num_threads=4, seed=0, deterministic=True)
ROUNDS = 200
CHUNK = 200_000
lg = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))


def norm(s: str) -> list:
    s = unicodedata.normalize("NFKD", s.replace(".", ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9ऀ-෿]+", " ", s).split()


full = lambda s: " ".join(norm(s))
core = lambda s: " ".join(sorted(set(norm(s)) - LEGAL))


def grab(path, ids, cols) -> pd.DataFrame:
    """Rows of a parquet whose first column is in ids, streamed by row group (bounded RAM)."""
    v = pa.array(sorted(set(ids)), pa.large_string())
    f, out = pq.ParquetFile(path), []
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=cols)
        out.append(t.filter(pc.is_in(t[cols[0]], value_set=v)).to_pandas())
        del t
    return pd.concat(out).drop_duplicates(cols[0]).set_index(cols[0])


def texts(split: str, c: str, s1_ids, sx_ids):
    """(S1 name/addr/hn, SX name/addr/is_web/hn) of the given ids, rows in id order (NaN-free)."""
    s1 = grab(W / f"{split}_source1.parquet", s1_ids, ["entity_id", "business_name", "business_address"])
    s1["hn"] = grab(W / f"rf_{split}_s1_{c}.parquet", s1_ids, ["id", "hn"]).hn
    sx = pd.concat([grab(W / f"{split}_source{i}.parquet", sx_ids, ["entity_id", "business_name", "business_address"])
                    for i in (2, 3)])
    rf = grab(W / f"rf_{split}_sx_{c}.parquet", sx_ids, ["id", "is_web", "hn"])
    sx["is_web"], sx["hn"] = rf.is_web, rf.hn
    s1, sx = s1.reindex(list(s1_ids)), sx.reindex(list(sx_ids))
    for d in (s1, sx):
        d.columns = [{"business_name": "name", "business_address": "addr"}.get(k, k) for k in d.columns]
        d["hn"] = d.hn.fillna("")
    sx["is_web"] = sx.is_web.fillna(False).astype(bool)
    assert not s1.name.isna().any() and not sx.name.isna().any(), "text lookup missed ids"
    return s1.reset_index(drop=True), sx.reset_index(drop=True)


def cat_frame(s1_name, sx_name, sx_addr, sx_is_web, s1_hn, sx_hn) -> np.ndarray:
    """P23's error category of a pair (cats.cat_frame), as CATS codes."""
    n1, nx = s1_name.map(norm), sx_name.map(norm)
    l1, lx = n1.map(lambda s: set(s) - LEGAL), nx.map(lambda s: set(s) - LEGAL)
    exact = (n1.map("".join) == nx.map("".join)).to_numpy()
    legal = ((l1 == lx) & (l1.map(len) > 0)).to_numpy()
    sub = np.array([bool(a) and bool(b) and (a <= b or b <= a) for a, b in zip(l1, lx)])
    ov = np.array([len(a & b) for a, b in zip(l1, lx)])
    noaddr = (sx_addr.str.strip() == "").to_numpy()
    indic = sx_name.str.contains(INDIC).to_numpy()
    h1, hx = s1_hn.to_numpy(object), sx_hn.to_numpy(object)
    hn_diff = (h1 != "") & (hx != "") & (h1 != hx)
    cat = np.select([indic, sx_is_web.to_numpy(bool), noaddr & exact, noaddr & legal, noaddr & sub, noaddr, ov == 0,
                     hn_diff, exact | legal], CATS[:9], CATS[9])
    return pd.Categorical(cat, categories=CATS).codes


def features(q1, qx, p, p_m2, r, am, scored, s1t, sxt, t: float = T) -> pd.DataFrame:
    """The a13 twin_p2mask columns of the argmax pairs of scored S1.
    q1 / qx: int codes into s1t / sxt of every scored pair of the SX in play (holdout: incl. competing train S1);
    p = #8c's final p, p_m2 = M2, r = CE score (NaN outside the band), am = #8c's argmax, scored = the S1 is scored.
    s1t: name, addr, hn of every S1 code; sxt: name, addr, is_web, hn (needed for the am rows of scored S1).
    -> one row per am pair of a scored S1: q1, qx, p, row (index into the inputs), COLS."""
    q1, qx = np.asarray(q1, np.int64), np.asarray(qx, np.int64)
    p, p_m2, r = (np.asarray(x, np.float64) for x in (p, p_m2, r))

    def twins(key_of_s1):              # other rows of the same SX whose S1 has the same key
        k = pd.factorize(key_of_s1)[0].astype(np.int64)[q1]
        _, inv, cnt = np.unique((qx << 32) | k, return_inverse=True, return_counts=True)
        return cnt[inv] - 1

    n_tw, n_atw, n_ctw = twins(s1t.name.map(full)), twins(s1t.addr.map(full)), twins(s1t.name.map(core))
    n_cand = np.bincount(qx)[qx]
    # runner-up = the SX's best non-argmax row: p2, and whether it carries the e5 blend (band, no bge r)
    na = np.flatnonzero(~am)
    o = na[np.lexsort((-p[na], qx[na]))]
    o = o[np.r_[True, qx[o][1:] != qx[o][:-1]]] if len(o) else o
    p2 = np.zeros(qx.max() + 1 if len(qx) else 0)
    e5 = np.zeros(len(p2), bool)
    p2[qx[o]] = p[o]
    e5[qx[o]] = (p_m2[o] > 0.02) & (p_m2[o] < 0.98) & np.isnan(r[o])
    row = np.flatnonzero(am & scored)
    a = pd.DataFrame({"q1": q1[row], "qx": qx[row], "p": p[row], "row": row, "lp": lg(p[row]), "lm2": lg(p_m2[row]),
                      "r": r[row], "n_cand_sx": n_cand[row], "n_twins": n_tw[row], "n_atwins": n_atw[row],
                      "n_ctwins": n_ctw[row]})
    a["p2"] = p2[a.qx]
    m = e5[a.qx]
    a["p2m"] = np.where(m, np.nan, a.p2)
    a["marginm"] = np.where(m, np.nan, a.lp - lg(a.p2.to_numpy()))
    # S1 context over the S1's argmax pairs
    g = pd.factorize(a.q1)[0]
    conf = (a.p >= t).to_numpy()
    a["n_conf_other"] = np.bincount(g, conf, len(set(g)))[g] - conf
    a["s1_n_am"] = np.bincount(g)[g]
    o = np.lexsort((-a.p.to_numpy(), g))
    rk = np.empty(len(a), np.int64)
    first = np.r_[True, g[o][1:] != g[o][:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(o)), 0))
    rk[o] = np.arange(len(o)) - start
    top1, top2 = np.zeros(g.max() + 1 if len(g) else 0), np.zeros(g.max() + 1 if len(g) else 0)
    top1[g[o][rk[o] == 0]] = a.p.to_numpy()[o][rk[o] == 0]
    top2[g[o][rk[o] == 1]] = a.p.to_numpy()[o][rk[o] == 1]
    a["s1_max_other"] = np.where(rk == 0, top2[g], top1[g])
    # SX text columns, in chunks: the token lists of 2M test pairs would not fit the cap
    import stack as S
    noaddr, nk, cat, hc = (np.zeros(len(a), dt) for dt in (np.float32, np.int64, np.int8, np.int8))
    for i in range(0, len(a), CHUNK):
        j = slice(i, i + CHUNK)
        x, s = (d.iloc[a[k].to_numpy()[j]].reset_index(drop=True) for d, k in ((sxt, "qx"), (s1t, "q1")))
        noaddr[j] = (x.addr.str.strip() == "").to_numpy(np.float32)
        nk[j] = x.name.map(lambda v: hash("".join(norm(v)))).to_numpy(np.int64)   # in-process key only
        cat[j] = cat_frame(s.name, x.name, x.addr, x.is_web, s.hn, x.hn)
        hc[j] = S.hn_cls(s.hn.to_numpy(object), x.hn.to_numpy(object))
        del x, s
    keys = set(zip(a.q1.to_numpy()[conf], nk[conf]))
    a["noaddr"], a["cat"], a["hc"] = noaddr, cat, hc     # hc: P17's hn class, the guard's key (not a column)
    a["name_sup"] = np.fromiter(((q, k) in keys for q, k in zip(a.q1.to_numpy(), nk)), bool, len(a)) & ~conf
    return a


def fit(X: pd.DataFrame, y: np.ndarray):
    import lightgbm as lgb
    ds = lgb.Dataset(X[COLS], y, init_score=X.lp.to_numpy(), categorical_feature=["cat"], free_raw_data=True)
    return lgb.train(PRM, ds, ROUNDS)


def stacked(b, X: pd.DataFrame, guard: bool = True) -> np.ndarray:
    """p' = sigma(raw + logit p) of the argmax pairs. guard: a sibling-offset pair (stack.hn_cls 1: sx hn = s1 hn + d,
    d in stack.SIB) keeps #8c's p -- P17's US stack owns that class, and test US holds ~5x holdout's share of it near t."""
    out = 1 / (1 + np.exp(-(b.predict(X[COLS], raw_score=True) + X.lp.to_numpy())))
    return np.where(X.hc.to_numpy() == 1, X.p.to_numpy(), out) if guard else out


def apply(b, split: str, c: str, q1, qx, p, p_m2, r, id1, idx, scored=None) -> tuple:
    """The run.py step, after P17's stack: #8c's rows of one country -> (p', am, a). p' = the stacked p on #8c's argmax
    pairs of scored S1, p elsewhere; am = #8c's argmax, which the decision must keep (acc = am & p' >= t; re-deriving
    the argmax from p' changes ~600 test decisions per country). id1 / idx: S1 / SX id arrays the q1 / qx codes index;
    split: whose text files ('train' for the holdout, 'test'); scored: rows whose S1 is scored (default all).
    id1 / idx may be arrow arrays (run.py's), only the ids in a pair are materialised."""
    am, a = frame(split, c, q1, qx, p, p_m2, r, id1, idx, scored)
    out = np.asarray(p, np.float64).copy()
    out[a.row.to_numpy()] = stacked(b, a)
    return out, am, a


def frame(split: str, c: str, q1, qx, p, p_m2, r, id1, idx, scored=None) -> tuple:
    """(am, features) of #8c's rows of one country: the text lookup + the columns (apply's input; iu_stack's base).
    The frame's q1 / qx are compact codes (only the records in a pair); row indexes the inputs."""
    import model as M
    q1, qx, p = np.asarray(q1), np.asarray(qx), np.asarray(p, np.float64)
    am = M.argmax_sx(qx, p)
    k1, u1 = pd.factorize(q1)                 # compact codes: only the records that are in a pair
    kx, ux = pd.factorize(qx)
    take = lambda ids, u: ids.take(pa.array(u)).to_numpy(zero_copy_only=False) if isinstance(ids, pa.Array) \
        else np.asarray(ids, object)[u]
    s1t, sxt = texts(split, c, take(id1, u1), take(idx, ux))
    a = features(k1, kx, p, p_m2, r, am, np.ones(len(p), bool) if scored is None else scored, s1t, sxt)
    return am, a


def fb(tp, npred, ntrue) -> np.ndarray:
    """Per-S1 F0.5 (metric.py's rule: no truth -> 1 iff nothing predicted)."""
    tp, npred, ntrue = (np.asarray(x, float) for x in (tp, npred, ntrue))
    with np.errstate(divide="ignore", invalid="ignore"):
        P, R = tp / npred, tp / ntrue
        f = 1.25 * P * R / (0.25 * P + R)
    f = np.where(tp == 0, 0.0, f)
    return np.where(ntrue == 0, np.where(npred == 0, 1.0, 0.0), f)


if __name__ == "__main__":
    # toy: SX 0 has argmax S1 0 and runner-up S1 1 (same name -> twin; band + no r -> e5-masked); SX 1 only S1 0
    s1t = pd.DataFrame({"name": ["Acme Pvt Ltd", "Acme Pvt Ltd", "Acme Private Limited"], "addr": ["x", "y", "x"],
                        "hn": ["12", "", ""]})
    sxt = pd.DataFrame({"name": ["acme", "Acme Pvt. Ltd"], "addr": ["", "12 main"], "is_web": [False, False],
                        "hn": ["", "14"]})
    q1, qx = np.array([0, 1, 2, 0]), np.array([0, 0, 0, 1])
    p = np.array([0.8, 0.5, 0.1, 0.6]); am = np.array([True, False, False, True])
    a = features(q1, qx, p, p, np.array([0.3, np.nan, np.nan, 0.2]), am, np.ones(4, bool), s1t, sxt)
    r0, r1 = a.iloc[0], a.iloc[1]
    assert (r0.n_twins, r0.n_ctwins, r0.n_atwins, r0.n_cand_sx) == (1, 2, 1, 3), r0
    assert r0.p2 == 0.5 and np.isnan(r0.p2m) and np.isnan(r0.marginm) and r1.p2 == 0 and r1.p2m == 0
    assert (r0.s1_n_am, r0.n_conf_other, r0.s1_max_other, r1.s1_max_other) == (2, 0, 0.6, 0.8)
    assert CATS[r0["cat"]] == "noaddr_legal" and CATS[r1["cat"]] == "hn_diff" and r0.noaddr == 1 and r1.noaddr == 0
    assert (r0.hc, r1.hc) == (0, 1) and np.allclose(stacked(type("B", (), {"predict": lambda self, X, raw_score: np.ones(len(X))})(), a), [1 / (1 + np.exp(-1 - lg(0.8))), 0.6])
    assert not r1.name_sup and np.allclose(fb([1, 0, 0], [2, 0, 1], [2, 0, 0]), [0.5, 1, 0])
    print("a10 self-check ok")
