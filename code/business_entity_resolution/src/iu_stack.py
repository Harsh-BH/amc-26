"""#52_max's India / US stage (P44 combiner + P45 r_all / zoo + P50 full-holdout refit), replacing P34's a10 in run.py.

Recipe sub52_iu (iu="fullzoo") = the India / US lines of output/sub52_max (and so of #54 / #clean before the dup step).
The combiner re-scores #8c's argmax pairs only: each S2/S3 keeps its #8c argmax S1 and is kept when p'' >= t (0.71).
p'' = sigmoid of the mean over 5 bagged residual boosters (init = logit #8c p) of logit(p'), with P17's guard
(stack.hn_cls 1 keeps #8c's p). Columns (FULL; ZOO = FULL + 16 zoo members):
  a10.COLS            P34's twin / runner-up / S1-context columns (a10.frame, the same builder)
  l_m2hn, l_hn21      logits of P44's two house-number GBDTs (SageMaker jobs p44ms/job_{m2hn,hn21}.txt)
  hn_*                P44's house-number corruption columns (solution_p44.hn_feats) on the rules.feats hn
  lq, tw_any          P10 pruner logit, any exact-name twin
  CTX2 / CTX3         P44's SX / S1 group context from #8c p (ctx.ctx2 / ctx.add3)
  l_r_all             P45: the shipped bge CE (ce_bge_pool.pt) on EVERY argmax pair, not only the band
  z_*                 P45's 16 zoo GBDT / MLP / rank members (logits; the two rank members raw)
Supplied inputs (label-free test files, read here; everything else is computed from run.model_parts' rows):
  data/work/p10/q_test_{c}.parquet            q1, qx, q: P10's pruner score of every blocked test pair
  data/work/p45/oof/r_all_test_{c}.parquet    s1, sx, r: bge CE of every #8c test argmax pair (P13 band r + P45 jobs)
  data/work/p50/members_test_{c}.parquet      q1, qx, p_m2hn, p_hn21, p_<16 zoo>: the member GBDTs' test preds on
                                              the argmax pairs (pack(): P50's T_{c} frame + zoo_test_{c}, which
                                              zoo_test.py aligned from the members' S3 pred/test_{c}.parquet)
  data/work/p50/{full,fullzoo}_s{0..4}.txt    the boosters (fit on the FULL India + US holdout, 5 seeds; the fit path
                                              stays in probes/P50-idea-batch/e_ship.py fit / e_zoo.py fit)
Test only: the boosters saw every holdout S1, so a holdout run through them would be in-sample.
Checked (27 Sep): run.py --recipe sub52_iu -> p'' bit-identical to P50's prob_test_{c}_fullzoo on every argmax pair,
India / US lines = sub52_max's (0 differing as ID sets; sub54 = those + P54's 14 dup copies); --iu a10 -> sub37_gc70
byte for byte. The iu stage: India 321 s / US 155 s, run peak 1.41 GB.
    python code/business_entity_resolution/src/iu_stack.py          # self-check
    python code/business_entity_resolution/src/iu_stack.py pack     # members_test_{c} from P50's frames (once)
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a10 as A
from worlds import WORK as W

T = A.T
MEM = ["bag1", "bag2", "bag3", "bag4", "cat", "dart", "deep", "et2", "extra", "fsub", "goss", "mlp", "mlp2", "noctx"]
RANK = ["rank_s1", "rank_sx2"]
HNF = ["hn_d", "hn_cls", "hn_lev", "hn_lendiff", "has_hn", "exact_hn"]
CTX2 = ["sx_n_ge_t", "sx_n_ge_05", "s1_ncand", "s1_rank", "s1_best_nonam", "n_same_hn_acc", "n_acc_hn_eq_s1"]
CTX3 = ["s1_gap_next", "s1_gap_prev", "sx_psum", "s1_psum", "is_s3", "n_same_src_acc"]
FULL = A.COLS + ["l_m2hn", "l_hn21"] + HNF + ["lq", "tw_any"] + CTX2 + CTX3 + ["l_r_all"]    # te44.FINAL, VAR "r"
ZOO = FULL + [f"z_{m}" for m in MEM + RANK]                                                   # e_zoo.COLS
BOOSTERS = {"full": "p50/full_s{}.txt", "fullzoo": "p50/fullzoo_s{}.txt"}
SEEDS = range(5)
Q, R_ALL, MEMBERS = "p10/q_test_{}.parquet", "p45/oof/r_all_test_{}.parquet", "p50/members_test_{}.parquet"
CHUNK = 400_000
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))   # P44's (ev.lg), not a10's
sig = lambda z: 1 / (1 + np.exp(-z))
keys = lambda q1, qx: (np.asarray(q1, np.int64) << 32) | np.asarray(qx, np.int64)


def by_key(path: Path, want: np.ndarray, cols: list) -> pd.DataFrame:
    """cols of a (q1, qx, ...) parquet at the keys want, first occurrence, streamed by row group (bounded RAM)."""
    f, parts = pq.ParquetFile(path), []
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=["q1", "qx"] + cols)
        k = keys(t.column("q1").to_numpy(), t.column("qx").to_numpy())
        m = np.isin(k, want)
        parts.append(pd.DataFrame({"k": k[m]} | {x: t.column(x).to_numpy()[m] for x in cols}))
        del t, k
    d = pd.concat(parts, ignore_index=True).drop_duplicates("k")
    pos = pd.Index(d.k).get_indexer(want)
    assert (pos >= 0).all(), f"{path.name}: {int((pos < 0).sum()):,} argmax pairs missing"
    return d.iloc[pos].reset_index(drop=True)


def hn_cols(h1, hx) -> dict:
    """solution_p44.hn_feats' hn_d / hn_lev / hn_lendiff (NaN where a side has no house number); hn_cls = a10's hc."""
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cpdist
    h1, hx = pa.array(h1, pa.string()), pa.array(hx, pa.string())
    l1, lx = (pc.utf8_length(h).to_numpy(zero_copy_only=False).astype(np.int64) for h in (h1, hx))
    both = (l1 > 0) & (lx > 0)
    num = lambda h, l: np.where(l > 0, pc.cast(pc.if_else(pc.equal(h, ""), "0", h), pa.int64()).to_numpy(), 0)
    nan = lambda v: np.where(both, v, np.nan).astype(np.float32)
    lev = cpdist(h1.to_pylist(), hx.to_pylist(), scorer=Levenshtein.distance, workers=-1)
    return {"hn_d": nan(np.clip(num(hx, lx) - num(h1, l1), -1000, 1000)), "hn_lev": nan(lev), "hn_lendiff": nan(lx - l1)}


def context(a: pd.DataFrame, q1, qx, p, am, h1, hx, s3) -> pd.DataFrame:
    """CTX2 + CTX3 onto a (argmax rows: row, p, s1_n_am) from every #8c row of the country (q1, qx, p, am);
    h1 / hx / s3 = house numbers / S3 flag of a's rows. ctx.ctx2 / ctx.add3 with their (s1, sx) merges as row takes."""
    row = a.row.to_numpy()
    kx, k1 = pd.factorize(qx)[0], pd.factorize(q1)[0]
    a["sx_n_ge_t"] = np.bincount(kx, p >= T)[kx][row]
    a["sx_n_ge_05"] = np.bincount(kx, p >= 0.5)[kx][row]
    a["s1_ncand"] = np.bincount(k1)[k1][row]
    best = pd.Series(np.where(am, -1.0, p)).groupby(k1).transform("max").to_numpy()[row]
    a["s1_best_nonam"] = np.where(best < 0, np.nan, best)
    a["sx_psum"] = np.bincount(kx, p)[kx][row]
    del kx, k1
    a["g"] = g = pd.factorize(a.q1)[0]
    pa_ = a.p.to_numpy()
    a["s1_rank"] = a.groupby("g").p.rank(ascending=False, method="first").to_numpy() / a.s1_n_am.to_numpy()
    acc, has = pa_ >= T, hx != ""
    grp = pd.DataFrame({"g": g, "k": pd.factorize(pd.Series(hx))[0], "a": (acc & has).astype(int)})
    same = grp.groupby(["g", "k"]).a.transform("sum").to_numpy() - grp.a.to_numpy()
    a["n_same_hn_acc"] = np.where(has, same, np.nan).astype(np.float32)
    eq = (acc & has & (hx == h1)).astype(int)
    a["n_acc_hn_eq_s1"] = (pd.Series(eq).groupby(g).transform("sum").to_numpy() - eq).astype(np.float32)
    del grp, same, eq
    o = np.lexsort((-pa_, g))
    go, po = g[o], pa_[o]
    nxt, prv = np.r_[po[1:], np.nan], np.r_[np.nan, po[:-1]]
    nxt[np.r_[go[1:] != go[:-1], True]] = np.nan
    prv[np.r_[True, go[1:] != go[:-1]]] = np.nan
    gn, gp = np.empty(len(a)), np.empty(len(a))
    gn[o], gp[o] = po - nxt, prv - po
    a["s1_gap_next"], a["s1_gap_prev"] = gn, gp
    a["s1_psum"] = np.bincount(g, pa_)[g]
    a["is_s3"] = s3.astype(np.float32)
    n = np.zeros(len(a))
    for v in (False, True):
        m = s3 == v
        cnt = np.bincount(g[m], acc[m].astype(int), minlength=g.max() + 1)
        n[m] = cnt[g[m]] - acc[m]
    a["n_same_src_acc"] = n
    return a


def features(c: str, q1, qx, p, p_m2, r, id1: pa.Array, idx: pa.Array, zoo: bool = True) -> tuple:
    """#8c's test rows of one country -> (am, one row per argmax pair: row, p, lp, hc + the combiner's columns).
    q1 / qx: codes into id1 / idx (features.ids order); p = #8c's p after P17's stack, p_m2 = M2, r = band CE."""
    import stack as S
    q1, qx, p = np.asarray(q1, np.int64), np.asarray(qx, np.int64), np.asarray(p, np.float64)
    am, a = A.frame("test", c, q1, qx, p, p_m2, r, id1, idx)
    row = a.row.to_numpy()
    a["q1"], a["qx"] = q1[row], qx[row]
    a = a.drop(columns=["p2"])
    h1, hx = (S.hn_of("test", s, c, ids, q[row]) for s, ids, q in (("s1", id1, q1), ("sx", idx, qx)))
    for k, v in hn_cols(h1, hx).items():
        a[k] = v
    a["hn_cls"] = a.hc.astype(np.float32)                   # stack.hn_cls = solution_p44's class (P44 asserted)
    a["has_hn"] = a.hn_lev.notna().astype(np.float32)
    a["exact_hn"] = ((a.hn_cls == 0) & (a.has_hn > 0)).astype(np.float32)
    a["tw_any"] = (a.n_twins > 0).astype(np.float32)
    s3 = pc.starts_with(idx.take(pa.array(a.qx.to_numpy())), "S3-").to_numpy(zero_copy_only=False)
    a = context(a, q1, qx, p, am, h1, hx, s3)
    del h1, hx, s3
    k = keys(a.q1, a.qx)
    a["lq"] = lg(by_key(W / Q.format(c), k, ["q"]).q.to_numpy(np.float32))
    mem = by_key(W / MEMBERS.format(c), k, ["p_m2hn", "p_hn21"] + ([f"p_{m}" for m in MEM + RANK] if zoo else []))
    for m in ("m2hn", "hn21"):
        a[f"l_{m}"] = lg(mem[f"p_{m}"].to_numpy(np.float32))
    for m in (MEM if zoo else []):
        a[f"z_{m}"] = lg(mem[f"p_{m}"].to_numpy(np.float64)).astype(np.float32)
    for m in (RANK if zoo else []):
        a[f"z_{m}"] = mem[f"p_{m}"].to_numpy(np.float32)
    del mem
    # r_all: keyed by ids; its band pairs carry the pipeline's own r
    t = pq.read_table(W / R_ALL.format(c), columns=["s1", "sx", "r"])
    kr = [pc.index_in(t.column(s).cast(ids.type), value_set=ids).to_numpy(zero_copy_only=False)
          for s, ids in (("s1", id1), ("sx", idx))]
    ok = ~(pd.isna(kr[0]) | pd.isna(kr[1]))
    rk = pd.DataFrame({"k": keys(kr[0][ok].astype(np.int64), kr[1][ok].astype(np.int64)),
                       "r": t.column("r").to_numpy()[ok]}).drop_duplicates("k")
    del t, kr
    pos = pd.Index(rk.k).get_indexer(k)
    assert (pos >= 0).all(), f"r_all missing for {int((pos < 0).sum()):,} argmax pairs"
    r_all = rk.r.to_numpy(np.float64)[pos]
    band = a.r.notna().to_numpy()
    assert np.allclose(r_all[band], a.r.to_numpy()[band], atol=1e-6), "r_all band r != the pipeline's r"
    a["l_r_all"] = lg(r_all)
    return am, a


def boosters(tag: str) -> list:
    import lightgbm as lgb
    return [lgb.Booster(model_file=str(W / BOOSTERS[tag].format(s))) for s in SEEDS]


def booster_tag(tag: str) -> str:
    """Hash of the tag's 5 booster files (run.py's cache key)."""
    return hashlib.sha1(b"".join((W / BOOSTERS[tag].format(s)).read_bytes() for s in SEEDS)).hexdigest()[:8]


def combine(bs: list, a: pd.DataFrame, cols: list) -> np.ndarray:
    """te44.combine_p / e_zoo.probs: sigmoid of the mean logit of the guarded boosters, in row chunks."""
    out = np.empty(len(a))
    for i in range(0, len(a), CHUNK):
        x = a.iloc[i:i + CHUNK]
        outs = []
        for b in bs:
            assert b.feature_name() == cols, "booster columns != " + ("ZOO" if cols is ZOO else "FULL")
            s = sig(b.predict(x[cols], raw_score=True) + x.lp.to_numpy())
            outs.append(lg(np.where(x.hc.to_numpy() == 1, x.p.to_numpy(), s)))
        out[i:i + CHUNK] = sig(np.mean(outs, axis=0))
    return out


def apply(tag: str, split: str, c: str, q1, qx, p, p_m2, r, id1: pa.Array, idx: pa.Array) -> tuple:
    """The run.py step after P17's stack (a10_stage's place): #8c's rows of one country -> (p'' float32, am).
    p'' = the combiner on #8c's argmax pairs, #8c's p elsewhere; am = #8c's argmax, which the decision keeps
    (acc = am & p'' >= t, as P44 / P50 decided). tag: "fullzoo" (sub52_max) | "full" (sub50e)."""
    assert split == "test", "iu_stack: the boosters were fit on the whole holdout (test only)"
    am, a = features(c, q1, qx, p, p_m2, r, id1, idx, zoo=tag == "fullzoo")
    out = np.asarray(p, np.float32).copy()
    out[a.row.to_numpy()] = combine(boosters(tag), a, ZOO if tag == "fullzoo" else FULL).astype(np.float32)
    return out, am


def pack(c: str):
    """members_test_{c}: P50's member test preds keyed by (q1, qx) -- p_m2hn / p_hn21 from T_{c} (te44.build's
    reindex of pred_{m2hn,hn21}/test_{c}), the 16 zoo columns from zoo_test_{c} (positional to T_{c})."""
    t = pq.read_table(W / f"p50/T_{c}.parquet", columns=["q1", "qx", "p_m2hn", "p_hn21"])
    z = pq.read_table(W / f"p50/zoo_test_{c}.parquet", columns=[f"p_{m}" for m in MEM + RANK])
    assert t.num_rows == z.num_rows
    out = pa.table({"q1": t.column("q1").cast(pa.int32()), "qx": t.column("qx").cast(pa.int32()),
                    "p_m2hn": t.column("p_m2hn"), "p_hn21": t.column("p_hn21")} | {k: z.column(k) for k in z.column_names})
    (W / MEMBERS.format(c)).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, W / MEMBERS.format(c), row_group_size=500_000)
    print(c, out.num_rows, "argmax pairs packed ->", W / MEMBERS.format(c), flush=True)


if __name__ == "__main__" and sys.argv[1:] == ["pack"]:
    for c in ("India", "US"):
        pack(c)
elif __name__ == "__main__":
    # toy: S1 0 holds SX 0 (hn 12 = its own, S2) and SX 1 (hn 12, S3); S1 1 is SX 0's runner-up and SX 2's argmax
    a = pd.DataFrame({"row": [0, 2, 3], "q1": [0, 0, 1], "p": [0.9, 0.8, 0.3], "s1_n_am": [2, 2, 1]})
    q1, qx = np.array([0, 1, 0, 1]), np.array([0, 0, 1, 2])
    p, am = np.array([0.9, 0.6, 0.8, 0.3]), np.array([True, False, True, True])
    h1, hx, s3 = np.array(["12", "12", ""], object), np.array(["12", "12", ""], object), np.array([False, True, False])
    x = context(a, q1, qx, p, am, h1, hx, s3)
    assert x.sx_n_ge_t.tolist() == [1, 1, 0] and x.sx_n_ge_05.tolist() == [2, 1, 0] and x.s1_ncand.tolist() == [2, 2, 2]
    assert np.allclose(x.sx_psum, [1.5, 0.8, 0.3]) and np.isnan(x.s1_best_nonam[0]) and x.s1_best_nonam[2] == 0.6
    assert x.s1_rank.tolist() == [0.5, 1.0, 1.0] and x.n_same_hn_acc.tolist()[:2] == [1, 1] and np.isnan(x.n_same_hn_acc[2])
    assert x.n_acc_hn_eq_s1.tolist() == [1, 1, 0] and x.n_same_src_acc.tolist() == [0, 0, 0] and x.is_s3.tolist() == [0, 1, 0]
    assert np.allclose(x.s1_gap_next[:1], [0.1]) and np.isnan(x.s1_gap_next[1]) and np.allclose(x.s1_gap_prev[1], 0.1)
    assert np.allclose(x.s1_psum, [1.7, 1.7, 0.3])
    h = hn_cols(["10", "1795", "", "5"], ["13", "179", "4", "5"])
    assert h["hn_d"].tolist()[:2] == [3, -1000] and np.isnan(h["hn_d"][2]) and h["hn_lev"].tolist()[1] == 1
    assert h["hn_lendiff"].tolist()[1] == -1 and h["hn_d"][3] == 0
    B = lambda v: type("B", (), {"predict": lambda self, X, raw_score: np.full(len(X), v),
                                 "feature_name": lambda self: ["f"]})()
    f = pd.DataFrame({"f": [0.0, 0.0], "lp": [0.0, 1.0], "p": [0.5, 0.2], "hc": [0, 1]})
    assert np.allclose(combine([B(1.0), B(-1.0)], f, ["f"]), [0.5, 0.2])     # mean logit; guard keeps #8c p
    print("iu_stack self-check ok")
