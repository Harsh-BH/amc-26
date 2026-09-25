"""P5-decide: pair probabilities -> one answer set per S1, for macro F0.5 per S1 (singletons included).

    from decide import decide, CFG
    sets = decide(pairs, CFG)       # pairs: DataFrame[s1, sx, p (, country)] -> {s1: set(sx)}
    .venv/bin/python probes/P5-decide/decide.py        # self-check

Steps, each switched by cfg (probes/P5-decide/RESULT.md has the evidence for every knob):
  calib   (x, y) knots of an isotonic map, applied with np.interp before anything else
  stack   D7: LightGBM model text (or its path); replaces p by a second-stage p computed from ctx() -- the pair's rank, gaps
          and sums among its S1's candidates and its SX's competing S1s (stack_cols picks the ctx columns)
  excl    many-to-one exclusivity: each SX survives only at its argmax-p S1 (argmax is exact -- no Hungarian),
          taken on the matcher's p (before the stack); an `am` column (P4's argmax over every train S1) overrides
          the in-frame argmax -- holdout frames hold only 10% of the competing S1s
  mode    "thresh": keep p >= t (t_country[country] if given, else t -- the unseen-country default);
                    t0 = singleton gate: the S1 answers nothing unless its best p >= t0, and its best pair is
                    admitted at t0 even when t > t0;  rel: keep only p >= rel * best p of the S1
          "ef":     per S1, the top-k set maximising expected F0.5 under independent calibrated p
                    (exact Poisson-binomial DP, Jansche 2007 / Ye et al. 2012; top-k by p is optimal there);
                    with t0, the gate decides empty-or-not and EF only picks k >= 1
  sib     D6: also admit pairs with p >= sib whose SX shares a non-empty `sib` column key with an answered SX
  m       cap the answer at the m highest-p pairs of the S1
"""
from pathlib import Path

import numpy as np
import pandas as pd

BETA2 = 0.25


def _codes(s: pd.Series) -> np.ndarray:
    return pd.factorize(s)[0]


def _gmax(g: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, -np.inf)
    np.maximum.at(out, g, v)
    return out


def _rank(g: np.ndarray, p: np.ndarray) -> np.ndarray:
    """0-based rank of each row by p (desc) within its group g; ties broken by row order."""
    o = np.lexsort((-p, g))
    gs = g[o]
    start = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    r = np.arange(len(g)) - np.repeat(start, np.diff(np.r_[start, len(g)]))
    out = np.empty(len(g), np.int64)
    out[o] = r
    return out


def ef_k(P: np.ndarray, miss: float = 0.0, empty: bool = True) -> np.ndarray:
    """P: (G, n) probabilities sorted descending per row. Returns (G,) the k in 0..n whose top-k set maximises
    E[F0.5] when each pair is true independently with its p, plus Poisson(miss) true SX outside the candidates."""
    G, n = P.shape
    L = n + 1 + (6 if miss > 0 else 0)
    F = np.zeros((n + 1, G, n + 1))            # F[k][:, a] = P(a true among the top k)
    F[0, :, 0] = 1
    for k in range(n):
        q = P[:, k:k + 1]
        F[k + 1] = F[k] * (1 - q)
        F[k + 1, :, 1:] += F[k, :, :-1] * q
    B = np.zeros((n + 1, G, L))                # B[k][:, b] = P(b true outside the top k)
    if miss > 0:
        j = np.arange(L)
        B[n] = np.exp(-miss + j * np.log(miss) - np.cumsum(np.log(np.maximum(j, 1))))[None, :]
    else:
        B[n, :, 0] = 1
    for k in range(n - 1, -1, -1):
        q = P[:, k:k + 1]
        B[k] = B[k + 1] * (1 - q)
        B[k, :, 1:] += B[k + 1, :, :-1] * q
    E = np.empty((n + 1, G))
    E[0] = B[0, :, 0] if empty else -1.0       # answer nothing: right only if nothing is true
    a, b = np.arange(n + 1)[:, None], np.arange(L)[None, :]
    for k in range(1, n + 1):
        W = (1 + BETA2) * a / (k + BETA2 * (a + b))
        E[k] = np.einsum("ga,ab,gb->g", F[k], W, B[k])
    return E.argmax(0)


def ef_select(g: np.ndarray, p: np.ndarray, cap: int = 24, miss: float = 0.0, chunk: int = 20_000,
              empty: bool = True) -> np.ndarray:
    """Row mask of the expected-F0.5-optimal set per group g. Only the top `cap` p of a group are considered."""
    r = _rank(g, p)
    sel = np.zeros(len(g), bool)
    top = np.flatnonzero(r < cap)
    gt, rt, pt = g[top], r[top], p[top]
    ng = np.bincount(gt, minlength=g.max() + 1)
    size = ng[gt]
    for n in np.unique(size):                  # bucket groups by candidate count: no padding
        m = size == n
        rows, gg, rr = top[m], gt[m], rt[m]
        u, gi = np.unique(gg, return_inverse=True)
        P = np.zeros((len(u), n))
        P[gi, rr] = pt[m]
        kbest = np.concatenate([ef_k(P[i:i + chunk], miss, empty) for i in range(0, len(u), chunk)])
        sel[rows] = rr < kbest[gi]
    return sel


def _gsum(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.bincount(g, weights=v)[g]


CTX = ["p", "r1", "n1", "max1", "rel1", "gap1", "sum1", "c50", "c20", "prev1", "next1", "rx", "nx", "maxox",
       "p2", "p3", "p4", "p5"]
S1_CTX = [c for c in CTX if c not in ("rx", "nx", "maxox")]   # S1-side only: exact in S1 chunks given `am`


def ctx(p: np.ndarray, g: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Context of each pair from the (s1, sx, p) table alone: its standing among the S1's candidates and among
    the SX's competing S1s. Features for the D7 second stage (no raw text, no country -> France-safe)."""
    r1 = _rank(g, p)
    o = np.lexsort((r1, g))
    ps, gs = p[o], g[o]
    same_prev = np.r_[False, gs[1:] == gs[:-1]]
    same_next = np.r_[gs[:-1] == gs[1:], False]
    prev_, next_ = np.empty(len(p)), np.empty(len(p))
    prev_[o] = np.where(same_prev, np.r_[0, ps[:-1]], 1.0)          # next-higher p in the S1 (1 if top)
    next_[o] = np.where(same_next, np.r_[ps[1:], 0], 0.0)           # next-lower p in the S1 (0 if last)
    max1 = _gmax(g, p, g.max() + 1)[g]
    rx = _rank(x, p)
    top_x = _gmax(x, p, x.max() + 1)[x]
    # best competing S1 for this SX: the SX max unless this pair is it, then the SX's second-best
    second = _gmax(x, np.where(rx == 1, p, -1.0), x.max() + 1)[x]
    maxox = np.where(rx == 0, np.maximum(second, 0.0), top_x)
    one = np.ones(len(p))
    cols = [lambda: p, lambda: r1, lambda: _gsum(g, one), lambda: max1, lambda: p / np.maximum(max1, 1e-9),
            lambda: max1 - p, lambda: _gsum(g, p), lambda: _gsum(g, (p >= 0.5) * 1.0),
            lambda: _gsum(g, (p >= 0.2) * 1.0), lambda: prev_, lambda: next_, lambda: rx, lambda: _gsum(x, one),
            lambda: maxox] + [lambda j=j: np.maximum(_gmax(g, np.where(r1 == j, p, -1.0), g.max() + 1)[g], 0.0)
                              for j in (1, 2, 3, 4)]
    X = np.empty((len(p), len(cols)), np.float32)   # filled column by column: no N x 18 float64 temporary
    for i, f in enumerate(cols):
        X[:, i] = f()
    return X


def select(d: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Row mask over d[s1, sx, p (, country)]: the pairs to answer."""
    p = d.p.to_numpy(np.float64)
    if cfg.get("calib") is not None:
        p = np.interp(p, *cfg["calib"])
    g, x = _codes(d.s1), _codes(d.sx)
    keep = np.ones(len(d), bool)
    if cfg.get("excl"):                        # on the matcher's p; `am` = precomputed argmax flag when the frame
        keep = d.am.to_numpy(bool) if "am" in d else p >= _gmax(x, p, x.max() + 1)[x]   # lacks competing S1s
    if cfg.get("stack"):                       # D7: second-stage p from the pair's context in the (s1, sx, p) table
        import lightgbm as lgb
        ms = cfg["stack"]                     # model text, or a path to it (cfg_*.json stores the path)
        booster = lgb.Booster(model_str=ms if ms.startswith("tree") else Path(ms).read_text())
        cols = cfg.get("stack_cols", CTX)
        X = ctx(p, g, x)
        if list(cols) != CTX:
            X = X[:, [CTX.index(c) for c in cols]]
        p = np.concatenate([booster.predict(X[i:i + 1_000_000]) for i in range(0, len(X), 1_000_000)])
    ng = g.max() + 1 if len(g) else 0
    pe = np.where(keep, p, 0.0)
    if cfg.get("mode", "thresh") == "ef":
        gate = cfg.get("t0") is not None     # t0: the gate decides empty vs not, EF only picks k >= 1
        sel = ef_select(g, pe, cfg.get("cap", 24), cfg.get("miss", 0.0), empty=not gate) & keep & (pe > 0)
        if gate:
            sel &= _gmax(g, pe, ng)[g] >= cfg["t0"]
    else:
        t = np.full(len(d), cfg["t"])
        if cfg.get("t_country"):
            t = d.country.map(cfg["t_country"]).fillna(cfg["t"]).to_numpy(np.float64)
        best = _gmax(g, pe, ng)[g]
        sel = pe >= t
        if cfg.get("rel"):
            sel &= pe >= cfg["rel"] * best
        if cfg.get("t0") is not None:
            top1 = _rank(g, pe) == 0
            sel = (sel | top1) & (best >= cfg["t0"])
        sel &= keep
    if cfg.get("sib") is not None:             # D6: admit an SX whose sibling key matches an answered SX
        k = pd.factorize(d.sib)[0].astype(np.int64)           # of the same S1 (NaN/empty key -> no sibling)
        ok = (k >= 0) & (d.sib.fillna("").to_numpy() != "")
        gk = g.astype(np.int64) * (k.max() + 2) + k
        sel |= keep & ok & (pe >= cfg["sib"]) & np.isin(gk, gk[sel & ok])
    if cfg.get("m"):
        sel &= _rank(g, np.where(sel, pe, -1.0)) < cfg["m"]
    return sel


def decide(pairs: pd.DataFrame, cfg: dict) -> dict:
    """{s1: set(sx)} for every S1 in pairs (empty set = no match). S1s absent from pairs are the caller's to add."""
    sel = select(pairs, cfg)
    out = {s: set() for s in pairs.s1.unique()}
    for a, b in zip(pairs.s1.to_numpy()[sel], pairs.sx.to_numpy()[sel]):
        out[a].add(b)
    return out


# Winner on P4's M2 holdout p (report half 0.9824 vs 0.9814 for P4's argmax + t 0.71; RESULT.md). The stack is fitted
# to M2's p: refit it (bench_stack.py p4 D7-stack-s1only) whenever the matcher changes.
CFG = {"excl": True, "mode": "ef", "t0": 0.55, "cap": 24,
       "stack": str(Path(__file__).with_name("stack_p4_s1only.txt")), "stack_cols": S1_CTX}
FALLBACK = {"excl": True, "t": 0.71, "t0": 0.72}         # no stack file / new matcher not yet restacked: 0.9815


def _check():
    import itertools
    rng = np.random.default_rng(0)
    # ef_k equals brute force over every subset size on a small instance (independent Bernoulli truth)
    P = np.sort(rng.random((5, 4)), 1)[:, ::-1].copy()
    for row, k in zip(P, ef_k(P)):
        ev = []
        for kk in range(5):
            e = 0.0
            for z in itertools.product([0, 1], repeat=4):
                pr = np.prod([q if t else 1 - q for q, t in zip(row, z)])
                tp, nt = sum(z[:kk]), sum(z)
                f = (1.0 if kk == 0 else 0.0) if nt == 0 else (1 + BETA2) * tp / (BETA2 * nt + kk)
                e += pr * f
            ev.append(e)
        assert k == int(np.argmax(ev)), (row, k, ev)
    d = pd.DataFrame({"s1": ["a", "a", "a", "b", "b", "c"], "sx": ["x", "y", "z", "x", "w", "v"],
                      "p": [0.9, 0.6, 0.2, 0.95, 0.3, 0.1], "country": ["US", "US", "US", "US", "US", "FR"]})
    s = decide(d, {"excl": True, "t": 0.5})
    assert s == {"a": {"y"}, "b": {"x"}, "c": set()}, s          # x goes to its argmax S1 only
    s = decide(d, {"t": 0.5, "t0": 0.92})
    assert s == {"a": set(), "b": {"x"}, "c": set()}, s          # gate: a's best 0.9 < 0.92
    s = decide(d, {"t": 0.25, "t_country": {"US": 0.5}})
    assert s["c"] == set() and s["a"] == {"x", "y"}, s           # FR falls back to the default t
    s = decide(d, {"t": 0.0, "m": 1})
    assert s == {"a": {"x"}, "b": {"x"}, "c": {"v"}}, s
    s = decide(d.assign(sib=["k", "k", "", "j", "k", None]), {"t": 0.5, "t0": 0.5, "sib": 0.1})
    assert s == {"a": {"x", "y"}, "b": {"x"}, "c": set()}, s     # z has no key; b's w is not x's sibling
    s = decide(d, {"mode": "ef"})
    assert s["b"] == {"x"} and s["c"] == set(), s
    print("decide self-check ok")


if __name__ == "__main__":
    _check()
