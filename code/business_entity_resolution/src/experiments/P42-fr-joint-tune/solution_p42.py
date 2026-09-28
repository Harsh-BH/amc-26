"""P42: the joint France-lever grid on LOCO, shipped alone as solution.py (sm_train calls main()). The local driver
(p42.py) imports the same functions, so the France build and the grid use one implementation.

Recipe (France's order): p(tau) -> accept argmax & p >= t -> cap (an S1 over c2 S2 / c3 S3 loses its lowest pairs)
  -> rescue (empty S1: best argmax non-sibling pair, p >= t_r) -> gc (one-sided S1: every missing-source argmax
  non-sibling pair, p >= t_gc, room = cap) -> hn extras (non-empty S1: hn-equal argmax non-sibling pairs, p >= t_hn).
LOCO = P39's France chain analogue (P6 source-only M2 g, P30 source-only bge r, P13 France blend weights, veto tau)
at P30's label-free t_base, or P37's M2-only analogue (p = g, t_base = P6 empty-rate t). Every France threshold x maps
to the LOCO threshold with the same signed density of argmax pairs per S1 between x and the anchor (France 0.9868 <->
t_base), computed at the cell's tau on both sides (P37's rule, extended above the anchor).

Channel 'train': loco_<src>_<tgt>.npz (prep.py) + grid.json (levers, France densities, shard plan). params: shard.
Output: cells_<shard>.jsonl, one line per (analogue, direction, cell).
"""
import itertools
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

LO, HI, W_FR = 0.02, 0.98, (0.9245, 0.2595, 0.0515)
T_FR = 0.9868
T_BASE = {"chain": {"US_India": 0.915, "India_US": 0.92}, "m2": {"US_India": 0.97, "India_US": 0.81}}
OFF = None


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def loco_p(g, r, tau, analogue):
    """P39 frame.blend_veto with tau free (chain), or the raw M2 p with the same veto (m2)."""
    p = g.astype(np.float64).copy()
    if analogue == "chain":
        b = (g > LO) & (g < HI)
        rb = np.where(np.isnan(r), 0.5, r)
        p[b] = 1 / (1 + np.exp(-(W_FR[0] * logit(g[b]) + W_FR[1] * logit(rb[b]) + W_FR[2])))
    return np.where((g >= HI) & (np.nan_to_num(r, nan=1.0) < tau), 0.0, p)


def argmax_sx(qx, p):
    o = np.lexsort((-p, qx))
    m = np.zeros(len(p), bool)
    m[o[np.r_[True, qx[o][1:] != qx[o][:-1]]] if len(o) else o] = True
    return m


def sdens(p, am, anchor, x, n):
    """Signed argmax pairs per S1 between x and the anchor: + below it ([x, anchor)), - above it ([anchor, x))."""
    q = p[am]
    return float(((q >= x) & (q < anchor)).sum() / n) if x <= anchor else -float(((q >= anchor) & (q < x)).sum() / n)


def inv_sdens(p, am, anchor, g, n):
    q = p[am]
    k = int(round(abs(g) * n))
    if k == 0:
        return anchor
    if g > 0:
        v = np.sort(q[q < anchor])[::-1]
        return float(v[min(k, len(v)) - 1])
    v = np.sort(q[q >= anchor])
    return float(v[k]) if k < len(v) else 1.0 + 1e-9


def rank_in(key, p, rows):
    """rows sorted by (key, -p) and each row's 0-based rank within its key."""
    o = rows[np.lexsort((-p[rows], key[rows]))]
    k = key[o]
    i = np.arange(len(o))
    return o, i - np.maximum.accumulate(np.where(np.r_[True, k[1:] != k[:-1]], i, 0))


def counts(s1, is2, pred, n):
    return np.bincount(s1[pred & is2], minlength=n), np.bincount(s1[pred & ~is2], minlength=n)


def base(s1, is2, p, am, sib, t, cap):
    """Accept argmax & p >= t, then the cap (lowest p over c2 S2 / c3 S3 per S1 leave, and their SX leaves every later
    stage: France's cap zeroes the SX). Returns (accepted, free = pairs a later stage may add)."""
    key = s1.astype(np.int64) * 2 + is2
    acc = am & (p >= t)
    o, rk = rank_in(key, p, np.flatnonzero(acc))
    over = o[rk >= np.where(is2[o], cap[0], cap[1])]
    acc[over] = False
    free = am & ~sib
    free[over] = False
    return acc, free


def add(stage, s1, is2, p, pred, free, hn_eq, n, bar, cap):
    """One additive stage on pred: rescue (empty S1: its best free pair, p >= bar), gc (one-sided S1: free pairs of the
    missing source) or hn (non-empty S1: free hn-equal pairs); gc / hn fill up to the cap, best p first."""
    if bar is None:
        return pred
    n2, n3 = counts(s1, is2, pred, n)
    m2, m3 = n2[s1], n3[s1]
    ok = {"rescue": m2 + m3 == 0, "gc": np.where(is2, (m2 == 0) & (m3 > 0), (m3 == 0) & (m2 > 0)),
          "hn": (m2 + m3 > 0) & hn_eq}[stage]
    c = np.flatnonzero(free & ~pred & ok & (p >= bar))
    if stage == "rescue":
        o, rk = rank_in(s1.astype(np.int64), p, c)
        take = o[rk == 0]
    else:
        o, rk = rank_in(s1.astype(np.int64) * 2 + is2, p, c)
        take = o[rk < np.where(is2[o], cap[0] - m2[o], cap[1] - m3[o])]
    pred = pred.copy()
    pred[take] = True
    return pred


def decide(s1, is2, p, am, sib, hn_eq, n, t, cap, t_r=OFF, t_gc=OFF, t_hn=OFF, steps=False):
    """The France recipe (see the module doc): base -> rescue -> gc -> hn. cap = (c2, c3). pred, or every stage."""
    out = {}
    out["base"], free = base(s1, is2, p, am, sib, t, cap)
    pred = out["base"]
    for stage, bar in (("rescue", t_r), ("gc", t_gc), ("hn", t_hn)):
        out[stage] = pred = add(stage, s1, is2, p, pred, free, hn_eq, n, bar, cap)
    return out if steps else pred


def fbeta(tp, npred, ntrue):
    pr, rc = tp / np.maximum(npred, 1), tp / np.maximum(ntrue, 1)
    f = np.where(tp > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0)
    return np.where(ntrue == 0, (npred == 0).astype(float), f)


# ---------------------------------------------------------------- the grid on LOCO
def cells(G):
    L = G["levers"]
    return list(itertools.product(L["tau"], L["t"], [tuple(c) for c in L["cap"]], L["rescue"], L["gc"], L["hn"]))


def unit_list(G):
    """Work units: (analogue, direction, tau, t, cap); a unit evaluates every (rescue, gc, hn) under it."""
    L = G["levers"]
    return list(itertools.product(G["analogues"], G["directions"], L["tau"], L["t"], [tuple(c) for c in L["cap"]]))


_Z = {}


def _frame(data, dr):
    if dr not in _Z:
        z = np.load(Path(data) / f"loco_{dr}.npz")
        _Z[dr] = {k: z[k] for k in z.files}
    return _Z[dr]


def fr_to_loco(G, analogue, dr, tau, p, am, n):
    """France threshold -> LOCO threshold at this tau (density map, anchored at T_FR <-> t_base)."""
    tb = T_BASE[analogue][dr]
    dens = G["fr_dens"][str(tau)]
    return {x: inv_sdens(p, am, tb, g, n) for x, g in dens.items()}


def run_unit(args):
    data, G, (analogue, dr, tau, t, cap) = args
    z = _frame(data, dr)
    s1, is2, y, n = z["s1"].astype(np.int64), z["is2"], z["y"], len(z["ntrue"])
    ntrue, hv = z["ntrue"].astype(np.int64), z["hv"]
    sib, hn_eq = z["hn_rel"] == 2, z["hn_rel"] == 1
    am = np.ones(len(s1), bool)                                      # the frame holds SX-argmax pairs only
    p = loco_p(z["g"], z["r"], tau, analogue)
    tm = fr_to_loco(G, analogue, dr, tau, p, am, n)
    L = G["levers"]
    mp = lambda x: None if x is None else tm[str(x)]
    ref = G.get("_ref")
    rows = []
    acc, free = base(s1, is2, p, am, sib, mp(t), cap)
    for t_r in L["rescue"]:
        pr = add("rescue", s1, is2, p, acc, free, hn_eq, n, mp(t_r), cap)
        for t_gc in L["gc"]:
            pg = add("gc", s1, is2, p, pr, free, hn_eq, n, mp(t_gc), cap)
            for t_hn in L["hn"]:
                pred = add("hn", s1, is2, p, pg, free, hn_eq, n, mp(t_hn), cap)
                tp, npd = np.bincount(s1[pred & y], minlength=n), np.bincount(s1[pred], minlength=n)
                f = fbeta(tp, npd, ntrue)
                rows.append({"an": analogue, "dir": dr, "tau": tau, "t": t, "cap": list(cap), "r": t_r, "gc": t_gc, "hn": t_hn,
                             "F": round(float(f.mean()), 7), "F0": round(float(f[hv == 0].mean()), 7),
                             "F1": round(float(f[hv == 1].mean()), 7), "npred": int(pred.sum()), "tp": int(tp.sum()),
                             "t_loco": round(mp(t), 5), "sd_ref": round(float((f - ref[(analogue, dr)]).std() / np.sqrt(n)), 7)})
    return rows


def ref_f(data, G, analogue, dr):
    """Per-S1 F of the uploaded recipe (sub37_gc70) on this analogue: SE of every cell's paired difference."""
    z = _frame(data, dr)
    s1, is2, y, n = z["s1"].astype(np.int64), z["is2"], z["y"], len(z["ntrue"])
    c = G["current"]
    p = loco_p(z["g"], z["r"], c["tau"], analogue)
    am = np.ones(len(s1), bool)
    tm = fr_to_loco(G, analogue, dr, c["tau"], p, am, n)
    mp = lambda x: None if x is None else tm[str(x)]
    pred = decide(s1, is2, p, am, z["hn_rel"] == 2, z["hn_rel"] == 1, n, mp(c["t"]), tuple(c["cap"]), mp(c["r"]), mp(c["gc"]), mp(c["hn"]))
    return fbeta(np.bincount(s1[pred & y], minlength=n), np.bincount(s1[pred], minlength=n), z["ntrue"].astype(np.int64))


def run_grid(data, G, shard, nshard, out, procs=None):
    t0 = time.time()
    units = unit_list(G)[shard::nshard]
    G = dict(G, _ref={(a, d): ref_f(data, G, a, d) for a in G["analogues"] for d in G["directions"]})
    _Z.clear()
    n = 0
    with Pool(procs or os.cpu_count()) as pool, open(out, "w") as fo:
        for rows in pool.imap_unordered(run_unit, [(data, G, u) for u in units]):
            for r in rows:
                fo.write(json.dumps(r) + "\n")
            n += len(rows)
            print(f"[p42] {n:,} cells, {time.time() - t0:.0f}s", flush=True)
    return n


def main(train_dir, test_dir, model_dir, out_dir, params):
    G = json.loads((Path(train_dir) / "grid.json").read_text())
    shard, nshard = int(params.get("shard", 0)), int(params.get("nshard", 1))
    if params.get("smoke"):                                          # 2 units only: the whole I/O path in a minute
        G["levers"] = dict(G["levers"], t=G["levers"]["t"][:1], cap=G["levers"]["cap"][:1], tau=G["levers"]["tau"][:1])
        G["analogues"], G["directions"] = G["analogues"][:1], G["directions"]
    out = Path(out_dir) / f"cells_{shard}.jsonl"
    n = run_grid(train_dir, G, shard, nshard, out)
    (Path(model_dir) / f"cells_{shard}.jsonl").write_bytes(out.read_bytes())
    print(f"[p42] shard {shard}/{nshard}: {n:,} rows -> {out}", flush=True)


def _selfcheck():
    s1 = np.array([0, 1, 1, 0, 0, 1, 0]); is2 = np.array([0, 1, 0, 1, 1, 1, 0], bool)
    p = np.array([.99, .99, .99, .92, .95, .91, .93]); am = np.ones(7, bool); z = np.zeros(7, bool)
    st = decide(s1, is2, p, am, z, z, 2, .98, (5, 6), None, .9, None, steps=True)
    assert list(np.flatnonzero(st["gc"] & ~st["base"])) == [3, 4]                    # P37 oneall toy
    hn = np.r_[[False] * 6, True]
    st = decide(s1, is2, p, am, z, hn, 2, .98, (5, 6), None, .9, .9, steps=True)
    assert list(np.flatnonzero(st["hn"] & ~st["base"])) == [3, 4, 6]                 # P37 allhn toy
    st = decide(s1, is2, p, am, z, z, 2, .98, (1, 1), None, .9, None, steps=True)
    assert list(np.flatnonzero(st["gc"] & ~st["base"])) == [4]                       # cap 1: best S2 only
    st = decide(s1, is2, p, am, z, z, 2, .9, (1, 1), None, None, None, steps=True)
    assert list(np.flatnonzero(st["base"])) == [0, 1, 2, 4]          # cap 1/1 keeps the best S2 and S3 per S1
    assert decide(s1 * 0 + np.r_[0, 0, 0, 0, 0, 0, 1], is2, p, am, z, z, 2, .995, (5, 6), .9).tolist() == [True] + [False] * 5 + [True]   # rescue: best per empty S1
    g, r = np.array([.99, .99, .5, .01]), np.array([.05, .5, np.nan, np.nan])
    assert np.allclose(loco_p(g, r, 0.1, "chain"), [0.0, 0.99, 1 / (1 + np.exp(-W_FR[2])), 0.01])     # = P39 blend_veto
    q = np.array([.1, .5, .6, .7, .9, .95, .99]); a = np.ones(7, bool)
    assert sdens(q, a, .9, .6, 7) == 2 / 7 and sdens(q, a, .9, .96, 7) == -2 / 7
    assert inv_sdens(q, a, .9, 2 / 7, 7) == .6 and inv_sdens(q, a, .9, -2 / 7, 7) == .99
    print("p42 self-check ok")


if __name__ == "__main__":
    _selfcheck()
