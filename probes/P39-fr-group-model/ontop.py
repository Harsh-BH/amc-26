"""P39 on top of the current France recipe: chain + #34v_r90 rescue (p >= 0.90) + P37 gc70 (oneall, p >= 0.70) =
sub37_gc70 (LB 0.986304). France t -> LOCO target t by P37's rule: the same number of argmax pairs per S1 in
[t, t_base). The model (scores cached by curve.py) then adds on top (add) or also removes (addrm, chain pairs with
s < 0.3). Paired dF per target S1 vs that base, both directions.

    .venv/bin/python probes/P39-fr-group-model/ontop.py <arm> [<arm> ...] [--hn90]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build import best_per  # noqa: E402
from curve import TAUS, target_scores  # noqa: E402
from fit import CAP, FR_SHARE, W, capped, load, per_s1  # noqa: E402
from frame import T_FR, T_LOCO  # noqa: E402

T_RESCUE, T_GC, T_HN = 0.90, 0.70, 0.90
HN90 = "--hn90" in sys.argv              # base = sub37_gc70_hn90 (P37's next upload) instead of sub37_gc70


def dens(p, t_base, t, n):
    return float(((p >= t) & (p < t_base)).sum() / n)


def inv_dens(p, t_base, g, n):
    v = np.sort(p[p < t_base])[::-1]
    k = int(round(g * n))
    return float(v[min(k, len(v)) - 1]) if k > 0 else t_base


def rescue(d, pred, t):
    s1, p = d.s1.to_numpy(), d.p.to_numpy()
    has = np.bincount(s1[pred], minlength=int(s1.max()) + 1)[s1] > 0
    return pred | best_per(s1, p, (d.hn_rel.to_numpy() != 2) & ~has & ~pred & (p >= t))


def gc(d, pred, t, hn_extra=False):
    """P37 gc_add 'oneall': every non-sibling pair of the missing source of a one-sided S1 with p >= t, caps 5 / 6.
    hn_extra: P37 'allhn' with the hn-equal guard (sub37_gc70_hn90's extras): also any non-empty S1, hn equal."""
    s1, is2, p = d.s1.to_numpy(), d.is2.to_numpy(), d.p.to_numpy()
    n = int(s1.max()) + 1
    n2, n3 = np.bincount(s1[pred & is2], minlength=n)[s1], np.bincount(s1[pred & ~is2], minlength=n)[s1]
    miss = np.where(is2, (n2 == 0) & (n3 > 0), (n3 == 0) & (n2 > 0))
    if hn_extra:
        miss = miss | ((n2 + n3 > 0) & (d.hn_rel.to_numpy() == 1))
    c = np.flatnonzero(miss & ~pred & (d.hn_rel.to_numpy() != 2) & (p >= t))
    c = c[np.lexsort((-p[c], s1[c] * 2 + is2[c]))]
    k = s1[c] * 2 + is2[c]
    i = np.arange(len(c))
    rank = i - np.maximum.accumulate(np.where(np.r_[True, k[1:] != k[:-1]], i, 0))
    room = np.where(is2[c], CAP[True] - n2[c], CAP[False] - n3[c])
    out = pred.copy()
    out[c[rank < room]] = True
    return out


def fr_density():
    f = pd.read_parquet(W / "france.parquet", columns=["p"])
    n = len(pd.read_parquet(W / "s1_france.parquet"))
    return {t: dens(f.p.to_numpy(), T_FR, t, n) for t in (T_RESCUE, T_GC)}          # T_HN = T_RESCUE


def bases(d, nt, src, tgt, g):
    p, acc = d.p.to_numpy(), d.acc.to_numpy()
    tb, n = T_LOCO[(src, tgt)], len(nt)
    tr, tg = inv_dens(p, tb, g[T_RESCUE], n), inv_dens(p, tb, g[T_GC], n)
    br = rescue(d, acc, tr)
    bg = gc(d, br, tg)
    return tr, tg, br, (gc(d, bg, tr, hn_extra=True) if HN90 else bg)


def run(arms):
    g = fr_density()
    print(f"France argmax pairs per S1 in [t, {T_FR}): {g}", flush=True)
    out = {}
    for dr in ("US>India", "India>US"):
        src, tgt = dr.split(">")
        d, nt, hv = load(src, tgt)
        y, acc = d.y.to_numpy(), d.acc.to_numpy()
        tr, tg, br, bg = bases(d, nt, src, tgt, g)
        f0 = per_s1(d, acc, nt)
        fr, fg = per_s1(d, br, nt), per_s1(d, bg, nt)
        a = bg & ~br
        out[dr] = {"t_rescue": tr, "t_gc": tg, "rescue_vs_chain": round(float((fr - f0).mean()), 5),
                   "gc_vs_rescue": round(float((fg - fr).mean()), 5), "gc_added": int(a.sum()), "gc_prec": round(float(y[a].mean()), 3)}
        print(f"== {dr}: t_rescue {tr:.4f} t_gc {tg:.4f} | rescue vs chain {(fr - f0).mean():+.5f} | gc70{'+hn90' if HN90 else ''} vs rescue "
              f"{(fg - fr).mean():+.5f} (+{a.sum():,} @ {y[a].mean():.3f}) | base F {fg.mean():.5f}", flush=True)
        d2 = d.assign(acc=bg)
        s1, is2 = d.s1.to_numpy(), d.is2.to_numpy()
        for arm in arms:
            s = target_scores(arm, dr)
            for mode in ("add", "addrm"):
                for t in TAUS:
                    keep = (bg | (s >= t)) if mode == "add" else ((bg & (s >= 0.3)) | (s >= t))
                    pred = capped(s1, is2, np.where(bg, s + 1, s), keep)      # caps never evict a base pair first
                    dl = per_s1(d, pred, nt) - fg
                    ad, rm = pred & ~bg, bg & ~pred
                    out[dr].setdefault(arm, {}).setdefault(mode, {})[t] = (
                        round(float(dl.mean()), 5), round(float(dl.std() / np.sqrt(len(dl))), 5), round(float(dl[hv == 1].mean()), 5),
                        int(ad.sum()), round(float(y[ad].mean()), 3) if ad.any() else None,
                        int(rm.sum()), round(float(y[rm].mean()), 3) if rm.any() else None)
    for arm in arms:
        print(f"=== {arm} on top of chain+rescue90+gc70{'+hn90' if HN90 else ''}: dF (se) h1 added@prec removed@prec", flush=True)
        for mode in ("add", "addrm"):
            for t in TAUS:
                u, i = out["US>India"][arm][mode][t], out["India>US"][arm][mode][t]
                print(f"  {mode:<5} tau {t:.2f} | US>India {u[0]:+.5f} ({u[1]:.5f}) h1 {u[2]:+.5f} +{u[3]:,}@{u[4]} -{u[5]:,}@{u[6]} | "
                      f"India>US {i[0]:+.5f} ({i[1]:.5f}) h1 {i[2]:+.5f} +{i[3]:,}@{i[4]} -{i[5]:,}@{i[6]} | min {min(u[0], i[0]):+.5f} "
                      f"(LB-eq {min(u[0], i[0]) * FR_SHARE:+.5f})", flush=True)
    json.dump({dr: {k: ({m: {str(t): x for t, x in vv.items()} for m, vv in v.items()} if isinstance(v, dict) else v)
                    for k, v in o.items()} for dr, o in out.items()}, open(W / f"ontop{'_hn90' if HN90 else ''}_{'_'.join(arms)}.json", "w"))
    return out


if __name__ == "__main__":
    run([a for a in sys.argv[1:] if not a.startswith("--")] or ["p_group", "full"])
