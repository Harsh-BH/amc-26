"""P54 idea 1 vs P55: on LOCO (recipe - P48 veto), rebuild P55's shipped France decision from its saved target scores
(cb_plain_<dr>.npy: band veto c < 0.6, free adds c > 0.95 up to caps) and measure the P54 rescue (band, r >= u) on top of
it, plus the overlap of the two add sets."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import loco54  # noqa: E402

L, S = loco54.L, loco54.S
W = HERE.parents[1] / "data/work/p55"


def add(z, base, cand_mask, score, n):
    s1, is2 = z["s1"].astype(np.int64), z["is2"]
    n2, n3 = S.counts(s1, is2, base, n)
    c = np.flatnonzero(cand_mask & ~base)
    o, rk = S.rank_in(s1 * 2 + is2, score, c)
    take = o[rk < np.where(is2[o], L.CUR["cap"][0] - n2[s1[o]], L.CUR["cap"][1] - n3[s1[o]])]
    out = base.copy()
    out[take] = True
    return out


out = {}
for dr in L.DIRS:
    z = L.frame(dr)
    g, n = z["g"], len(z["ntrue"])
    r = np.nan_to_num(z["r"], nan=0.0)
    band = (g > S.LO) & (g < S.HI)
    p = S.loco_p(g, z["r"], L.CUR["tau"], "chain")
    st, tm = L.recipe(z, dr, p)
    ref = st["hn"] & ~(st["gc"] & ~st["base"] & band & (np.nan_to_num(z["r"], nan=1.0) < 0.2))
    free = L.free_mask(z, st, p, tm)
    c = np.load(W / f"cb_plain_{dr}.npy")
    assert len(c) == len(g)
    fref = L.F(z, ref)
    v55 = ref & ~(band & (c < 0.6))
    d55 = add(z, v55, free & (c > 0.95), c, n)
    f55 = L.F(z, d55)
    R = out[dr] = {"p55_vs_ref": L.cmp(z, ref, d55, fref), "p55veto_vs_ref": L.cmp(z, ref, v55, fref)}
    for u in (0.99, 0.995):
        mine = add(z, ref, free & band & (r >= u), r, n)
        both = add(z, d55, free & band & ~ref & (r >= u), r, n)       # as fr54: recipe pairs (P55-vetoed ones too) never re-added
        a54, a55 = mine & ~ref, d55 & ~ref
        R[f"u{u}"] = {"p54_vs_ref": L.cmp(z, ref, mine, fref), "p54_on_p55": L.cmp(z, d55, both, f55),
                      "both_vs_ref": L.cmp(z, ref, both, fref),
                      "overlap": {"p54_adds": int(a54.sum()), "p55_adds": int(a55.sum()), "common": int((a54 & a55).sum()),
                                  "p54_only_prec": round(float(z["y"][a54 & ~a55].mean()), 3) if (a54 & ~a55).any() else None,
                                  "p54_adds_vetoed_by_p55": int((a54 & band & (c < 0.6)).sum())}}
    print(dr, json.dumps(R), flush=True)
(HERE / "on55.json").write_text(json.dumps(out, indent=1))
