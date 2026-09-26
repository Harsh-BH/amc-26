"""P40 France op rule for run.py (no op tagger): accept free, argmax, non-sibling, unaccepted France pairs with p >= t
whose SX is a generator variant the France threshold under-accepts:
  gib       : SX name has a token spelled by >= 2 of the 44 generator syllables and shares no token with the S1 name
  xu_noaddr : SX address empty, SX name core == S1 name core, no other France S1 has that core, core >= 5 chars
Caps 5 S2 / 6 S3 per S1 (existing members count), best p first. Reproduces loco41/build40's `xu_noaddr|gib@0.9`.
    .venv/bin/python probes/P40-generator/rule40.py          # self-check
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P36-iu-residual")]
import gib as G  # noqa: E402

SYLL = set(__import__("json").loads((ROOT / "data/work/p36/morph_train.json").read_text()))


def op_class(n1, nx, ax, c1, cx, core_count) -> np.ndarray:
    """n1 / nx: raw S1 / SX names; ax: raw SX address; c1 / cx: rf name cores; core_count: dict core -> #S1 (France)."""
    out = np.zeros(len(nx), bool)
    for i, (a, b, adr, u, v) in enumerate(zip(n1, nx, ax, c1, cx)):
        tb = G.toks(b)
        if tb and not set(tb) & set(G.toks(a)) and any(G.n_syll(t, SYLL) >= 2 for t in tb) and "." not in b and "@" not in b:
            out[i] = True
        elif not (adr or "").strip() and u == v and len(u) >= 5 and core_count.get(u, 0) == 1:
            out[i] = True
    return out


def add(s1, sx, is2, p, am, pred, sib, cls, t) -> np.ndarray:
    """Mask of pairs to add (never removes). s1 / sx: int codes."""
    free = np.bincount(sx[pred], minlength=int(sx.max()) + 1)[sx] == 0
    c = am & ~pred & free & ~sib & cls & (p >= t)
    ns = int(s1.max()) + 1
    n2 = np.bincount(s1[pred & is2], minlength=ns)[s1]
    n3 = np.bincount(s1[pred & ~is2], minlength=ns)[s1]
    o = np.flatnonzero(c)
    o = o[np.lexsort((-p[o], s1[o] * 2 + is2[o]))]
    k = s1[o] * 2 + is2[o]
    i = np.arange(len(o))
    rank = i - np.maximum.accumulate(np.where(np.r_[True, k[1:] != k[:-1]], i, 0))
    room = np.where(is2, 5 - n2, 6 - n3)
    m = np.zeros(len(p), bool)
    m[o[rank < room[o]]] = True
    return m


if __name__ == "__main__":
    got = op_class(["Baronnies Union SAS", "Club des Africa", "Club des Africa", "Deltabrix SAS"],
                   ["Deltabrix", "Club des Africa", "Club des Africa", "Deltabrix"], ["11 Rue X", "", "", "1 Rue Y"],
                   ["baronnies union", "club africa", "club africa", "deltabrix"],
                   ["deltabrix", "club africa", "club africa", "deltabrix"], {"club africa": 1})
    assert list(got) == [True, True, True, False], got
    s1 = np.array([0, 0, 1, 1]); sx = np.arange(4); is2 = np.array([1, 0, 1, 1], bool)
    p = np.array([.99, .95, .92, .5]); am = np.ones(4, bool); pred = np.array([1, 0, 0, 0], bool); sib = np.zeros(4, bool)
    assert list(np.flatnonzero(add(s1, sx, is2, p, am, pred, sib, np.ones(4, bool), .9))) == [1, 2]
    print("rule40 self-check ok")
