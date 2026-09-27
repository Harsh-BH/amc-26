"""P62 idea 3 -- clone09 density-robustness check (idea from a public repo's write-up; re-implemented on P57's frames).

The holdout world is ~1.9x sparser in distractors than test. clone09: every unmatched SX of the #8c holdout world
(an SX in no train truth list) gets Poisson(lam) exact clones -- same text, same per-pair p / r / hn / zoo p on every
S1 row it has -- then everything context-dependent is re-derived exactly as the test path does (argmax, a10 columns,
#34's p', P44 CTX2 / CTX3) via P57's world(). The shipped recipes (boosters fit on half h of the FULL world, P57's
cache) score half 1-h of the clone world; per-S1 F0.5 vs #34 and #8c, paired SE, LB-eq weights.
    .venv/bin/python probes/P62-public-ideas/clone09.py build <c> [lam] [seed] [mode]   # -> data/work/p62/A_<w>_<c>.parquet
    .venv/bin/python probes/P62-public-ideas/clone09.py eval [lam] [seed] [mode]
mode "pois" (default): Poisson(lam) clones per free SX; "bern": one clone with probability lam.
ponytail: m2's own input context (inside #8c's p) is not recomputed -- only the stacker-level context is.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["P44_VARIANT"] = "r"
sys.path[:0] = ["probes/P44-meta-stack", "probes/P50-idea-batch", "probes/P57-iu-transfer", "."]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ev as E  # noqa: E402
import p57  # noqa: E402

OUT = E.W / "p62"
OUT.mkdir(exist_ok=True)
wname = lambda lam, seed, mode: f"clone{mode}{float(lam):g}s{seed}"


def clone(d: pd.DataFrame, lam: float, seed: int, mode: str) -> pd.DataFrame:
    """Append clone rows: every row of a free SX (ofold -1) repeated n times, n ~ Poisson(lam) (or Bernoulli) per SX."""
    free = d.ofold.to_numpy() < 0
    kx, ux = pd.factorize(d.sx[free])
    rng = np.random.default_rng(seed)
    n = rng.poisson(lam, len(ux)) if mode == "pois" else (rng.random(len(ux)) < lam).astype(np.int64)
    ix = np.flatnonzero(free)
    rep = n[kx]
    src = np.repeat(ix, rep)                                # row of d to copy, once per clone
    k = np.concatenate([np.arange(r) for r in rep]) + 1 if rep.sum() else np.zeros(0, np.int64)
    cl = d.iloc[src].copy()
    assert not cl.y.any()
    cl["sx"] = cl.sx.to_numpy(object) + "~" + k.astype(str)
    print(f"free SX {len(ux):,} ({free.mean():.3f} of rows), clones {n.sum():,}, clone rows {len(cl):,} "
          f"(+{len(cl) / len(d):.3f})", flush=True)
    return pd.concat([d, cl], ignore_index=True)


def build(c: str, lam="0.9", seed="0", mode="pois"):
    from solution_p44 import HnTab
    import a10
    lam, seed = float(lam), int(seed)
    t0 = time.time()
    d = clone(p57.rows(c), lam, seed, mode)
    k1, u1 = pd.factorize(d.s1)
    kx, ux = pd.factorize(d.sx)
    ids = pd.read_parquet(E.W / f"p23/s1ctx_{c}.parquet", columns=["n_true"]).index
    gi = pd.Series(np.arange(len(ids)), index=ids).reindex(d.s1).to_numpy()
    gi = np.where(np.isnan(gi), -1, gi).astype(np.int64)
    orig = pd.Series(np.asarray(ux, object)).str.split("~").str[0].to_numpy(object)   # a clone reads its original's text
    s1t, sxt = a10.texts("train", c, np.asarray(u1, object), orig)
    d = d.drop(columns=["s1"])
    S = pd.read_parquet(E.OUT / f"S1_{c}.parquet")                 # E.load would also read the big A frame
    hn = HnTab(E.W / "p44ms/hn", "train")
    a = p57.world(d, s1t, sxt, k1, kx, gi, S, hn, c, tuple(range(10)))
    a["clone"] = pd.Series(np.asarray(ux, object)[a.sx.to_numpy()]).str.contains("~").to_numpy()
    w = wname(lam, seed, mode)
    a.to_parquet(OUT / f"A_{w}_{c}.parquet", index=False)
    print(c, w, f"{len(a):,} argmax pairs ({a.clone.sum():,} clones, acc#8c among clones "
                f"{(a.p.to_numpy()[a.clone.to_numpy()] >= E.T).sum():,}); mean n_comp_rows {a.n_comp_rows.mean():.3f}, "
                f"s1_n_am {a.s1_n_am.mean():.3f} | {time.time() - t0:.0f}s", flush=True)


def evaluate(lam="0.9", seed="0", mode="pois"):
    import ev57 as V
    w = wname(float(lam), int(seed), mode)
    t0 = time.time()
    S = V.load_s()
    D = {}
    for c in E.C:                                           # ev57.load, from this probe's frame
        A = V.te44.derive(pd.read_parquet(OUT / f"A_{w}_{c}.parquet"))
        for m in V.Z.MEM:
            A[f"z_{m}"] = V.MT.lg(A[f"p_{m}"].to_numpy())
        for m in V.Z.RANK:
            A[f"z_{m}"] = A[f"p_{m}"].to_numpy(np.float64)
        D[c] = V.relcols(A)[V.KEEP]
    res = {}
    for rep in (0, 1):
        r = V.report(D, S, {c: D[c].pa.to_numpy() for c in E.C}, rep)
        res[f"{w}|yard|34|rep{rep}"] = r
        print(f"{w} #34 vs #8c rep h{rep}: LB-eq {r['lb8c']:+.6f} ± {r['lbse8c']:.6f}", flush=True)
    for name, (cols, seeds) in V.ARMS.items():
        for fit_h in (0, 1):
            bs = V.boosters(name, fit_h)                    # P57's cache: fit on the FULL world's half fit_h
            r = V.report(D, S, {c: V.score(bs, D[c], cols) for c in E.C}, 1 - fit_h)
            res[f"{w}|orig|{name}|h{fit_h}"] = r
            print(V.line(f"{w} {name} h{fit_h}->h{1 - fit_h}", r), f"{time.time() - t0:.0f}s", flush=True)
    (OUT / f"ev_{w}.json").write_text(json.dumps(res, indent=0, default=float))


if __name__ == "__main__":
    {"build": build, "eval": evaluate}[sys.argv[1]](*sys.argv[2:])
