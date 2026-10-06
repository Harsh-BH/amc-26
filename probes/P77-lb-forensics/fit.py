"""P77: LB-delta decomposition. For each scored transition A -> B, split the changed pairs by country and per-S1 context
(fill / extra / trim / empty), price every pair with the F0.5 gain a (if true) and loss b (if false) at its S1's state,
and solve N * dLB = sum sign * (pi * (a + b) - b) for the precision pi of the change set.
    .venv/bin/python probes/P77-lb-forensics/fit.py   -> work/trans.json, work/classfit.json
Assumptions (label-free, stated in RESULT.md): kept pairs of an S1 are true; an added pair that is true completes the
S1; a filled S1 that gets no true pair was a singleton with prob 0.86 (France empty 6.5% vs train singleton law 5.6%)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
N = 1_732_544
F05 = lambda P, R: np.where(P + R > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0)
# India/US component of each transition from the holdout (LB-eq, SUBMISSIONS.md); sd = 30% of it + 2e-5
IU_PRIOR = {("d4", "sub6c"): -0.00020, ("sub6c", "sub9ab"): 0.0010, ("sub31", "sub34_veto"): 0.0002,
            ("sub37_gc70", "sub54"): 0.00072}
TRANS = [("sub5", "d2"), ("d2", "d4"), ("d4", "sub6c"), ("sub6c", "sub9ab"), ("sub9ab", "sub31"), ("sub31", "sub34_veto"),
         ("sub34_veto", "sub34v_rescue90"), ("sub34v_rescue90", "sub37_gc70"), ("sub37_gc70", "sub54"), ("sub54", "sub66_final"), ("sub66_final", "sub77")]
FRAC_PUB = 0.5   # ponytail: unknown public share; SE scales 1/sqrt(f)


def mc_table(pi0, lam, p_single=0.86, S=40000, K=16, NN=8, seed=0):
    """Per-pair gain a / loss b of adding n pairs to an S1 with k predicted pairs, simulated state: each kept pair true
    w.p. pi0, m0 ~ Poisson(lam) further true members nobody predicted; k = 0 (fill): the S1 is a singleton w.p. p_single,
    else it has 1 + Poisson(2) unfound members. a = mean F gain / n if all n are true, b = mean F loss / n if all false."""
    rng = np.random.default_rng(seed)
    A, B = np.zeros((K, NN)), np.zeros((K, NN))
    for n in range(1, NN):
        m0 = np.where(rng.random(S) < p_single, 0, 1 + rng.poisson(2, S))
        A[0, n] = F05(np.ones(S), n / (n + m0)).mean() / n
        B[0, n] = p_single / n
        for k in range(1, K):
            t, m = rng.binomial(k, pi0, S), rng.poisson(lam, S)
            T1, T0 = t + m + n, t + m
            A[k, n] = (F05((t + n) / (k + n), (t + n) / T1) - F05(t / k, t / T1)).mean() / n
            R0 = np.where(T0 > 0, t / np.maximum(T0, 1), 0)
            B[k, n] = (F05(t / k, R0) - F05(t / (k + n), R0)).mean() / n
    return A, B


PRICING = {"mc": mc_table(0.95, 0.35)}


def price_mc(add, kept, n_s):
    A, B = PRICING["mc"]
    k, n = np.minimum(kept, A.shape[0] - 1), np.clip(n_s, 1, A.shape[1] - 1)
    return A[k, n], B[k, n]


def price(add, kept, n_s):
    """Per-pair gain a / loss b of ADDING the pair to an S1 holding `kept` true pairs, n_s pairs added together
    (removals are priced as adding back to the B state)."""
    k, n = kept.astype(float), np.maximum(n_s, 1).astype(float)
    a = (1 - F05(np.ones_like(k), k / (k + n))) / n
    b = (1 - F05(k / (k + n), np.ones_like(k))) / n
    fill = kept == 0
    a, b = np.where(fill, 0.9 / n, a), np.where(fill, 0.86 / n, b)
    return a, b


def frame(u, names, A, B):
    iA, iB = names.index(A), names.index(B)
    m = u["mask"].to_numpy().astype(np.int64)
    inA, inB = (m >> iA) & 1, (m >> iB) & 1
    s1 = u["s1i"].to_numpy()
    nS = s1.max() + 1
    kept = np.bincount(s1, weights=inA & inB, minlength=nS).astype(int)
    nadd = np.bincount(s1, weights=inB & ~inA & 1, minlength=nS).astype(int)
    nrem = np.bincount(s1, weights=inA & ~inB & 1, minlength=nS).astype(int)
    ch = np.flatnonzero(inA != inB)
    sg = np.where(inB[ch] == 1, 1, -1)
    s = s1[ch]
    a, b = (price_mc if PRICING.get('use') == 'mc' else price)(sg > 0, kept[s], np.where(sg > 0, nadd[s], nrem[s]))
    nA, nB = kept[s] + nrem[s], kept[s] + nadd[s]
    ctx = np.select([(sg > 0) & (nA == 0), sg > 0, nB == 0], ["fill", "extra", "empty"], "trim")
    return pd.DataFrame({"row": ch, "s1i": s, "sign": sg, "a": a, "b": b, "ctx": ctx})


def solve(d, rhs):
    """pi such that sum sign*(pi*(a+b) - b) = rhs; SE from Bernoulli noise on the public subset."""
    X = (d.sign * (d.a + d.b)).sum()
    pi = (rhs + (d.sign * d.b).sum()) / X if X else np.nan
    pc = np.clip(pi, 0.02, 0.98)
    se = np.sqrt(((d.a + d.b) ** 2 * pc * (1 - pc)).sum() / FRAC_PUB) / abs(X) if X else np.nan
    return pi, se


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "mc":
        PRICING["use"] = "mc"
        if len(sys.argv) > 3:
            PRICING["mc"] = mc_table(float(sys.argv[2]), float(sys.argv[3]))
    files = json.loads((HERE / "work/files.json").read_text())
    names, lb = [n for n, _ in files], dict(files)
    u = pd.read_parquet(HERE / "work/union.parquet")
    s1 = u.key.to_numpy() >> 31
    us, u["s1i"] = np.unique(s1, return_inverse=True)
    t = pd.read_parquet(HERE.parents[1] / "data/work/test_source1.parquet", columns=["entity_id", "country"])
    cmap = pd.Series(t.country.to_numpy(), index=t.entity_id.str.slice(3).astype(np.int64).to_numpy())
    cty = cmap.reindex(us).to_numpy()
    del t, cmap
    tag = pd.read_parquet(HERE / "work/tagged.parquet") if (HERE / "work/tagged.parquet").exists() else None
    out, rows = [], []
    for A, B in TRANS:
        d = frame(u, names, A, B)
        d["c"] = cty[d.s1i]
        dlb = lb[B] - lb[A]
        iu = d[d.c != "France"]
        fr = d[d.c == "France"]
        prior = IU_PRIOR.get((A, B), 0.0)
        rec = {"A": A, "B": B, "dLB": round(dlb, 6), "iu_prior": prior}
        for c, g in d.groupby("c"):
            rec[c] = {k: int(v) for k, v in g.groupby(["ctx"]).size().items()} | {"s1": int(g.s1i.nunique())}
        # France change set: one pi for adds and one for removals is not identified; solve for the dominant action with
        # the minor one at its own prior (adds .85 / removals .35, P38/P52 audits), India/US at the holdout prior.
        nadd, nrem = (fr.sign > 0).sum(), (fr.sign < 0).sum()
        dom = "add" if nadd >= nrem else "remove"
        minor = fr[fr.sign < 0] if dom == "add" else fr[fr.sign > 0]
        pm = 0.35 if dom == "add" else 0.85
        v_minor = (minor.sign * (pm * (minor.a + minor.b) - minor.b)).sum()
        rhs = N * (dlb - prior) - v_minor
        main_ = fr[fr.sign > 0] if dom == "add" else fr[fr.sign < 0]
        pi, se = solve(main_, rhs)
        be = (main_.b.sum() / (main_.a + main_.b).sum()) if len(main_) else np.nan
        rec |= {"fr_dom": dom, "fr_n_dom": int(len(main_)), "fr_n_minor": int(len(minor)), "fr_pi": round(float(pi), 3),
                "fr_pi_se": round(float(se), 3), "breakeven_pi": round(float(be), 3),
                "value_per_pair_1e6": round(float((N * (dlb - prior) - v_minor) / max(len(main_), 1) * 1e6 / N), 3)}
        out.append(rec)
        print(json.dumps(rec), flush=True)
        if tag is not None:
            d["key"] = u.key.to_numpy()[d.row]
            d["k"] = f"{A}>{B}"
            rows.append(d.drop(columns=["row"]))
    (HERE / f"work/trans_{PRICING.get('use', 'simple')}.json").write_text(json.dumps(out, indent=1))
    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(HERE / "work/trans_pairs.parquet", index=False)


if __name__ == "__main__":
    main()
