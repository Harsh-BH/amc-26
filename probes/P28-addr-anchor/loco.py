"""P28: the address-anchor rule in France's regime -- a matcher that never saw the country (P6's leave-one-country-out
protocol, P27's copies of P6's dense-holdout scores: data/work/p27/p6/S_holdout_<src>.parquet = M2 trained on <src>
only, scoring every fold-0 train pair of both countries, argmax over all train S1). Target = the other country; t = the
target's P6 empty-rate rule (t_profile) and its oracle t (P27 loco_base.log). Paired dF0.5 per fold-0 S1 of the target,
half 1 (run.half) with SE; rule-pair / flipped-pair precision. Aggregates only.
    .venv/bin/python probes/P28-addr-anchor/loco.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import features as F  # noqa: E402
from anchor import VARIANTS, rule_masks  # noqa: E402
from ho import half  # noqa: E402

W = ROOT / "data/work"
CI = {"India": 0, "US": 1}                                   # P4 train.COUNTRIES order; ids offset by ci * OFF
OFF1, OFF2 = 10_000_000, 100_000_000
TS = {("US", "India"): (0.97, 0.98), ("India", "US"): (0.81, 0.87)}   # (t_profile, oracle t), P27 loco_base.log


def fbeta(tp, npred, ntrue):
    p, r = tp / np.maximum(npred, 1), tp / np.maximum(ntrue, 1)
    f = np.where(tp > 0, 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12), 0.0)
    return np.where(ntrue == 0, (npred == 0).astype(float), f)


def run(src: str, tgt: str) -> dict:
    t0 = time.time()
    ci = CI[tgt]
    S = pd.read_parquet(W / f"p27/p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all",
                                                                         "sx_best_p"])
    S = S[S.country.to_numpy() == ci].reset_index(drop=True)
    id1, idx = F.ids("train", tgt)
    q1 = (S.s1.to_numpy() - ci * OFF1)
    d = pd.DataFrame({"s1": id1.take(pa.array(q1)).to_numpy(zero_copy_only=False),
                      "sx": idx.take(pa.array(S.sx.to_numpy() - ci * OFF2)).to_numpy(zero_copy_only=False),
                      "p": S.p.to_numpy(np.float64)})
    y, am, best = S.is_true.to_numpy(), S.am_all.to_numpy(), S.sx_best_p.to_numpy(np.float64)
    del S
    fo = pd.read_parquet(W / "s1_folds.parquet")
    fo = fo[(fo.fold == 0) & fo.entity_id.isin(set(id1.to_numpy(zero_copy_only=False)))].entity_id.to_numpy()
    gt = pd.read_parquet(W / "gt_pairs.parquet", columns=["s1"]).s1.value_counts()
    ntrue = gt.reindex(fo).fillna(0).to_numpy()
    pos = pd.Index(fo).get_indexer(d.s1)
    assert (pos >= 0).all()
    hv = half(fo)
    res = {}
    for t in TS[(src, tgt)]:
        m = rule_masks(d, "holdout", tgt, t, am=am, best=best)
        base = am & (d.p.to_numpy() >= t)

        def per(pred):
            tp = np.bincount(pos[pred & y], minlength=len(fo))
            return fbeta(tp, np.bincount(pos[pred], minlength=len(fo)), ntrue)
        f0 = per(base)
        print(f"{src}->{tgt} t {t}: {len(d):,} pairs, base F0.5 {f0.mean():.5f} (half1 {f0[hv == 1].mean():.5f}); "
              f"{m['info']}", flush=True)
        print("  sibling words:", sorted(m["sibw"]), flush=True)
        res[f"t{t}_base"] = round(f0.mean(), 5)
        for v in VARIANTS:
            r = m[v]                                           # eligible by construction: argmax, or SX best p < t
            # the boosted pair becomes its SX's argmax (p >= 0.99 > t); several rule pairs on one SX: the highest p
            g = pd.Series(np.where(r, d.p.to_numpy(), -1.0)).groupby(pd.factorize(d.sx)[0])
            win = r & (np.where(r, d.p.to_numpy(), -1.0) == g.transform("max").to_numpy())
            sx_r = pd.Series(r).groupby(pd.factorize(d.sx)[0]).transform("any").to_numpy()
            pred = np.where(sx_r, win, base)
            dl = per(pred) - f0
            flip = pred & ~base
            x = dl[hv == 1]
            k = {"d_h1": x.mean(), "se_h1": x.std() / np.sqrt(len(x)), "d_all": dl.mean(), "rule_pairs": int(r.sum()),
                 "rule_prec": y[r].mean(), "flip": int(flip.sum()), "flip_prec": y[flip].mean() if flip.any() else np.nan,
                 "lost": int((base & ~pred).sum())}
            for kind in ("sub1", "add1", "drop1"):
                kf = flip & (m["kind"] == kind)
                k[f"flip_{kind}"], k[f"flip_prec_{kind}"] = int(kf.sum()), (y[kf].mean() if kf.any() else np.nan)
            print(f"  {v:3s} dF half1 {k['d_h1']:+.5f} ± {k['se_h1']:.5f} (all {k['d_all']:+.5f}) | rule pairs "
                  f"{k['rule_pairs']:,} prec {k['rule_prec']:.3f} | flipped {k['flip']:,} prec {k['flip_prec']:.3f} "
                  f"(sub1 {k['flip_sub1']} {k['flip_prec_sub1']:.3f}, add1 {k['flip_add1']} {k['flip_prec_add1']:.3f}, "
                  f"drop1 {k['flip_drop1']} {k['flip_prec_drop1']:.3f}) | lost {k['lost']}", flush=True)
            res |= {f"t{t}_{v}_{a}": (round(float(b), 5) if isinstance(b, float) else b) for a, b in k.items()}
    res["runtime_s"] = round(time.time() - t0)
    return res


if __name__ == "__main__":
    import resource
    from ledger import log
    pairs = [tuple(a.split(">")) for a in sys.argv[1:]] or [("India", "US"), ("US", "India")]
    for src, tgt in pairs:
        r = run(src, tgt)
        gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        tp = TS[(src, tgt)][0]
        best = max(VARIANTS, key=lambda v: r[f"t{tp}_{v}_d_all"])
        log("exp", probe="P28-addr-anchor", subproblem="decide", name=f"addr-anchor-loco-{src}>{tgt}",
            world="holdout-dense-fold0 LOCO (P6 S_holdout, M2 no CE)", metrics=r,
            config={"t_profile_oracle": TS[(src, tgt)], "boost": 0.99, "variants": "a/b/bx/c as ho.py"},
            hypothesis="an unseen-country matcher under-accepts one-word-swap same-address pairs; the rule recovers them",
            verdict="keep" if r[f"t{tp}_{best}_d_h1"] > 2 * r[f"t{tp}_{best}_se_h1"] else
                    ("drop" if r[f"t{tp}_{best}_d_h1"] < 0 else "inconclusive"),
            runtime_s=r["runtime_s"], peak_gb=round(gb, 2))
        print(f"{src}->{tgt}: peak {gb:.2f} GB", flush=True)
