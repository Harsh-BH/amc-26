"""P35: twin rules in France's regime -- P6's leave-one-country-out frames (data/work/p27/p6/S_holdout_<src>.parquet:
M2 trained on <src> only, scoring every fold-0 train pair of both countries; am_all = argmax over all train S1).
Target = the other country, t = its P6 empty-rate rule (t_profile) as in P28. Paired dF0.5 per fold-0 S1 of the target,
half 1 with SE, plus rule-pair precision. Aggregates only.

    .venv/bin/python probes/P35-twin-propagate/loco.py [India>US US>India]      # ~3 min, ~1.5 GB per direction
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P35-twin-propagate"), str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import features as F  # noqa: E402
from ev import fb, half  # noqa: E402
from frame import noaddr, read_ids  # noqa: E402
from twins import twin_frame  # noqa: E402

W = ROOT / "data/work"
CI = {"India": 0, "US": 1}
OFF1, OFF2 = 10_000_000, 100_000_000
TS = {("US", "India"): 0.97, ("India", "US"): 0.81}         # P6 t_profile (P27 loco_base.log, as P28)


def frame(src, tgt):
    ci = CI[tgt]
    S = pd.read_parquet(W / f"p27/p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all"])
    S = S[S.country.to_numpy() == ci]
    id1, idx = F.ids("train", tgt)
    d = pd.DataFrame({"s1": id1.take(pa.array(S.s1.to_numpy() - ci * OFF1)).to_numpy(zero_copy_only=False),
                      "sx": idx.take(pa.array(S.sx.to_numpy() - ci * OFF2)).to_numpy(zero_copy_only=False),
                      "p": S.p.to_numpy(np.float64), "y": S.is_true.to_numpy(), "am": S.am_all.to_numpy()})
    del S, id1, idx
    ids = set(pd.unique(d.sx))
    raw = pd.concat([read_ids(W / f"train_source{i}.parquet", ids, ["entity_id", "business_address"], "entity_id")
                     for i in (2, 3)], ignore_index=True)
    d["noaddr"] = pd.Series(noaddr(raw.business_address), index=raw.entity_id.to_numpy()).reindex(d.sx).fillna(False).to_numpy(bool)
    del raw
    rf = read_ids(W / f"rf_train_sx_{tgt}.parquet", ids, ["id", "core"], "id").set_index("id").core
    d["core"] = rf.reindex(d.sx).fillna("").to_numpy()
    d["src"] = np.where(d.sx.str.startswith("S2-"), 2, 3).astype(np.int8)
    fo = pd.read_parquet(W / "s1_folds.parquet")
    fo = fo[(fo.fold == 0) & (fo.country == tgt)].entity_id.to_numpy()
    gt = pd.read_parquet(W / "gt_pairs.parquet", columns=["s1"]).s1.value_counts()
    pos = pd.Index(fo).get_indexer(d.s1)
    assert (pos >= 0).all()
    return d, pos, gt.reindex(fo).fillna(0).to_numpy(), half(fo)


def run(src, tgt):
    t0 = time.time()
    d, pos, ntrue, hv = frame(src, tgt)
    t = TS[(src, tgt)]
    y, am, p = d.y.to_numpy(), d.am.to_numpy(), d.p.to_numpy()
    base = am & (p >= t)

    def per(pred):
        return fb(np.bincount(pos[pred & y], minlength=len(ntrue)), np.bincount(pos[pred], minlength=len(ntrue)), ntrue)
    f0 = per(base)
    tw = twin_frame(d, am, t, 0.85)                        # twin = addressed SX accepted at X (argmax & p >= t)
    k = d[["s1", "sx"]].reset_index().merge(tw, on=["s1", "sx"], how="left").set_index("index").sort_index()
    sup = k.tw_p.notna().to_numpy()
    n_sup = pd.Series(sup).groupby(pd.factorize(d.sx)[0]).transform("sum").to_numpy()
    sim1 = k.tw_sim.fillna(0).to_numpy()
    print(f"{src}->{tgt} t {t}: {len(d):,} pairs, address-less SX share {d.drop_duplicates('sx').noaddr.mean():.3f}, "
          f"base F0.5 {f0.mean():.5f} (half1 {f0[hv == 1].mean():.5f}), supported pairs {int(sup.sum()):,}", flush=True)
    res = {"base": round(float(f0.mean()), 5)}
    cand = am & ~base & sup
    print(f"  rule-A pool (argmax, p < t, twin): {int(cand.sum()):,} pairs, precision {y[cand].mean():.3f}", flush=True)
    for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, t)):
        m = cand & (p >= lo) & (p < hi)
        if m.any():
            print(f"    p [{lo}, {hi}): {int(m.sum()):,} prec {y[m].mean():.3f}  (n_sup 1: {int((m & (n_sup == 1)).sum()):,} "
                  f"prec {y[m & (n_sup == 1)].mean() if (m & (n_sup == 1)).any() else float('nan'):.3f})", flush=True)
    for name, m in (("A_all", cand), ("A_p0.3", cand & (p >= 0.3)), ("A_p0.5", cand & (p >= 0.5)),
                    ("A_p0.5_n1", cand & (p >= 0.5) & (n_sup == 1)), ("A_n1_sim1", cand & (n_sup == 1) & (sim1 >= 0.999)),
                    ("A_p0.3_n1_sim1", cand & (p >= 0.3) & (n_sup == 1) & (sim1 >= 0.999))):
        dl = per(base | m) - f0
        x = dl[hv == 1]
        res[name] = dict(d_h1=round(float(x.mean()), 6), se=round(float(x.std() / np.sqrt(len(x))), 6),
                         d_all=round(float(dl.mean()), 6), pairs=int(m.sum()), prec=round(float(y[m].mean()), 3) if m.any() else None)
        print(f"  {name}: {res[name]}", flush=True)
    # conflict guard: accepted address-less pair without twin while another candidate S1 of the SX has one
    g = base & d.noaddr.to_numpy() & ~sup & (n_sup > 0)
    dl = per(base & ~g) - f0
    res["C_guard"] = dict(d_h1=round(float(dl[hv == 1].mean()), 6), d_all=round(float(dl.mean()), 6), pairs=int(g.sum()),
                          prec=round(float(y[g].mean()), 3) if g.any() else None)
    print(f"  C_guard: {res['C_guard']}  ({time.time() - t0:.0f}s)", flush=True)
    return res


if __name__ == "__main__":
    import json
    out = {}
    for a in sys.argv[1:] or ["India>US", "US>India"]:
        s, tg = a.split(">")
        out[a] = run(s, tg)
    print(json.dumps(out))
