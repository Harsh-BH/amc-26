"""P39 diagnostics on the LOCO targets (aggregates only):
  oracle   headroom of accept / reject over SX-argmax pairs (perfect labels), overall and for one-sided S1; where the
           missing-source true members of one-sided S1 sit (argmax below t / in frame not argmax / not a candidate)
  adds     what the US-labelled 'full' model adds on India at s >= 0.98 (hn / name relation mix, precision)
  lens     label-free hn / name relation mix below t: India target vs US target vs France
    .venv/bin/python probes/P39-fr-group-model/diag.py oracle | adds | lens
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "code/business_entity_resolution/src")]
from fit import W, load, per_s1  # noqa: E402
from frame import CI, HNREL, KINDS, OFF1, OFF2  # noqa: E402

WK = ROOT / "data/work"


def oracle():
    import features as F
    for src, tgt in (("US", "India"), ("India", "US")):
        d, nt, _ = load(src, tgt)
        s1, is2, y, acc = d.s1.to_numpy(), d.is2.to_numpy(), d.y.to_numpy(), d.acc.to_numpy()
        n = len(nt)
        base = per_s1(d, acc, nt)
        n2, n3 = np.bincount(s1[acc & is2], minlength=n), np.bincount(s1[acc & ~is2], minlength=n)
        one2 = (n2 > 0) & (n3 == 0)                  # S3 missing
        one3 = (n3 > 0) & (n2 == 0)                  # S2 missing
        orc_add = per_s1(d, acc | y, nt) - base
        orc_rm = per_s1(d, acc & y, nt) - base
        orc = per_s1(d, y, nt) - base
        miss_src = np.where(is2, one3[s1], one2[s1])  # row's source is the missing one of a one-sided S1
        orc_one = per_s1(d, acc | (y & miss_src), nt) - base
        print(f"== {src}>{tgt}: base {base.mean():.5f}; oracle add-only {orc_add.mean():+.5f}, remove-only {orc_rm.mean():+.5f}, "
              f"both {orc.mean():+.5f}; complete one-sided S1 only {orc_one.mean():+.5f}")
        # where are the missing-source true members of one-sided S1 whose truth has that source?
        S = pd.read_parquet(WK / f"p27/p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "is_true", "am_all", "p"])
        S = S[(S.country.to_numpy() == CI[tgt]) & S.is_true.to_numpy()]
        id1, idx = F.ids("train", tgt)
        e1 = id1.take(pa.array(S.s1.to_numpy() - CI[tgt] * OFF1)).to_numpy(zero_copy_only=False)
        ex = idx.take(pa.array(S.sx.to_numpy() - CI[tgt] * OFF2)).to_numpy(zero_copy_only=False)
        ids = pd.read_parquet(W / f"s1_{src}_{tgt}.parquet").id.to_numpy()
        c = pd.Index(ids).get_indexer(e1)
        inframe = pd.DataFrame({"s1": c, "is2": pd.Series(ex).str.startswith("S2-").to_numpy(), "am": S.am_all.to_numpy()})
        gt = pd.read_parquet(WK / "gt_pairs.parquet")
        gt = gt[gt.s1.isin(set(ids))]
        g = pd.DataFrame({"s1": pd.Index(ids).get_indexer(gt.s1), "is2": gt.sx.str.startswith("S2-").to_numpy()})
        for name, one, want2 in (("S3-missing", one2, False), ("S2-missing", one3, True)):
            has_true = np.bincount(g.s1[g.is2 == want2], minlength=n) > 0
            k = one & has_true
            am_below = np.bincount(s1[y & ~acc & (is2 == want2)], minlength=n) > 0
            infr = np.bincount(inframe.s1[inframe.is2 == want2], minlength=n) > 0
            infr_nam = np.bincount(inframe.s1[(inframe.is2 == want2) & ~inframe.am], minlength=n) > 0
            print(f"  {name}: {int(one.sum()):,} S1 ({one.mean():.4f}); truth has that source {int(k.sum()):,} "
                  f"({k.sum() / max(one.sum(), 1):.3f}); of those: a true member is argmax below t {am_below[k].mean():.3f}, "
                  f"a true member is a candidate {infr[k].mean():.3f}, a candidate but not argmax {(infr_nam & ~am_below)[k].mean():.3f}, "
                  f"no true member among candidates {(~infr)[k].mean():.3f}")
        del S, e1, ex, gt


def adds():
    d, nt, _ = load("US", "India")
    s = np.load(W / "s_full_US_India.npy")
    a = (s >= 0.98) & ~d.acc.to_numpy()
    y = d.y.to_numpy()
    x = d[a].assign(y=y[a])
    print(f"added {a.sum():,} prec {y[a].mean():.3f}; lt quartiles {np.percentile(x["lt"], [25, 50, 75]).round(2)}")
    for col, lab in (("hn_rel", HNREL), ("kind", KINDS)):
        t = x.groupby(col).y.agg(["size", "mean"])
        t.index = [lab[i] if i >= 0 else "?" for i in t.index]
        print(col, t.round(3).to_dict("index"))
    t = x.groupby([x.addr_eq, x.grp_hn]).y.agg(["size", "mean"])
    print("addr_eq x grp_hn", t.round(3).to_dict("index"))
    # the same slices on the US-labelled training frame (India first stage on US): precision below t
    dt, _, _ = load("India", "US")
    b = ~dt.acc.to_numpy() & (dt.p.to_numpy() >= 0.02)
    for col, lab in (("hn_rel", HNREL),):
        t = dt[b].groupby(col).y.agg(["size", "mean"])
        t.index = [lab[i] for i in t.index]
        print("train frame below-t", col, t.round(3).to_dict("index"))
    tt = d[~d.acc.to_numpy() & (d.p.to_numpy() >= 0.02)].groupby("hn_rel").y.agg(["size", "mean"])
    tt.index = [HNREL[i] for i in tt.index]
    print("India target below-t hn_rel", tt.round(3).to_dict("index"))




def lens():
    """Label-free: hn / name relation mix of live argmax pairs below t (the slices that broke US>India), per world."""
    rows = {}
    for name, f in (("India target (US>India)", "loco_US_India"), ("US target (India>US)", "loco_India_US"), ("France", "france")):
        d = pd.read_parquet(W / f"{f}.parquet", columns=["p", "acc", "hn_rel", "kind"])
        b = d[~d.acc.to_numpy() & (d.p.to_numpy() >= 0.02)]
        a = d[d.acc.to_numpy()]
        r = {f"below hn:{HNREL[k]}": round(float(v), 3) for k, v in b.hn_rel.value_counts(normalize=True).sort_index().items()}
        r |= {f"below name:{KINDS[k]}": round(float(v), 3) for k, v in b.kind.value_counts(normalize=True).sort_index().items() if KINDS[k] in ("eq", "add1", "sub1", "drop1")}
        r["below:miss&add1"] = round(float(((b.hn_rel == 0) & (b.kind == 1)).mean()), 3)
        r["acc:hn_miss"] = round(float((a.hn_rel == 0).mean()), 3)
        r["acc:hn_eq"] = round(float((a.hn_rel == 1).mean()), 3)
        r["below_per_acc"] = round(len(b) / len(a), 3)
        rows[name] = r
    t = pd.DataFrame(rows)
    print(t.to_string())


if __name__ == "__main__":
    {"oracle": oracle, "adds": adds, "lens": lens}[sys.argv[1]]()
