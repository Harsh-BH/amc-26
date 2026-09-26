"""P49 tests 3 + 5 (raw-string channel, within-group distinctness) on the ambiguous twin rows. Aggregates only.
3: twins identical after P40's key but raw names differ -> is the raw-closest twin (rapidfuzz Levenshtein on the raw,
   case-sensitive string; exact raw equality as its own slice) the owner more often than 1/nset?
5: do true groups contain two identical raw (name, address) records? Train truth, every train S1 of the country, vs the
   same count across S1 of the same raw name (twins). And in-frame: an ambiguous SX identical to another address-less SX
   owned by twin A -> how often is A its owner?
    probes/P49-twin-prior/py.sh probes/P49-twin-prior/test35.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein as Lv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame import W, sx_text, noaddr  # noqa: E402


def se(p, n):
    return np.sqrt(p * (1 - p) / max(n, 1))


for c in ("India", "US"):
    tw = pd.read_parquet(W / f"p49/tw_{c}.parquet")
    tw["own"] = tw.groupby("sx").y_full.transform("any")
    tw["anyho"] = tw.groupby("sx").holdout.transform("any")
    q = tw[tw.own & tw.anyho].copy()
    q["nraw"] = q.groupby("sx").s1_name.transform("nunique")
    q["d"] = [Lv.distance(a, b) for a, b in zip(q.sx_name, q.s1_name)]
    q["eq"] = q.sx_name.to_numpy() == q.s1_name.to_numpy()
    q["dl"] = [Lv.distance(a.lower(), b.lower()) for a, b in zip(q.sx_name, q.s1_name)]
    print(f"\n{c}: ambiguous SX with owner in set: {q.sx.nunique():,}; twins' raw names all identical in "
          f"{(q.groupby('sx').nraw.first() == 1).mean():.3f}", flush=True)
    diff = q[q.nraw > 1]
    for col, nm in (("d", "raw Levenshtein"), ("dl", "lower-cased Levenshtein")):
        mn = diff.groupby("sx")[col].transform("min")
        diff = diff.assign(best=(diff[col] == mn))
        nb = diff.groupby("sx").best.transform("sum")
        uniq = diff[diff.best & (nb == 1)]
        tot = diff.sx.nunique()
        base = (1 / diff[diff.best & (nb == 1)].nset).mean()
        r = uniq.y_full.mean()
        # share of the owner among the tied-best set when ties
        print(f"  raw names differ: {tot:,} SX. {nm}: unique closest twin in {len(uniq):,}; it is the owner "
              f"{r:.3f} ± {se(r, len(uniq)):.3f} (uniform {base:.3f}); nset==2 slice {uniq[uniq.nset == 2].y_full.mean():.3f} "
              f"± {se(uniq[uniq.nset == 2].y_full.mean(), (uniq.nset == 2).sum()):.3f} (n {(uniq.nset == 2).sum():,})", flush=True)
    ex = diff[diff.eq]
    nex = ex.groupby("sx").size()
    one = ex[ex.sx.isin(nex[nex == 1].index)]
    print(f"  exact raw equality with exactly one twin (raw names differ): {len(one):,} SX, owner {one.y_full.mean():.3f} ± "
          f"{se(one.y_full.mean(), len(one)):.3f} (uniform {(1 / one.nset).mean():.3f})", flush=True)

    # test 5, in frame: another address-less SX with the identical raw name, owned by a twin of the same set
    own = q[q.y_full][["sx", "s1", "sx_name"]]
    both = own.merge(own, on="sx_name", suffixes=("", "_o"))
    both = both[both.sx != both.sx_o]
    sets = q.groupby("sx").s1.apply(frozenset)
    both = both[[a in sets[x] for a, x in zip(both.s1_o, both.sx)]]
    b1 = both.drop_duplicates("sx")
    same = (b1.s1 == b1.s1_o).mean()
    nset = q.groupby("sx").nset.first().reindex(b1.sx).to_numpy()
    print(f"  test 5 in frame: ambiguous SX with an identical-raw address-less sibling SX owned by a twin in its set: "
          f"{len(b1):,}; same owner {same:.3f} ± {se(same, len(b1)):.3f} (uniform {(1 / nset).mean():.3f})", flush=True)

# test 5 on the full train truth: identical raw (name, address) inside one S1 group vs across same-raw-name S1
gt = pd.read_parquet(W / "gt_pairs.parquet")
s1c = pd.read_parquet(W / "s1_folds.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
for c in ("India", "US"):
    g = gt[s1c.reindex(gt.s1).to_numpy() == c].sample(frac=0.25, random_state=0) if False else gt[s1c.reindex(gt.s1).to_numpy() == c]
    s1s = pd.Series(g.s1.unique()).sample(frac=0.2, random_state=0)
    g = g[g.s1.isin(set(s1s))]
    tx = sx_text(set(g.sx)).set_index("entity_id")
    g = g.assign(n=tx.business_name.reindex(g.sx).to_numpy(), a=tx.business_address.reindex(g.sx).to_numpy())
    g["na"] = noaddr(g.a)
    g["src"] = g.sx.str[:2]
    # pairs of members inside a group with identical raw name + raw address
    k = g.groupby(["s1", "n", "a"]).size()
    grp_pairs = g.groupby("s1").size()
    within = (k * (k - 1) / 2).sum() / (grp_pairs * (grp_pairs - 1) / 2).sum()
    kn = g[g.na].groupby(["s1", "n"]).size()
    gna = g[g.na].groupby("s1").size()
    within_na = (kn * (kn - 1) / 2).sum() / max((gna * (gna - 1) / 2).sum(), 1)
    # across S1 with the same raw S1 name: address-less members, identical raw name
    s1n = pd.read_parquet(W / "train_source1.parquet", columns=["entity_id", "business_name"]).set_index("entity_id").business_name
    g["s1n"] = s1n.reindex(g.s1).to_numpy()
    del s1n
    h = g[g.na]
    tot_same = h.groupby(["s1n", "n"]).size()
    all_pairs_name = h.groupby("s1n").size()
    # cross-S1 pairs = all pairs sharing s1n minus within-S1 pairs
    wp = h.groupby(["s1n", "s1"]).size()
    cross_pairs = (all_pairs_name * (all_pairs_name - 1) / 2).sum() - (wp * (wp - 1) / 2).sum()
    wi = h.groupby(["s1n", "s1", "n"]).size()
    cross_ident = (tot_same * (tot_same - 1) / 2).sum() - (wi * (wi - 1) / 2).sum()
    print(f"\n{c} train truth (20% of S1): P(two members of one group identical raw name+addr) {within:.4f}; "
          f"address-less member pairs of one group with identical raw name {within_na:.4f} "
          f"({int((kn * (kn - 1) / 2).sum()):,} of {int((gna * (gna - 1) / 2).sum()):,}); across same-raw-name S1: "
          f"{cross_ident / max(cross_pairs, 1):.4f} ({int(cross_ident):,} of {int(cross_pairs):,})", flush=True)
    del g, h, tx
