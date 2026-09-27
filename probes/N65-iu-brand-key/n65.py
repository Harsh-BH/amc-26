"""N65: India/US synthetic-brand ("gib") recovery with P41's France rule, never applied to India/US.
A free SX whose name is a bare generator brand (one token) joins the S1 that uniquely holds its exact hn + street + city key.
    .venv/bin/python probes/N65-iu-brand-key/n65.py test      # test pool on sub76 (label-free counts)
    .venv/bin/python probes/N65-iu-brand-key/n65.py train     # train truth of the rule on the residual the #8c holdout leaves
"""
import sys, csv, json
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "probes/P41-fr-brand")]
import p41 as P
W = R / "data/work"; OUT = W / "n65"; OUT.mkdir(exist_ok=True)
RD = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)

def owners(path):
    m = pd.read_csv(path, **RD); m.columns = ["s1", "ms"]
    return {x: s for s, ms in zip(m.s1, m.ms) if ms for x in ms.split(",")}, m

def test():
    own, m = owners(W / "n63/sub76_matching.tsv")
    k_of = {s: len(ms.split(",")) if ms else 0 for s, ms in zip(m.s1, m.ms)}
    for c in ("India", "US"):
        k1, kx = P.keys("test", c)
        gk = P.gib_keys(k1, kx)
        gk["bare"] = P.bare("test", c, gk.index)
        gk["owner"] = gk.index.map(own).fillna("")
        gk["free"] = gk.owner == ""
        u = gk.key1 != ""
        print(f"== test {c}: gib SX {len(gk):,}  assigned {(~gk.free).mean():.3f}  | unique key {u.sum():,} "
              f"(owner==key {(gk.owner[u] == gk.key1[u]).mean():.3f}, free {gk.free[u].mean():.3f})", flush=True)
        for tag, mk in (("bare", gk.bare), ("non-bare", ~gk.bare)):
            z = gk[u & mk]
            print(f"   {tag:8s}: unique-key {len(z):,}  owner==key {(z.owner == z.key1).mean():.3f}  owner other {((z.owner != '') & (z.owner != z.key1)).mean():.3f}"
                  f"  FREE {z.free.sum():,}  (free & key S1 empty {sum(k_of.get(s, 0) == 0 for s in z.key1[z.free]):,})", flush=True)
        gk.to_parquet(OUT / f"gk_test_{c}.parquet")

def train():
    gt = pd.read_parquet(W / "gt_pairs.parquet"); t_of = gt.set_index("sx").s1
    for c in ("India", "US"):
        k1, kx = P.keys_train(c)
        km, kn = P.key_match(k1, kx)
        kx["key1"] = np.where(kn == 1, k1.id.to_numpy()[np.maximum(km, 0)], "")
        kx["bare"] = P.bare("train", c, kx.id).reindex(kx.id).fillna(False).to_numpy()
        kx["true_s1"] = t_of.reindex(kx.id).fillna("").to_numpy()
        d = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "pred"])
        ho = set(d.s1[d.holdout]); predsx = d[d.pred].set_index("sx").s1
        kx["pred_s1"] = predsx[~predsx.index.duplicated()].reindex(kx.id).fillna("").to_numpy()
        kx["in_dump"] = kx.id.isin(set(d.sx))
        z = kx[(kx.key1 != "") & kx.key1.isin(ho)]
        for tag, mk in (("bare", z.bare), ("non-bare", ~z.bare)):
            a = z[mk]; free = a.pred_s1 == ""
            print(f"== train {c} {tag}: unique key on holdout S1 {len(a):,} | key==truth {(a.key1 == a.true_s1).mean():.3f} | pred==key {(a.pred_s1 == a.key1).mean():.3f}"
                  f" | FREE {free.sum():,} ({free.mean():.3f}) key==truth among free {(a.key1 == a.true_s1)[free].mean():.3f}"
                  f" | free & not in dump {(free & ~a.in_dump).sum():,} truth {(a.key1 == a.true_s1)[free & ~a.in_dump].mean():.3f}", flush=True)
        kx.to_parquet(OUT / f"kx_train_{c}.parquet")

if __name__ == "__main__":
    {"test": test, "train": train}[sys.argv[1]]()
