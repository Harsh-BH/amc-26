"""Generator-trace features (from the public repo's stage C) for our India/US combiner, row-aligned with a frame's (q1, qx, g).
  t_site_same    S1 and SX share the exact normalised hn|street|city key
  t_site_n1      S1 pool rows at the SX's key (0 = no key)          t_s1site_n  S1 pool rows at the S1's own key
  t_site_uniq    t_site_same and the S1 is alone at that key
  t_lasttok      cores differ only in the last token, same site
  t_src_agree    other frame SX of the same S1 and source with the identical folded address
  t_gib          SX name is a P36 synthetic brand
    traces.py SPLIT C  -> data/work/p70/tr_<split>_<c>.parquet (holdout frame for train, T frame for test)"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P70-fr-roles"), str(ROOT / "probes/P44-meta-stack"), str(ROOT / "code/business_entity_resolution/src")]
import p70  # noqa: E402
import features as FT  # noqa: E402
import normalize as N  # noqa: E402
import gib as G  # noqa: E402
W = ROOT / "data/work"
TR = ["t_site_same", "t_site_n1", "t_s1site_n", "t_site_uniq", "t_lasttok", "t_src_agree", "t_gib"]


def frame_rows(split, c):
    if split == "train":
        A = pd.read_parquet(W / f"p44ms/A_{c}.parquet", columns=["q1", "qx", "g"])
    else:
        A = pd.read_parquet(W / f"p50/T_{c}.parquet", columns=["q1", "qx", "g"])
    return A


def build(split, c):
    t0 = time.time()
    A = frame_rows(split, c)
    id1, idx = FT.ids(split, c)
    id1, idx = id1.to_numpy(zero_copy_only=False), idx.to_numpy(zero_copy_only=False)
    s1, sx = id1[A.q1.to_numpy()], idx[A.qx.to_numpy()]
    # S1 pool keys (all S1 of the country)
    t = pq.read_table(W / f"{split}_source1.parquet", columns=["entity_id", "business_address", "country"], filters=[("country", "=", c)]).to_pandas()
    rf1 = pd.read_parquet(W / f"rf_{split}_s1_{c}.parquet", columns=["id", "core", "city"]).set_index("id")
    k1 = p70._street(t.business_address.fillna(""), t.country)
    city1 = rf1.city.reindex(t.entity_id).fillna("").to_numpy()
    key1 = np.where((k1.to_numpy() != "") & (city1 != ""), k1.to_numpy() + "|" + city1, "")
    kser = pd.Series(key1, index=t.entity_id.to_numpy())
    kcnt = pd.Series(key1[key1 != ""]).value_counts()
    del t, k1
    # SX of the frame
    ux = pd.unique(sx)
    rfx = pd.read_parquet(W / f"rf_{split}_sx_{c}.parquet", columns=["id", "core", "city"]).set_index("id")
    M = set(json.loads((W / f"p36/morph_{'test' if split == 'test' else 'train'}.json").read_text()))
    kl, al, gl = [], [], []
    for i in range(0, len(ux), 700_000):           # chunked: test frames hold ~3.7M SX
        u = ux[i:i + 700_000]
        rx = p70._raw(split, u, [2, 3], ["business_name", "business_address", "country"])
        ax = rx.business_address.reindex(u).fillna("")
        kx = p70._street(ax, pd.Series([c] * len(u), index=ax.index)).to_numpy()
        cityx = rfx.city.reindex(u).fillna("").to_numpy()
        kl.append(np.where((kx != "") & (cityx != ""), kx + "|" + cityx, ""))
        al.append(N.fold(ax).str.replace(r"[^a-z0-9]", "", regex=True).to_numpy())
        nm = rx.business_name.reindex(u).fillna("").astype(str).to_numpy()
        un, inv = np.unique(nm, return_inverse=True)
        gl.append(G.is_gib(un, M)[inv])
        del rx, ax, kx, nm
        print(f"  sx chunk {i + len(u):,}/{len(ux):,} peak {p70.peak():.2f} GB", flush=True)
    keyx = pd.Series(np.concatenate(kl), index=ux)
    addrx = pd.Series(np.concatenate(al), index=ux)
    gibx = pd.Series(np.concatenate(gl), index=ux)
    del kl, al, gl
    ks1, ksx = kser.reindex(s1).fillna("").to_numpy(), keyx.reindex(sx).fillna("").to_numpy()
    out = pd.DataFrame(index=A.index)
    out["t_site_same"] = ((ks1 == ksx) & (ksx != "")).astype(np.float32)
    out["t_site_n1"] = pd.Series(ksx).map(kcnt).fillna(0).to_numpy(np.float32)
    out["t_s1site_n"] = pd.Series(ks1).map(kcnt).fillna(0).to_numpy(np.float32)
    out["t_site_uniq"] = ((out.t_site_same > 0) & (out.t_site_n1 == 1)).astype(np.float32)
    c1, cx = rf1.core.reindex(s1).fillna("").to_numpy(), rfx.core.reindex(sx).fillna("").to_numpy()
    def last_only(a, b):
        ta, tb = a.split(), b.split()
        return len(ta) == len(tb) >= 2 and ta[:-1] == tb[:-1] and ta[-1] != tb[-1]
    lt = np.fromiter((last_only(a, b) for a, b in zip(c1, cx)), bool, len(c1))
    out["t_lasttok"] = (lt & (out.t_site_same.to_numpy() > 0)).astype(np.float32)
    ad = addrx.reindex(sx).fillna("").to_numpy()
    src = np.array([x[1] for x in sx])
    grp = pd.DataFrame({"g": A.g.to_numpy(), "src": src, "ad": ad})
    n = grp.groupby(["g", "src", "ad"]).ad.transform("size").to_numpy()
    out["t_src_agree"] = np.where(ad != "", n - 1, 0).astype(np.float32)
    out["t_gib"] = gibx.reindex(sx).fillna(False).to_numpy(np.float32)
    out.to_parquet(W / f"p70/tr_{split}_{c}.parquet", index=False)
    print(f"{split} {c}: {len(out):,} rows, site_same {out.t_site_same.mean():.3f}, site_uniq {out.t_site_uniq.mean():.3f}, lasttok {out.t_lasttok.mean():.4f}, "
          f"src_agree>0 {(out.t_src_agree > 0).mean():.3f}, gib {out.t_gib.mean():.3f} | {time.time() - t0:.0f}s peak {p70.peak():.2f} GB", flush=True)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
