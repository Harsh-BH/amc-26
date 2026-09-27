"""N62b: label-free decoy-vs-noise test for a word w. Among SX records whose core contains w, share whose (core token-set,
hn, city) key is shared by >= 1 OTHER SX record that also contains w (decoy = own entity, copied with w; noise = per copy),
vs the share with >= 1 other SX at the w-less key (the S1-like copy). Calibrated on train words with known truth (N61)."""
import sys, numpy as np, pandas as pd, unicodedata
W = "data/work"
fold = lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
def prof(split, c, words):
    x = pd.read_parquet(f"{W}/rf_{split}_sx_{c}.parquet", columns=["id", "core", "hn", "city"])
    x = x[(x.hn != "") & (x.core != "")]
    x["tok"] = x.core.map(lambda s: tuple(sorted(set(fold(s).split()))))
    key = x.tok.astype(str) + "|" + x.hn + "|" + x.city
    cnt = key.value_counts()
    out = {}
    for w in words:
        m = x.tok.map(lambda t: w in t and len(t) >= 2).to_numpy()
        if m.sum() < 50: out[w] = (int(m.sum()),); continue
        s = x[m]
        same = cnt.reindex(key[m]).to_numpy() - 1                                  # other SX with the identical w-core key
        k2 = s.tok.map(lambda t: str(tuple(v for v in t if v != w))) + "|" + s.hn + "|" + s.city
        base = cnt.reindex(k2).fillna(0).to_numpy()                                # SX at the same key without w
        out[w] = (int(m.sum()), round(float((same >= 1).mean()), 3), round(float((base >= 1).mean()), 3),
                  round(float(((same >= 1) & (base == 0)).mean()), 3))
        print(f"  {c} {w:<16} n {out[w][0]:>7,}  w-twin {out[w][1]:.3f}  base-copy {out[w][2]:.3f}  w-twin&no-base {out[w][3]:.3f}", flush=True)
    return out
print("train US (decoy: group, holdings | noise: services, center, dr, company)")
prof("train", "US", ["group", "holdings", "enterprises", "services", "center", "dr", "company", "the"])
print("train India (decoy: group, holdings, overseas, infratech, enterprises | noise: sri, services, dr, company)")
prof("train", "India", ["group", "holdings", "overseas", "infratech", "enterprises", "public", "sri", "services", "dr", "company", "shri"])
print("test France (N61 decoy candidates vs other France extra words)")
prof("test", "France", ["groupe", "developpement", "participations", "holding", "distribution", "international", "associes", "fils", "compagnie", "services", "et"])
