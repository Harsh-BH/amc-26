"""Aggregates only. House-number label fix (H.No./D.No./Plt./Sy.No. with dots, '(12)' prefix): which argmax pairs
change hn relation, their true rate (India/US holdout, P23 dump) and current acceptance; France test counts by p band."""
import re, sys, numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.dataset as ds, pyarrow.parquet as pq
R = "/home/harsh1/github-repos/amc-26/"; W = R + "data/work/"
sys.path[:0] = [R + "code/business_entity_resolution/src"]
import normalize as N
LAB = re.compile(r"^(?:#|no\b|door\.?\s*no\b|[hd]\.?\s*no\b|hn\b|house\s*no\b|plot\s*no\b|plot\b|pl?t\b|s\.?\s*no\b|sy\.?\s*no\b"
                 r"|flat\s*no\b|flat\b|shop\s*no\b|shop\b|unit\s*no\b|n)?[\s.:#-]*")
def hn_new(addr):
    f = N.fold(addr).str.replace(N._PLACEHOLDER.pattern, " ", regex=True).str.replace(N._POBOX.pattern, " ", regex=True)
    def one(x):
        for seg in x.split(","):
            if N._UNIT.match(seg.strip()): continue
            seg = LAB.sub("", seg.strip().lstrip("#(").strip())
            m = re.match(r"[a-z]?[-/]?(\d+)", seg)
            if m: return m.group(1).lstrip("0") or "0"
        return ""
    return f.map(one).astype("str")
t = pd.Series(["H.No. 12-3, X", "D.No.5-6, X", "(12) rue X", "Plt. 44, X", "Sy. No. 7, X", "1822 Perkins Road, Unit # 533", "Shop No.5-6, K"], dtype="str")
assert list(hn_new(t)) == ["12", "5", "12", "44", "7", "1822", "5"], list(hn_new(t))
SIB = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
def rel(a, b):
    out = []
    for x, y in zip(a, b):
        if not x or not y: out.append("missing")
        elif x == y: out.append("eq")
        elif x.isdigit() and y.isdigit() and abs(int(y) - int(x)) in SIB: out.append("sib")
        else: out.append("other")
    return np.array(out)
def addrs(files, ids):
    ids = pa.array(np.unique(ids))
    parts = [ds.dataset(W + f).to_table(filter=pc.field("entity_id").isin(ids), columns=["entity_id", "business_address"]).to_pandas() for f in files]
    return pd.concat(parts).drop_duplicates("entity_id").set_index("entity_id").business_address
for c in sys.argv[1:]:
    if c == "France":
        d = pd.read_parquet(W + "p19/prob_test_France_idf144_bge.parquet", columns=["s1", "sx", "p"])
        d = d.sort_values("p", ascending=False).drop_duplicates("sx"); split = "test"; y = None
    else:
        d = pd.read_parquet(W + f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p", "am"])
        d = d[d.holdout & d.am]; split = "train"
    a1 = addrs([f"{split}_source1.parquet"], d.s1.to_numpy()); ax = addrs([f"{split}_source2.parquet", f"{split}_source3.parquet"], d.sx.to_numpy())
    A1, AX = d.s1.map(a1).fillna("").astype("str"), d.sx.map(ax).fillna("").astype("str")
    del a1, ax
    o1, ox = N.house_number(A1).to_numpy(), N.house_number(AX).to_numpy()
    n1, nx = hn_new(A1).to_numpy(), hn_new(AX).to_numpy()
    r0, r1 = rel(o1, ox), rel(n1, nx)
    ch = r0 != r1
    print(f"== {c}: argmax pairs {len(d):,}; hn relation changed {ch.sum():,} ({ch.mean():.4f}); S1 hn newly parsed {(o1!=n1).sum():,}, SX {(ox!=nx).sum():,}")
    p = d.p.to_numpy(); t = 0.985 if c == "France" else 0.71
    df = pd.DataFrame({"old": r0[ch], "new": r1[ch], "acc": p[ch] >= t, "near": (p[ch] >= 0.5) & (p[ch] < t)})
    if c != "France": df["y"] = d.y.to_numpy()[ch]
    print(df.groupby(["old", "new"]).agg(**{k: (k, "mean") for k in df.columns if k not in ("old", "new")}, n=("acc", "size")).to_string())
    if c != "France":
        yy = d.y.to_numpy().astype(bool); acc = p >= t
        print(f"   errors in changed rows: FN {(ch & yy & ~acc).sum():,}  FP {(ch & ~yy & acc).sum():,}; all FN {(yy & ~acc).sum():,} FP {(~yy & acc).sum():,}")
    del d, A1, AX
