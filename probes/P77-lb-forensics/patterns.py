"""P77 part B: label-free test-set patterns of a scored file (from the union bit), France vs India / US, by S1 segment.
    .venv/bin/python probes/P77-lb-forensics/patterns.py sub77   -> work/patterns_<name>.json"""
import gc, json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W = ROOT / "data/work"
LF = r"(?i)\b(?:sarl|sas|sasu|sa|eurl|sci|ei|ltd|llc|inc|pvt|private|limited|corp|co)\b"


def noaddr(s):
    return s.str.replace(r"(?i)<\s*null\s*>|\bnull\b|n/a", "", regex=True).str.replace(r"[\s,.;:/#\-]", "", regex=True).str.len() == 0


def main(name):
    names = [n for n, _ in json.loads((HERE / "work/files.json").read_text())]
    u = pd.read_parquet(HERE / "work/union.parquet")
    k = u.key.to_numpy()[((u["mask"].to_numpy().astype(np.int64) >> names.index(name)) & 1) == 1]
    del u
    s1n, is3, sxn = k >> 31, (k >> 30) & 1, k & (2**30 - 1)
    R = {}
    for c in ["France", "India", "US"]:
        t1 = pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "business_name", "business_address", "country"], filters=[("country", "=", c)])
        ids = t1.entity_id.str.slice(3).astype(np.int64).to_numpy()
        o = np.argsort(ids)
        ids_s = ids[o]
        pos = np.searchsorted(ids_s, s1n)
        ok = (pos < len(ids_s)) & (ids_s[np.minimum(pos, len(ids_s) - 1)] == s1n)
        idx = o[pos[ok]]
        n = np.bincount(idx, minlength=len(ids))
        n3 = np.bincount(idx, weights=is3[ok], minlength=len(ids))
        n2 = n - n3
        rf = pd.read_parquet(W / f"rf_test_s1_{c}.parquet", columns=["id", "core", "hn", "is_web"]).set_index("id").reindex(t1.entity_id)
        tw = rf.core.map(rf.core.value_counts()).fillna(1).to_numpy()
        seg = pd.DataFrame({"emp": n == 0, "n": n, "onesided": ((n2 == 0) | (n3 == 0)) & (n > 0),
                            "hn": (rf.hn.fillna("") != "").to_numpy(), "web": rf.is_web.fillna(False).astype(bool).to_numpy(),
                            "twin": tw > 1, "ntok": np.minimum(rf.core.fillna("").str.count(" ").to_numpy() + 1, 4),
                            "lf": t1.business_name.str.contains(LF, regex=True).to_numpy(), "noaddr1": noaddr(t1.business_address).to_numpy()})
        r = {"s1": len(seg), "empty": round(float(seg.emp.mean()), 4), "per_s1": round(float(seg.n.mean()), 3), "onesided": round(float(seg.onesided.mean()), 4)}
        for col in ["hn", "web", "twin", "ntok", "lf", "noaddr1"]:
            g = seg.groupby(col).agg(n=("emp", "size"), empty=("emp", "mean"), per_s1=("n", "mean"), onesided=("onesided", "mean"))
            r["by_" + col] = {str(kk): {a: round(float(b), 4) for a, b in row.items()} for kk, row in g.iterrows()}
        cset = set(s1n[ok].tolist()) if False else None
        mine = np.isin(s1n, ids_s)
        for src in (2, 3):
            sx = pd.read_parquet(W / f"test_source{src}.parquet", columns=["entity_id", "business_name", "business_address", "country"], filters=[("country", "=", c)])
            xn = sx.entity_id.str.slice(3).astype(np.int64).to_numpy()
            asg = np.isin(xn, sxn[mine & (is3 == (src == 3))])
            na = noaddr(sx.business_address).to_numpy()
            dup = (sx.business_name + "\t" + sx.business_address).duplicated(keep=False).to_numpy()
            r[f"S{src}"] = {"per_s1": round(len(sx) / len(seg), 3), "assigned": round(float(asg.mean()), 4), "noaddr": round(float(na.mean()), 4),
                            "assigned_noaddr": round(float(asg[na].mean()), 4) if na.any() else None, "assigned_addr": round(float(asg[~na].mean()), 4),
                            "dup_rows": int(dup.sum()), "dup_assigned": round(float(asg[dup].mean()), 4) if dup.any() else None}
            del sx, xn, asg, na, dup
            gc.collect()
        R[c] = r
        print(c, json.dumps({kk: v for kk, v in r.items() if not kk.startswith("by_")}), flush=True)
        del t1, rf, seg
        gc.collect()
    (HERE / f"work/patterns_{name}.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
