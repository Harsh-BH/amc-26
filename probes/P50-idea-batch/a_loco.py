"""P50 A2: label check of a house-number-suffix veto on the US/India holdout (the same generator).
Rows = #34's argmax pairs (P44 A frames), accepted = #34 (pa >= 0.71). Suffix = a letter / bis / ter attached to the
hn digits of the segment N.house_number reads (b = bis, t = ter, q = quater folded together).
    .venv/bin/python probes/P50-idea-batch/a_loco.py"""
import re, sys, json
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
sys.path[:0] = ["code/business_entity_resolution/src", "probes/P50-idea-batch"]
import normalize as N
import features as F

SUF = re.compile(r"^\d+\s*(?:-\s*)?(bis|ter|quater|[a-z])\b")
CAN = {"bis": "b", "ter": "t", "quater": "q"}


def seg(x):
    for s in x.split(","):
        if N._UNIT.match(s.strip()):
            continue
        s = N._HN_LABEL.sub("", s.strip().lstrip("#").strip())
        if re.match(r"[a-z]?[-/]?\d", s):
            return s
    return ""


def suffix(addr: pd.Series) -> np.ndarray:
    f = N.fold(addr).str.replace(N._PLACEHOLDER.pattern, " ", regex=True).str.replace(N._POBOX.pattern, " ", regex=True)
    s = f.map(seg).str.extract(SUF.pattern)[0].fillna("")
    return s.map(lambda x: CAN.get(x, x)).to_numpy(object)


def addr_of(split, ids, srcs, c):
    t = pa.concat_tables([pq.read_table(f"data/work/{split}_source{i}.parquet", columns=["entity_id", "business_address"],
                                        filters=[("country", "==", c)]) for i in srcs])
    t = t.filter(pc.is_in(t.column("entity_id"), value_set=pa.array(ids)))
    return t.to_pandas().drop_duplicates("entity_id").set_index("entity_id").business_address.reindex(ids).fillna("")


if __name__ == "__main__":
    out = {}
    for c in ("India", "US"):
        A = pd.read_parquet(f"data/work/p44ms/A_{c}.parquet", columns=["s1", "sx", "y", "pa", "hn_cls", "hn_lev", "g", "half"])
        u1, i1 = np.unique(A.s1.to_numpy(object), return_inverse=True)
        ux, ix = np.unique(A.sx.to_numpy(object), return_inverse=True)
        s1s = suffix(addr_of("train", list(u1), (1,), c))[i1]
        sxs = suffix(addr_of("train", list(ux), (2, 3), c))[ix]
        eq = (A.hn_cls.to_numpy() == 0) & A.hn_lev.notna().to_numpy()          # hn digits equal, both present
        acc = A.pa.to_numpy() >= 0.71
        y = A.y.to_numpy().astype(bool)
        r = {}
        for k, m in {"no_suffix": eq & (s1s == "") & (sxs == ""), "same_suffix": eq & (s1s != "") & (s1s == sxs),
                     "one_sided": eq & ((s1s == "") ^ (sxs == "")),
                     "differ": eq & (s1s != "") & (sxs != "") & (s1s != sxs)}.items():
            r[k] = {"pairs": int(m.sum()), "truth": round(float(y[m].mean()), 4) if m.any() else None,
                    "acc": int((m & acc).sum()), "acc_prec": round(float(y[m & acc].mean()), 4) if (m & acc).any() else None}
        out[c] = r
        print(c, json.dumps(r), flush=True)
        np.savez_compressed(f"data/work/p50/a_suf_{c}.npz", s1s=s1s.astype("U6"), sxs=sxs.astype("U6"))
        del A
    json.dump(out, open("data/work/p50/a_loco.json", "w"), indent=1)
