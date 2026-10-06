"""P77: tag every pair that changes between two non-diagnostic LB files with label-free classes.
    .venv/bin/python probes/P77-lb-forensics/tag.py  -> work/tagged.parquet
Classes: country, source, hn relation (0 miss 1 eq 2 sib3 3 +1/2 4 -1/2 5 other), P20 name relation (kind), P64 legal-form
class, address-less SX, synthetic-brand SX, word role s (sibling share of the swapped word; N54 lens s > .25 decoy, <= .05
noise), France g / bge r / Qwen r where the France frame has the pair."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P70-fr-roles"), str(ROOT / "probes/P42-fr-joint-tune")]
import p70  # noqa: E402
W = ROOT / "data/work"
SIB = {3, 4, 5, 7, 9, 11, 13, 21}


def hn_rel(h1, hx):
    ok = (h1 != "") & (hx != "") & np.char.isdigit(h1.astype(str)) & np.char.isdigit(hx.astype(str))
    d = np.full(len(h1), np.nan)
    d[ok] = hx[ok].astype(float) - h1[ok].astype(float)
    return np.select([~ok, d == 0, np.isin(d, list(SIB)), np.isin(d, [1, 2]), np.isin(d, [-1, -2])], [0, 1, 2, 3, 4], 5).astype(np.int8)


def ids(key):
    s1 = np.char.add("S1-", (key >> 31).astype(str))
    sx = np.char.add(np.where((key >> 30) & 1, "S3-", "S2-"), (key & (2**30 - 1)).astype(str))
    return s1.astype(object), sx.astype(object)


def main():
    t0 = time.time()
    files = json.loads((HERE / "work/files.json").read_text())
    u = pd.read_parquet(HERE / "work/union.parquet")
    full = (1 << len(files)) - 1
    nd = full & ~(1 << [n for n, _ in files].index("d1"))
    ch = u[(u["mask"].astype(int) & nd) != nd].reset_index(drop=True)      # pairs not in every non-diagnostic file
    print(f"changed pairs {len(ch):,} of {len(u):,}", flush=True)
    s1, sx = ids(ch.key.to_numpy())
    s1c = pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
    ch["c"] = s1c.reindex(s1).to_numpy()
    ch["is3"] = ((ch.key.to_numpy() >> 30) & 1).astype(np.int8)
    out = []
    for c in ["France", "India", "US"]:
        m = (ch.c == c).to_numpy()
        a1, ax = s1[m], sx[m]
        rxf = W / ("p29/w_fix/rf_test_sx_France.parquet" if c == "France" else f"rf_test_sx_{c}.parquet")
        R = p70.role_frame(a1, ax, "test", c, rxf, full=True)
        r1 = pd.read_parquet(W / f"rf_test_s1_{c}.parquet", columns=["id", "hn"]).set_index("id").hn
        rx = p70._rf(rxf, ax, ["hn"]).hn
        R["hn"] = hn_rel(r1.reindex(a1).fillna("").to_numpy(str), rx.reindex(ax).fillna("").to_numpy(str))
        d = ch[m].reset_index(drop=True)
        d = pd.concat([d, R[["kind", "w_s", "w_e", "lf_k", "gib_x", "noaddr_x", "twins1", "key_n", "hn"]].reset_index(drop=True)], axis=1)
        if c == "France":
            import p42
            z = p42.fr_frame()
            i1, ix = p42.fr_ids()
            fk = pd.DataFrame({"s1": i1[z["s1"]], "sx": ix[z["sx"]], "g": z["g"], "rb": z["r"], "sib": z["sib"]})
            q = pd.read_parquet(W / "p38/r/r_qwen_pool_test_France.parquet").rename(columns={"r": "rq"})
            d = d.assign(s1=a1, sx=ax).merge(fk, on=["s1", "sx"], how="left").merge(q, on=["s1", "sx"], how="left").drop(columns=["s1", "sx"])
            del z, fk, q
        out.append(d)
        print(f"{c}: {m.sum():,} pairs, {time.time() - t0:.0f}s, peak {p70.peak():.2f} GB", flush=True)
    pd.concat(out, ignore_index=True).to_parquet(HERE / "work/tagged.parquet", index=False)


if __name__ == "__main__":
    main()
