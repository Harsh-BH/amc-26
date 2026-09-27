"""Hidden sibling shift: the parsed house numbers are equal, but the address's other numbers (1-4 digit runs, postcodes excluded)
differ in exactly one position by a sibling offset (SIB3) or by 1-2. The parser reads only the first digit run of the first
numeric segment, so a decoy whose shift lands in a range end ('41-43' vs '41-45'), a second number ('13 1' vs '13 3') or the
real street number behind a postcode ('59430 X, 52 rue' vs '59430 X, 61 rue') looks hn-equal.
    shift.py loco            # LOCO: precision of accepted / free flagged pairs + veto dF (P55 frames)
    shift.py fr BASE         # France: flagged accepted pairs in BASE (label-free count)"""
import csv, gc, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P70-fr-roles"), str(ROOT / "code/business_entity_resolution/src")]
import p70
import normalize as N
W = ROOT / "data/work"
SIB = {3, 4, 5, 7, 9, 11, 13, 21}
_NUM = re.compile(r"(?<!\d)\d{1,4}(?!\d)")


def nums(addr: pd.Series) -> list:
    f = N.fold(addr.fillna("")).str.replace(r"(?<!\d)\d{5}(?!\d)", " ", regex=True)     # drop 5-digit postcodes
    return [tuple(int(x) for x in _NUM.findall(s)) for s in f]


def flag(a1: pd.Series, ax: pd.Series, h1, hx) -> np.ndarray:
    A, B = nums(a1), nums(ax)
    out = np.zeros(len(A), bool)
    for i, (a, b) in enumerate(zip(A, B)):
        if len(a) != len(b) or len(a) < 1 or h1[i] != hx[i] or h1[i] == "":
            continue
        d = [abs(u - v) for u, v in zip(a, b) if u != v]
        out[i] = len(d) == 1 and (d[0] in SIB or d[0] in (1, 2))
    return out


def loco():
    p55, L = p70.p55, p70.L
    out = {}
    for dr in L.DIRS:
        t0, tgt = time.time(), L.DIRS[dr][1]
        fr = p55.loco_frame(dr)
        ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet", columns=["s1", "sx"])
        a1 = p70._raw("train", ids.s1, [1], ["business_address", "country"]).business_address.reindex(ids.s1).reset_index(drop=True)
        ax = p70._raw("train", ids.sx, [2, 3], ["business_address", "country"]).business_address.reindex(ids.sx).reset_index(drop=True)
        h1 = pd.read_parquet(W / f"rf_train_s1_{tgt}.parquet", columns=["id", "hn"]).set_index("id").hn.reindex(ids.s1).fillna("").to_numpy()
        hx = pd.read_parquet(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "hn"]).set_index("id").hn.reindex(ids.sx).fillna("").to_numpy()
        f = flag(a1, ax, h1, hx)
        z, ref, free, y = fr["z"], fr["ref"], fr["free"], fr["z"]["y"].astype(bool)
        fref = L.F(z, ref)
        acc, fre = f & ref, f & free & ~ref
        veto = L.cmp(z, ref, ref & ~f, fref)
        out[dr] = {"flagged": int(f.sum()), "prec_all": round(float(y[f].mean()), 3) if f.any() else None,
                   "accepted_n": int(acc.sum()), "accepted_prec": round(float(y[acc].mean()), 3) if acc.any() else None,
                   "free_n": int(fre.sum()), "free_prec": round(float(y[fre].mean()), 3) if fre.any() else None,
                   "veto": {k: v for k, v in veto.items() if k in ("dF", "dF_h1", "se_h1", "rm", "rm_false")}}
        print(dr, json.dumps(out[dr]), f"{time.time() - t0:.0f}s peak {p70.peak():.2f} GB", flush=True)
        del fr, ids, a1, ax; gc.collect()
    (W / "p70/shift_loco.json").write_text(json.dumps(out, indent=1))


def fr(base):
    r1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "hn"]).set_index("id").hn
    m = pd.read_csv(ROOT / "output" / base / "matching_results.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    m = m[m.source1_entity_id.isin(r1.index) & (m.matched_entity_ids != "")]
    e = m.assign(sx=m.matched_entity_ids.str.split(",")).explode("sx")[["source1_entity_id", "sx"]].rename(columns={"source1_entity_id": "s1"}).reset_index(drop=True)
    hx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "hn"]).set_index("id").hn
    a1 = p70._raw("test", set(e.s1), [1], ["business_address", "country"]).business_address.reindex(e.s1).reset_index(drop=True)
    ax = p70._raw("test", set(e.sx), [2, 3], ["business_address", "country"]).business_address.reindex(e.sx).reset_index(drop=True)
    f = flag(a1, ax, r1.reindex(e.s1).fillna("").to_numpy(), hx.reindex(e.sx).fillna("").to_numpy())
    e["flag"] = f
    e[f].to_parquet(W / f"p70/shift_fr_{base}.parquet", index=False)
    print(json.dumps({"base": base, "accepted_pairs": len(e), "flagged": int(f.sum()), "s1": int(e.s1[f].nunique())}))


if __name__ == "__main__":
    loco() if sys.argv[1] == "loco" else fr(sys.argv[2])
