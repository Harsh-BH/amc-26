"""Scan hit: same name core, S1 core unique, same legal form, address present, not a brand, hn different (not a sibling
offset). LOCO free-pair precision split by |dhn| bucket and street relation (+ add dF); France free pool with the same splits.
    uhn.py loco | uhn.py fr BASE"""
import csv, gc, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P70-fr-roles")]
import p70, scan, street_fr, shift
W = ROOT / "data/work"
CELL = "eq|other|lf1|na0|gib0|u1"
bucket = lambda d: np.select([d <= 2, d <= 10, d <= 100], ["1-2", "6-10 (non-sib)", "11-100 (non-sib)"], ">100")
def wil(k, n):
    if n == 0: return None
    p, dd = k / n, 1 + 3.8416 / n
    c, h = (p + 1.92 / n) / dd, 1.96 * math.sqrt(p * (1 - p) / n + 0.9604 / n / n) / dd
    return [round(c - h, 3), round(c + h, 3)]


def streets(a1, ax, c):
    cc = pd.Series([c] * len(a1))
    k1, kx = p70._street(a1.fillna("").reset_index(drop=True), cc), p70._street(ax.fillna("").reset_index(drop=True), cc)
    return np.array([("same" if a.split("|", 1)[-1] == b.split("|", 1)[-1] else street_fr.classify(a, b)) if a and b else "nostreet" for a, b in zip(k1, kx)])


def loco():
    p55, L = p70.p55, p70.L
    out = {}
    for dr in L.DIRS:
        tgt = L.DIRS[dr][1]
        fr = p55.loco_frame(dr)
        ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet", columns=["s1", "sx"])
        R = pd.read_parquet(W / f"p70/roles_{dr}.parquet", columns=["kind", "lf_k", "gib_x", "noaddr_x", "twins1"])
        h1 = pd.read_parquet(W / f"rf_train_s1_{tgt}.parquet", columns=["id", "hn"]).set_index("id").hn.reindex(ids.s1).fillna("").to_numpy()
        hx = pd.read_parquet(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "hn"]).set_index("id").hn.reindex(ids.sx).fillna("").to_numpy()
        kind = pd.Series(R.kind.to_numpy()).map(scan.KINDS).fillna("none").to_numpy()
        c = scan.cell(kind, scan.hrel(h1, hx), R.lf_k.fillna(-1).astype(int).to_numpy(), R.noaddr_x.to_numpy() > 0, R.gib_x.to_numpy() > 0, R.twins1.to_numpy() == 1).to_numpy()
        m = c == CELL
        idx = np.flatnonzero(m)
        a1 = p70._raw("train", ids.s1.iloc[idx], [1], ["business_address", "country"]).business_address.reindex(ids.s1.iloc[idx])
        ax = p70._raw("train", ids.sx.iloc[idx], [2, 3], ["business_address", "country"]).business_address.reindex(ids.sx.iloc[idx])
        st = streets(a1, ax, tgt)
        sh = shift.flag(a1.reset_index(drop=True), ax.reset_index(drop=True), h1[idx], h1[idx])  # not used for hn-different pairs
        dl = np.abs(pd.to_numeric(pd.Series(hx[idx])).to_numpy(float) - pd.to_numeric(pd.Series(h1[idx])).to_numpy(float))
        z, ref, free, y = fr["z"], fr["ref"], fr["free"] & ~fr["ref"], fr["z"]["y"].astype(bool)
        fref = L.F(z, ref)
        res = {}
        splits = {"all": np.ones(len(idx), bool), "street same": st == "same", "street typo/type": np.isin(st, ["typo", "type_only"]),
                  "street other": st == "other", **{f"dhn {b}": bucket(dl) == b for b in ("1-2", "6-10 (non-sib)", "11-100 (non-sib)", ">100")}}
        for nm, sm in splits.items():
            sel = np.zeros(len(y), bool); sel[idx[sm]] = True
            f_ = sel & free
            k = int(y[f_].sum())
            r = {"n": int(sel.sum()), "free_n": int(f_.sum()), "free_p": round(k / f_.sum(), 3) if f_.any() else None, "wilson": wil(k, int(f_.sum()))}
            if f_.any():
                r |= {kk: v for kk, v in L.cmp(z, ref, p55.decide(fr, f_.astype(float), None, 0.5), fref).items() if kk in ("dF_h1", "se_h1")}
            res[nm] = r
            print(f"{dr} (target {tgt}) {nm:22s} {json.dumps(r)}", flush=True)
        out[dr] = res
        del fr, ids, R; gc.collect()
    (Path(__file__).parent / "uhn_loco.json").write_text(json.dumps(out, indent=1))


def fr(base):
    s = pd.read_parquet(W / f"p70/scan_fr_{base}.parquet")
    s = s[(s.cell == CELL) & s.free].reset_index(drop=True)
    h1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "hn"]).set_index("id").hn.reindex(s.s1).fillna("").to_numpy()
    hx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "hn"]).set_index("id").hn.reindex(s.sx).fillna("").to_numpy()
    a1 = p70._raw("test", set(s.s1), [1], ["business_address", "country"]).business_address.reindex(s.s1)
    ax = p70._raw("test", set(s.sx), [2, 3], ["business_address", "country"]).business_address.reindex(s.sx)
    s["street"] = streets(a1, ax, "France")
    s["dhn"] = bucket(np.abs(pd.to_numeric(pd.Series(hx)).to_numpy(float) - pd.to_numeric(pd.Series(h1)).to_numpy(float)))
    print(f"France free in {CELL}: {len(s)}; p median {s.p.median():.3f}")
    print(pd.crosstab(s.dhn, s.street, margins=True).to_string())
    s.to_parquet(W / f"p70/uhn_fr_{base}.parquet", index=False)


if __name__ == "__main__":
    loco() if sys.argv[1] == "loco" else fr(sys.argv[2])
