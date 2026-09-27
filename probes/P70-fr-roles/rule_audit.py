"""Rule-consistency audit of a file's France lines: pairs that a last-stage veto would still remove because an add ran after it.
  P64 legal-form conflict (pair_class 4), N54 decoy-word swap/add (france.decoy_mask's lens words, s > .25, n >= 20),
  hidden sibling shift (shift.flag, info), SX listed twice.
    rule_audit.py sub77 sub80"""
import csv, sys, collections
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P70-fr-roles"), str(ROOT / "probes/P68-iu-retrieval"), str(ROOT / "probes/P64-legal-form-feat"),
                str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import p70, shift, fr_key, lf  # noqa: E402
import france as FR  # noqa: E402
from patterns import name_rel  # noqa: E402
W = ROOT / "data/work"
FR.C = "France"
d = pd.read_parquet(W / "p31/fr31v0.1_t9868.parquet", columns=["s1", "sx", "p"]).sort_values("p", ascending=False).drop_duplicates("sx")
st = FR.lens_table(d.s1.to_numpy(object), d.sx.to_numpy(object))
dec = set(st.index[(st.n >= 20) & (st.s > 0.25)])
print("N54 decoy words:", len(dec), sorted(dec)[:15], "| groupe in:", "groupe" in dec, "developpement in:", "developpement" in dec, flush=True)
r1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "core", "hn"]).set_index("id")
rx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "core", "hn", "is_web"]).set_index("id")
for base in sys.argv[1:]:
    m = pd.read_csv(ROOT / "output" / base / "matching_results.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    m = m[m.source1_entity_id.isin(r1.index) & (m.matched_entity_ids != "")]
    e = m.assign(sx=m.matched_entity_ids.str.split(",")).explode("sx").rename(columns={"source1_entity_id": "s1"})[["s1", "sx"]].reset_index(drop=True)
    twice = sum(1 for c in collections.Counter(e.sx).values() if c > 1)
    n1, nx = fr_key.raw_names("test", set(e.s1), [1]).reindex(e.s1), fr_key.raw_names("test", set(e.sx), [2, 3]).reindex(e.sx)
    k = np.asarray(lf.pair_class(pd.Series(fr_key.sigs(n1)), pd.Series(fr_key.sigs(nx))))
    c1, cx = r1.core.reindex(e.s1).fillna("").to_numpy(), rx.core.reindex(e.sx).fillna("").to_numpy()
    wx = rx.is_web.reindex(e.sx).fillna(False).to_numpy()
    rel = [name_rel(a, b, w) if a and b else ("", "") for a, b, w in zip(c1, cx, wx)]
    n54 = np.array([kk in ("sub1", "add1") and w in dec for kk, w in rel])
    a1 = p70._raw("test", set(e.s1), [1], ["business_address", "country"]).business_address.reindex(e.s1).reset_index(drop=True)
    ax = p70._raw("test", set(e.sx), [2, 3], ["business_address", "country"]).business_address.reindex(e.sx).reset_index(drop=True)
    hs = shift.flag(a1, ax, r1.hn.reindex(e.s1).fillna("").to_numpy(), rx.hn.reindex(e.sx).fillna("").to_numpy())
    e["p64"], e["n54"], e["shift"] = k == 4, n54, hs
    e[e.p64 | e.n54 | e["shift"]].to_parquet(W / f"p70/audit_{base}.parquet", index=False)
    print(f"{base}: France pairs {len(e):,} | P64 class-4 left {int((k == 4).sum())} | N54 decoy-word left {int(n54.sum())} | "
          f"hidden shift {int(hs.sum())} | SX listed twice {twice}", flush=True)
