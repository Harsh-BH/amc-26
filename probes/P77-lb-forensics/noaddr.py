"""P77: France address-less S2/S3 left unassigned by a scored file: their argmax France candidate S1, g band and P20 name
relation (label-free). Raw examples go to data/work/p77/ (gitignored; work/ is a symlink to it), never into RESULT.md.
    .venv/bin/python probes/P77-lb-forensics/noaddr.py sub77 -> work/noaddr_<name>.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P70-fr-roles"), str(ROOT / "probes/P42-fr-joint-tune")]
import p70, p42  # noqa: E402
import importlib.util as _iu
_sp = _iu.spec_from_file_location("p77pat", HERE / "patterns.py"); _m = _iu.module_from_spec(_sp); _sp.loader.exec_module(_m)
noaddr = _m.noaddr
W = ROOT / "data/work"

name = sys.argv[1]
names = [n for n, _ in json.loads((HERE / "work/files.json").read_text())]
u = pd.read_parquet(HERE / "work/union.parquet")
k = u.key.to_numpy()[((u["mask"].to_numpy().astype(np.int64) >> names.index(name)) & 1) == 1]
del u
z = p42.fr_frame()
i1, ix = p42.fr_ids()
sx_id, s1_id = ix[z["sx"]], i1[z["s1"]]
enc = lambda a, b: (pd.Series(a).str.slice(3).astype(np.int64).to_numpy() << 31) | ((pd.Series(b).str.slice(0, 2) == "S3").to_numpy().astype(np.int64) << 30) | pd.Series(b).str.slice(3).astype(np.int64).to_numpy()
sxk = ((pd.Series(ix).str.slice(0, 2) == "S3").to_numpy().astype(np.int64) << 30) | pd.Series(ix).str.slice(3).astype(np.int64).to_numpy()
assigned_sx = np.isin(sxk, k & (2**31 - 1))
s1k = pd.Series(i1).str.slice(3).astype(np.int64).to_numpy()
s1_nonempty = np.isin(s1k, k >> 31)
raw = []
for src in (2, 3):
    t = pd.read_parquet(W / f"test_source{src}.parquet", columns=["entity_id", "business_address", "country"], filters=[("country", "=", "France")])
    raw.append(t.assign(na=noaddr(t.business_address))[["entity_id", "na"]])
na = pd.concat(raw).set_index("entity_id").na.reindex(ix).fillna(False).to_numpy()
g = z["g"]
# argmax candidate per SX
o = np.lexsort((-g, z["sx"]))
first = np.r_[True, z["sx"][o][1:] != z["sx"][o][:-1]]
am = o[first]
sel = am[na[z["sx"][am]] & ~assigned_sx[z["sx"][am]]]
print(f"France address-less SX: {na.sum():,}; assigned {assigned_sx[na].mean():.3f}; free with a candidate: {len(sel):,}", flush=True)
R = p70.role_frame(s1_id[sel], sx_id[sel], "test", "France", W / "p29/w_fix/rf_test_sx_France.parquet", full=False)
kind = pd.Series(R.kind.to_numpy()).map({v: k_ for k_, v in p70.KIND.items()}).fillna("?").to_numpy()
r1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "core"])
tw = r1.set_index("id").core.map(r1.core.value_counts()).reindex(s1_id[sel]).fillna(0).to_numpy()
d = pd.DataFrame({"kind": kind, "g": g[sel], "s1_empty": ~s1_nonempty[z["s1"][sel]], "core_unique": tw == 1,
                  "gb": pd.cut(g[sel], [-1, 0.1, 0.3, 0.5, 0.7, 0.9, 2], labels=["<.1", ".1-.3", ".3-.5", ".5-.7", ".7-.9", ">=.9"]).astype(str)})
out = {"noaddr_sx": int(na.sum()), "assigned_share": round(float(assigned_sx[na].mean()), 4), "free_with_cand": len(sel),
       "kind": d.kind.value_counts().to_dict(), "g_band": d.gb.value_counts().to_dict(),
       "kind_x_core_unique_g>=.3": d[d.g >= 0.3].groupby(["kind", "core_unique"]).size().unstack(fill_value=0).to_dict(),
       "s1_empty_share": round(float(d.s1_empty.mean()), 4)}
print(json.dumps(out, indent=0, default=int))
(HERE / f"work/noaddr_{name}.json").write_text(json.dumps(out, indent=1, default=int))
# 3 raw examples per top kind (gitignored)
t1 = pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "business_name", "business_address"], filters=[("country", "=", "France")]).set_index("entity_id")
tx = pd.concat([pd.read_parquet(W / f"test_source{s}.parquet", columns=["entity_id", "business_name", "business_address"], filters=[("country", "=", "France")]) for s in (2, 3)]).set_index("entity_id")
with open(W / "p77/examples_noaddr.txt", "w") as f:
    for kd in ["eq", "typo", "sub1", "add1", "drop1", "disjoint"]:
        m = np.flatnonzero((d.kind == kd).to_numpy() & (d.g >= 0.3).to_numpy())[:3]
        for i in m:
            a, b = s1_id[sel][i], sx_id[sel][i]
            f.write(f"{kd}\tg={g[sel][i]:.3f}\t{t1.business_name.get(a)} | {t1.business_address.get(a)}\t=>\t{tx.business_name.get(b)} | {tx.business_address.get(b)}\n")
