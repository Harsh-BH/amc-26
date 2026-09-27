"""P63 step 1: correction maps and rates per country (name family = P3 name + rules core; addr family = P3 addr).

    .venv/bin/python probes/P63-fr-typo-correct/rates.py train India | train US | test France
    -> data/work/p63/map_<split>_<c>_{name,addr}.parquet (oov -> corrected) + rates_<split>_<c>.json
On train it also checks the corrections on TRUE pairs: share of changed SX tokens whose correction is a token of the
true S1 (precision) vs the uncorrected token (it was already there) -- label use for the report only.
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "code/business_entity_resolution/src")]
import typo as T  # noqa: E402

W = ROOT / "data/work"
OUT = W / "p63"
TXT = {"train": W / "p3/txt", "test": W / "p3/txt_test"}
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def sx_files(split, c):
    return sorted(TXT[split].glob(f"{c}_sx_*.parquet"))


def core_of(split, c, side, ids):
    t = pq.read_table(W / f"rf_{split}_{side}_{c}.parquet", columns=["id", "core"]).to_pandas()
    return t.set_index("id").core.reindex(ids).fillna("")


def s1_texts(split, c):
    t = pd.read_parquet(TXT[split] / f"{c}_s1.parquet", columns=["entity_id", "name", "addr"])
    t["core"] = core_of(split, c, "s1", t.entity_id).to_numpy()
    return t


def maps(split, c):
    t0 = time.time()
    s1 = s1_texts(split, c)
    vn = T.counts(s1.name).add(T.counts(s1.core), fill_value=0)
    va = T.counts(s1.addr)
    del s1
    rf = pq.read_table(W / f"rf_{split}_sx_{c}.parquet", columns=["id", "core"]).to_pandas().set_index("id").core
    qn, qa = pd.Series(dtype=float), pd.Series(dtype=float)
    for f in sx_files(split, c):
        x = pd.read_parquet(f, columns=["entity_id", "name", "addr"])
        qn = qn.add(T.counts(x.name).add(T.counts(rf.reindex(x.entity_id).fillna("")), fill_value=0), fill_value=0)
        qa = qa.add(T.counts(x.addr), fill_value=0)
    del rf
    print(f"{split} {c}: V name {len(vn):,} addr {len(va):,}; SX name {len(qn):,} addr {len(qa):,} tokens "
          f"({time.time() - t0:.0f}s, {peak():.2f} GB)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for fam, v, q in (("name", vn, qn), ("addr", va, qa)):
        m = T.build_map(v, q)
        m.rename("to").rename_axis("tok").reset_index().to_parquet(OUT / f"map_{split}_{c}_{fam}.parquet", index=False)
        inv = ~q.index.isin(v.index)
        res[fam] = {"V": len(v), "sx_types": len(q), "oov_types": int(inv.sum()),
                    "oov_tok_share": round(float(q[inv].sum() / q.sum()), 4), "corrected_types": len(m),
                    "corrected_tok_share": round(float(q.reindex(m.index).sum() / q.sum()), 4)}
        print(f"  {fam}: {res[fam]}  ({time.time() - t0:.0f}s, {peak():.2f} GB)", flush=True)
    return res


def record_rates(split, c, res):
    """share of SX records with >= 1 corrected token per field, + 30 example rewrites per family."""
    mn = pd.read_parquet(OUT / f"map_{split}_{c}_name.parquet").set_index("tok").to
    ma = pd.read_parquet(OUT / f"map_{split}_{c}_addr.parquet").set_index("tok").to
    rf = pq.read_table(W / f"rf_{split}_sx_{c}.parquet", columns=["id", "core"]).to_pandas().set_index("id").core
    n, ch = 0, {"name": 0, "core": 0, "addr": 0, "any": 0}
    ex = {"name": [], "addr": []}
    for f in sx_files(split, c):
        x = pd.read_parquet(f, columns=["entity_id", "name", "addr"])
        x["core"] = rf.reindex(x.entity_id).fillna("").to_numpy()
        nn, a = T.apply(x.name, mn)
        _, b = T.apply(x.core, mn)
        na, cc = T.apply(x.addr, ma)
        n += len(x)
        ch["name"] += int(a.sum()); ch["core"] += int(b.sum()); ch["addr"] += int(cc.sum()); ch["any"] += int((a | b | cc).sum())
        if len(ex["name"]) < 30:
            ex["name"] += [f"{u} -> {w}" for u, w in zip(x.name[a].head(30), nn[a].head(30))]
            ex["addr"] += [f"{u} -> {w}" for u, w in zip(x.addr[cc].head(30), na[cc].head(30))]
    res["records"] = {"n_sx": n, **{f"share_{k}": round(v / n, 4) for k, v in ch.items()}}
    res["examples"] = {k: v[:30] for k, v in ex.items()}
    print(f"  records: {res['records']}", flush=True)
    return res


if __name__ == "__main__":
    split, c = sys.argv[1], sys.argv[2]
    t0 = time.time()
    res = record_rates(split, c, maps(split, c))
    res["runtime_s"], res["peak_gb"] = round(time.time() - t0), round(peak(), 2)
    (OUT / f"rates_{split}_{c}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"done {time.time() - t0:.0f}s peak {peak():.2f} GB")
