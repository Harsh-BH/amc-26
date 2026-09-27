"""P40 x P41: rule40's non-gib part (xu_noaddr: address-less SX, SX core == S1 core, core unique among the corpus S1,
>= 5 chars) ON TOP of P41's sub41 (gc70 + key1_bare gib rule at FR 0.3). LOCO base = P41's LOCO base + its rule;
France base = output/sub41. Also: overlap of sub40's xu_noaddr pairs with sub41's adds.
    P41_BASE=output/sub41 probes/P40-generator/py.sh probes/P40-generator/loco42.py loco|france
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P41-fr-brand"), str(HERE), str(ROOT / "probes/P28-addr-anchor")]
import p41 as Q  # noqa: E402  (sets P37_BASE = P41_BASE before p37 is imported)
import rule40 as R  # noqa: E402
from cand import raw  # noqa: E402
from loco41 import core_of, uniq_counts  # noqa: E402
from loco40 import fr_dens  # noqa: E402

P = Q.P
W = ROOT / "data/work"
TS_ = (0.9, 0.7, 0.5)
T41 = 0.3


MODE = os.environ.get("P40_XU", "core")      # core: rule40's xu_noaddr; full: full normalised name (legal kept) unique


def fkey(s):
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("&", " and ")
    return " ".join(re.findall(r"\w+", s))


_FULL = {}


def full_counts(split, country):
    if (split, country) not in _FULL:
        import pyarrow.parquet as pq
        t = pq.read_table(W / f"{split}_source1.parquet", columns=["business_name", "country"],
                          filters=[("country", "=", country)]).column(0).to_pylist()
        _FULL[(split, country)] = pd.Series([fkey(x) for x in t]).value_counts().to_dict()
    return _FULL[(split, country)]


def xu_class(ids1, idsx, n1, nx, ax, c1, cx, cc, split=None, country=None):
    if MODE == "core":
        return R.op_class(n1, nx, ax, c1, cx, cc, gib=False)
    return R.xf_noaddr(n1, nx, ax, full_counts(split, country))


def loco():
    t0 = time.time()
    dens = fr_dens()
    dens[T41] = float(__import__("pandas").read_parquet(W / "p40/fr_pairs.parquet", columns=["p"]).p.pipe(
        lambda p: ((p >= T41) & (p < P.T_FR)).sum()) / 259452)
    out = {"dens": dens}
    for src, tgt in (("US", "India"), ("India", "US")):
        k1, kx = Q.keys_train(tgt)
        gk = Q.gib_keys(k1, kx)
        gk["bare"] = Q.bare("train", tgt, gk.index).reindex(gk.index).fillna(False).to_numpy(bool)
        k1i = k1.set_index("id")[["hn", "street", "city"]]
        del k1, kx
        A = P.loco_load(src, tgt)
        s1, sx, is2, p, am, y, sib, n = (A[k] for k in ("s1", "sx", "is2", "p", "am", "y", "sib", "n_s1"))
        ntrue, hv, tb = A["t2"] + A["t3"], A["hv"], P.TS[(src, tgt)]
        g41 = Q.guards(A["s1_ids"], A["sx_ids"], p, gk, k1i)["key1_bare"]
        del k1i, gk
        tm = {t: P.inv_dens(p, am, tb, g, n) for t, g in dens.items()}
        per = lambda pr: P.fbeta(np.bincount(s1[pr & y], minlength=n), np.bincount(s1[pr], minlength=n), ntrue)
        pred = am & (p >= tb)
        pred |= P.rescue(s1, p, am, pred, sib, tm[P.T_RESCUE])
        pred |= P.gc_add(s1, sx, is2, p, am, pred, sib, tm[0.7], "oneall")
        f_gc = per(pred)
        add41 = Q.rule(P, s1, sx, is2, p, am, pred, sib, tm[T41], g41)
        pred41 = pred | add41
        f41 = per(pred41)
        r = {"t_base": tb, "tmap": {str(k): round(v, 4) for k, v in tm.items()}, "sub41_rule_vs_gc70": round((f41 - f_gc).mean(), 6),
             "sub41_added": int(add41.sum()), "sub41_prec": round(float(y[add41].mean()), 4)}
        print(f"{src}->{tgt}: sub41 rule on LOCO gc70 base dF {r['sub41_rule_vs_gc70']:+.6f}, added {r['sub41_added']}, prec {r['sub41_prec']}",
              flush=True)
        for base_name, bp, fb in (("vs_gc70", pred, f_gc), ("vs_sub41", pred41, f41)):
            pool = am & ~bp & (p >= min(tm[t] for t in TS_))
            ids1, idsx = A["s1_ids"][pool], A["sx_ids"][pool]
            r1, rx = raw(ids1, (1,)), raw(idsx, (2, 3))
            cls = np.zeros(len(p), bool)
            cls[pool] = xu_class(ids1, idsx, r1.business_name.reindex(ids1).to_numpy(), rx.business_name.reindex(idsx).to_numpy(),
                                 rx.business_address.reindex(idsx).to_numpy(), core_of(W / f"rf_train_s1_{tgt}.parquet", ids1),
                                 core_of(W / f"rf_train_sx_{tgt}.parquet", idsx), uniq_counts(W / f"rf_train_s1_{tgt}.parquet").to_dict(),
                                 "train", tgt)
            for t in TS_:
                add = R.add(s1, sx, is2, p, am, bp, sib, cls, tm[t])
                dl = per(bp | add) - fb
                x = dl[hv == 1]
                row = {"d": round(dl.mean(), 6), "d_h1": round(x.mean(), 6), "se_h1": round(x.std() / np.sqrt(len(x)), 6),
                       "added": int(add.sum()), "prec": round(float(y[add].mean()), 4) if add.any() else None,
                       "overlap_sub41_adds": int((add & add41).sum()), "shared_s1_with_sub41_adds":
                       int(np.isin(s1[add], s1[add41]).sum())}
                r[f"xu_noaddr_{base_name}@{t}"] = row
                print(f"  xu_noaddr {base_name} FR t {t} (tgt {tm[t]:.3f}): dF {row['d']:+.6f} (h1 {row['d_h1']:+.6f} ± {row['se_h1']:.6f}) "
                      f"added {row['added']} prec {row['prec']} overlap {row['overlap_sub41_adds']} sharedS1 {row['shared_s1_with_sub41_adds']}",
                      flush=True)
        out[f"{src}->{tgt}"] = r
        del A
        import gc
        gc.collect()
    out["runtime_s"] = round(time.time() - t0)
    (HERE / f"loco42{'' if MODE == 'core' else '_' + MODE}.json").write_text(json.dumps(out, indent=1, default=str))


def traw(ids, srcs):
    """Raw test rows by id (same as france.py's; that name clashes with the pipeline's france module)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    v = pa.array(sorted(set(ids)), pa.large_string())
    return pd.concat([pq.read_table(W / f"test_source{i}.parquet", filters=[("entity_id", "in", v)]).to_pandas()
                      for i in srcs]).drop_duplicates("entity_id").set_index("entity_id")


def fr_adds(t, base=None):
    """France xu_noaddr pairs on top of P37_BASE (= sub41 here): DataFrame s1, sx."""
    d, a = P.fr_load()
    s1, sx, is2, p, am, pred, sib = (a[k] for k in ("s1", "sx", "is2", "p", "am", "pred", "sib"))
    pool = am & ~pred & (p >= t)
    ids1, idsx = d.s1.to_numpy()[pool], d.sx.to_numpy()[pool]
    r1, rx = traw(ids1, (1,)), traw(idsx, (2, 3))
    cls = np.zeros(len(p), bool)
    cls[pool] = xu_class(ids1, idsx, r1.business_name.reindex(ids1).to_numpy(), rx.business_name.reindex(idsx).to_numpy(),
                         rx.business_address.reindex(idsx).to_numpy(), core_of(W / "rf_test_s1_France.parquet", ids1),
                         core_of(W / "rf_test_sx_France.parquet", idsx), uniq_counts(W / "rf_test_s1_France.parquet").to_dict(),
                         "test", "France")
    m = R.add(s1, sx, is2, p, am, pred, sib, cls, t)
    return d[m][["s1", "sx"]].reset_index(drop=True)


def pairs_of(path, fr):
    RD = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    s = pd.read_csv(path, **RD)
    s = s[s.source1_entity_id.isin(fr)]
    e = s.assign(sx=s.matched_entity_ids.str.split(",")).explode("sx")
    e = e[e.sx.notna() & (e.sx != "")]
    return set(zip(e.source1_entity_id, e.sx))


def france():
    import pyarrow.parquet as pq
    assert P.BASE.name == "sub41", P.BASE
    fr = set(pq.read_table(W / "test_source1.parquet", columns=["entity_id"], filters=[("country", "=", "France")]).column(0).to_pylist())
    gc70, s41, s40 = (pairs_of(ROOT / f"output/{x}/matching_results.tsv", fr) for x in ("sub37_gc70", "sub41", "sub40"))
    add41, add40 = s41 - gc70, s40 - gc70
    res = {"sub41_adds": len(add41), "sub40_adds": len(add40), "sub40_adds_in_sub41": len(add40 & add41)}
    for t in TS_:
        a = fr_adds(t)
        k = set(zip(a.s1, a.sx))
        res[f"xu_noaddr_on_sub41@{t}"] = {"added": len(k), "s1": int(a.s1.nunique()), "in_sub41_adds": len(k & add41),
                                          "in_sub40_adds": len(k & add40), "s1_shared_with_sub41_adds": len(set(a.s1) & {s for s, _ in add41})}
    print(json.dumps(res, indent=1))
    (HERE / f"loco42_france{'' if MODE == 'core' else '_' + MODE}.json").write_text(json.dumps(res, indent=1))


def build(t, out):
    """output/sub40x* = P37_BASE (sub41) + the class's France pairs at p >= t; India/US lines, candidates, code.zip = base's."""
    t, out = float(t), Path(out)
    assert P.BASE.name == "sub41", P.BASE
    a = fr_adds(t)
    new = a.groupby("s1").sx.agg(list).to_dict()
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(P.BASE / "matching_results.tsv", encoding="utf-8") as f, open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write(next(f))
        for line in f:
            s, m = line.rstrip("\n").split("\t", 1)
            if s in new:
                cur = [x for x in m.split(",") if x]
                assert not set(cur) & set(new[s])
                g.write(f"{s}\t{','.join(cur + sorted(new[s]))}\n")
                n += 1
            else:
                g.write(line)
    assert n == len(new)
    for name in ("candidate_pairs.tsv", "code.zip"):
        if not (out / name).exists():
            os.link(P.BASE / name, out / name)
    info = {"base": str(P.BASE), "rule": f"xf_noaddr@{t}" if MODE == "full" else f"xu_noaddr@{t}", "added": len(a), "s1_changed": n}
    (out / "p40_info.json").write_text(json.dumps(info))
    print(json.dumps(info))


if __name__ == "__main__":
    {"loco": loco, "france": france, "build": build}[sys.argv[1]](*sys.argv[2:])
