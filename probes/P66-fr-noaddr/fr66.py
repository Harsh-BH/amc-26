"""P66 France: the loco66 address-less classes applied to France on top of P37_BASE (default output/sub_iu54_frgc70).
    .venv/bin/python probes/P66-fr-noaddr/fr66.py count                  -> France adds per class / floor
    .venv/bin/python probes/P66-fr-noaddr/fr66.py build CLASS T OUT [BASE2]
        OUT = P37_BASE + the class's France adds at France p >= T. BASE2 (optional, e.g. output/sub_iu54_best): the same
        adds applied to BASE2 instead (pairs whose SX BASE2 already owns are dropped; caps 5 S2 / 6 S3 re-checked).
"""
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault("P37_BASE", str(ROOT / "output/sub_iu54_frgc70"))
sys.path[:0] = [str(ROOT / "probes/P37-fr-complete"), str(ROOT / "probes/P40-generator"), str(ROOT / "probes/P28-addr-anchor"),
                str(HERE)]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import p37 as P  # noqa: E402
import rule40 as R  # noqa: E402
from loco66 import FR_T, addrless_ids, classes, feats, fk_counts  # noqa: E402

RD = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def state():
    d, a = P.fr_load()
    na = d.sx.isin(set(addrless_ids("test", "France"))).to_numpy()
    w = np.flatnonzero(a["am"] & na)
    f = feats("test", "France", d.s1.to_numpy()[w], d.sx.to_numpy()[w], fk_counts("test", "France"))
    return d, a, w, classes(f)


def adds(d, a, w, m, t):
    full = np.zeros(len(d), bool)
    full[w] = m
    k = R.add(a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["pred"], a["sib"], full, t)
    return d[k][["s1", "sx", "p"]].reset_index(drop=True)


def count():
    d, a, w, cls = state()
    has = set(d.s1.to_numpy()[a["pred"]])
    res = {"noaddr_am": len(w), "noaddr_am_pred": int(a["pred"][w].sum())}
    for k, m in cls.items():
        for t in FR_T:
            x = adds(d, a, w, m, t)
            res[f"{k}@{t}"] = {"added": len(x), "s1": int(x.s1.nunique()), "fills_empty": int((~x.s1.drop_duplicates().isin(has)).sum())}
            print(k, t, res[f"{k}@{t}"], flush=True)
    (HERE / "fr66_count.json").write_text(json.dumps(res, indent=1))


def build(cls_name, t, out, base2=None):
    d, a, w, cls = state()
    x = adds(d, a, w, cls[cls_name], float(t))
    base = Path(base2) if base2 else P.BASE
    sub = pd.read_csv(base / "matching_results.tsv", **RD)
    e = sub.assign(sx=sub.matched_entity_ids.str.split(",")).explode("sx")
    e = e[e.sx.notna() & (e.sx != "")]
    owned = set(e.sx)
    n2 = e[e.sx.str.startswith("S2-")].groupby("source1_entity_id").size()
    n3 = e[e.sx.str.startswith("S3-")].groupby("source1_entity_id").size()
    del e
    x = x[~x.sx.isin(owned)].sort_values(["s1", "p"], ascending=[True, False])
    x["is2"] = x.sx.str.startswith("S2-")
    x["rk"] = x.groupby(["s1", "is2"]).cumcount()
    room = np.where(x.is2, 5 - n2.reindex(x.s1).fillna(0).to_numpy(), 6 - n3.reindex(x.s1).fillna(0).to_numpy())
    x = x[x.rk.to_numpy() < room]
    # P64's legal-form conflict veto (class 4) must not be undone by an add (none expected: exact-name adds share the form)
    sys.path.insert(0, str(ROOT / "probes/P64-legal-form-feat"))
    import lf
    from loco66 import names
    k = lf.pair_class(lf.sig(names("test", x.s1, (1,)).reindex(x.s1).reset_index(drop=True)),
                      lf.sig(names("test", x.sx, (2, 3)).reindex(x.sx).reset_index(drop=True)))
    n_conf = int((k == 4).sum())
    x = x[k != 4]
    new = x.groupby("s1").sx.agg(list).to_dict()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(base / "matching_results.tsv", encoding="utf-8") as f, open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write(next(f))
        for line in f:
            s, m = line.rstrip("\n").split("\t", 1)
            if s in new:
                cur = [y for y in m.split(",") if y]
                assert not set(cur) & set(new[s])
                g.write(f"{s}\t{','.join(cur + sorted(new[s]))}\n")
                n += 1
            else:
                g.write(line)
    assert n == len(new)
    for name in ("candidate_pairs.tsv", "code.zip"):
        if not (out / name).exists() and (base / name).exists():
            os.link(base / name, out / name)
    info = {"base": str(base), "pairs_from": str(P.BASE), "rule": f"p66 {cls_name}@{t}", "added": len(x), "s1_changed": n, "p64_conflicts_skipped": n_conf}
    (out / "p66_info.json").write_text(json.dumps(info))
    print(json.dumps(info))


if __name__ == "__main__":
    {"count": count, "build": build}[sys.argv[1]](*sys.argv[2:])
