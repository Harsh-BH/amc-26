"""P39 France build: sub34v_rescue90 with France's rows re-decided by the group model; India / US lines byte-identical,
candidate_pairs.tsv hard-linked (the France candidate set does not change; matches stay a subset of it).

    .venv/bin/python probes/P39-fr-group-model/build.py check                    # rebuild #34v_r90's France from the frame
    .venv/bin/python probes/P39-fr-group-model/build.py check [r90|gc70|hn90]     # ... or sub37_gc70(_hn90)'s France
    .venv/bin/python probes/P39-fr-group-model/build.py make <arm> <add|addrm> <tau> <out> [pool|India|US] [gc70|hn90|r90]
                                                                     # (<out> = '-': France counts only, no files)
"""
import csv
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]
from fit import ARMS, MONO, capped, fit, score  # noqa: E402

W = ROOT / "data/work/p39"
MAIN = Path(os.path.realpath(ROOT / "data")).parent          # the main checkout (output/ lives there)
BASE = MAIN / "output/sub34v_rescue90"
T_RESCUE = 0.90
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def best_per(key, p, c):
    o = np.flatnonzero(c)
    o = o[np.lexsort((-p[o], key[o]))]
    m = np.zeros(len(p), bool)
    m[o[np.r_[True, key[o][1:] != key[o][:-1]]] if len(o) else o] = True
    return m


def fr_base(d):
    """#34v_r90's France decisions on the frame: argmax & p >= 0.9868, plus the empty-S1 rescue (best argmax,
    non-sibling pair with p >= 0.90 for an S1 with no member)."""
    s1, p, acc = d.s1.to_numpy(), d.p.to_numpy(), d.acc.to_numpy()
    sib = d.hn_rel.to_numpy() == 2
    has = np.bincount(s1[acc], minlength=int(s1.max()) + 1)[s1] > 0
    return acc | best_per(s1, p, ~sib & ~has & ~acc & (p >= T_RESCUE))


def fr_gc70(d):
    """sub37_gc70's France (LB 0.986304): #34v_r90 + P37 gc 'oneall' at p >= 0.70."""
    from ontop import gc
    return gc(d, fr_base(d), 0.70)


def fr_hn90(d):
    """sub37_gc70_hn90's France: gc70 + P37 'allhn' extras (non-empty S1, hn equal, p >= 0.90)."""
    from ontop import gc
    return gc(d, fr_gc70(d), 0.90, hn_extra=True)


BASES = {"r90": (fr_base, BASE), "gc70": (fr_gc70, MAIN / "output/sub37_gc70"),
         "hn90": (fr_hn90, MAIN / "output/sub37_gc70_hn90")}


def read_base(base=BASE):
    m = pd.read_csv(base / "matching_results.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    return m


def lists(d, pred, ids):
    s = pd.DataFrame({"s1": ids[d.s1.to_numpy()[pred]], "sx": d.sx.to_numpy()[pred]})
    return s.groupby("s1").sx.agg(lambda x: ",".join(sorted(x)))


def check(which="r90"):
    d = pd.read_parquet(W / "france.parquet")
    ids = pd.read_parquet(W / "s1_france.parquet").id.to_numpy()
    mine = lists(d, BASES[which][0](d), ids)
    m = read_base(BASES[which][1])
    fr = m[m.source1_entity_id.isin(set(ids))].set_index("source1_entity_id").matched_entity_ids
    fr = fr.map(lambda x: ",".join(sorted(x.split(","))) if x else "")
    mine = mine.reindex(fr.index).fillna("")
    diff = int((mine != fr).sum())
    print(f"France S1 {len(fr):,}; rows differing from {which}: {diff:,}; pairs mine {int((mine != '').sum()):,} S1 "
          f"non-empty vs {int((fr != '').sum()):,}")
    return diff


TRAIN = {"pool": ("India_US", "US_India"), "India": ("US_India",), "US": ("India_US",)}   # frames by label country


def make(arm, mode, tau, out, train="pool", base_name="gc70"):
    """train: whose LOCO labels fit the model (India = loco_US_India, the frame behind the passing India>US direction);
    base_name: r90 (#34v_r90), gc70 (sub37_gc70, LB best at 01:00) or hn90 (sub37_gc70_hn90, P37's next upload). The model adds (s >= tau) and, in addrm, drops base
    pairs with s < 0.3; caps never evict a base pair first."""
    t0 = time.time()
    feats = ARMS[arm]
    tr = pd.concat([pd.read_parquet(W / f"loco_{a}.parquet") for a in TRAIN[train]], ignore_index=True)
    mdl = fit(tr, feats, mono=MONO.get(arm))
    del tr
    d = pd.read_parquet(W / "france.parquet")
    ids = pd.read_parquet(W / "s1_france.parquet").id.to_numpy()
    s = score(mdl, d, feats)
    np.save(W / f"s_france_{arm}_{train}.npy", s.astype(np.float32))
    base = BASES[base_name][0](d)
    keep = (base | (s >= tau)) if mode == "add" else ((base & (s >= 0.3)) | (s >= tau))
    pred = capped(d.s1.to_numpy(), d.is2.to_numpy(), np.where(base, s + 1, s), keep)
    new = lists(d, pred, ids)
    add, rm = pred & ~base, base & ~pred
    info = {"arm": arm, "mode": mode, "tau": tau, "train": train, "base": base_name, "added": int(add.sum()), "removed": int(rm.sum()),
            "s1_changed": int(len(set(d.s1.to_numpy()[add | rm]))), "pairs": int(pred.sum()), "pairs_base": int(base.sum()),
            "empty": round(float(1 - len(new) / len(ids)), 4), "empty_base": round(float(1 - len(lists(d, base, ids)) / len(ids)), 4)}
    sh = {}
    s1c, is2 = d.s1.to_numpy(), d.is2.to_numpy()
    for k, m in (("base", base), ("new", pred)):
        n2, n3 = np.bincount(s1c[m & is2], minlength=len(ids)), np.bincount(s1c[m & ~is2], minlength=len(ids))
        sh[k] = {"both": round(float(((n2 > 0) & (n3 > 0)).mean()), 4), "s3only": round(float(((n2 == 0) & (n3 > 0)).mean()), 4),
                 "s2only": round(float(((n2 > 0) & (n3 == 0)).mean()), 4), "mps_all": round(float((n2 + n3).mean()), 3)}
    info["shape"] = sh
    if out == "-":                           # dry run: counts only, nothing written
        print(info)
        return
    od = MAIN / "output" / out
    od.mkdir(parents=True, exist_ok=True)
    src = BASES[base_name][1]
    # apply the model's deltas to the base file's own lists (the frame rebuild of gc70 differs on 2 France rows)
    e1, ex = ids[s1c], d.sx.to_numpy()
    plus = pd.Series(ex[add]).groupby(e1[add]).agg(set).to_dict()
    minus = pd.Series(ex[rm]).groupby(e1[rm]).agg(set).to_dict()
    frset = set(ids)
    n_fr = n_other = n_chg = 0
    with open(src / "matching_results.tsv", encoding="utf-8", newline="") as fi, \
            open(od / "matching_results.tsv", "w", encoding="utf-8", newline="") as fo:
        fo.write(fi.readline())
        for line in fi:
            k = line.split("\t", 1)[0]
            if k in frset:
                n_fr += 1
                if k in plus or k in minus:
                    cur = line.rstrip("\n").split("\t")[1]
                    lst = (set(cur.split(",")) if cur else set()) - minus.get(k, set()) | plus.get(k, set())
                    fo.write(f"{k}\t{','.join(sorted(lst))}\n")
                    n_chg += 1
                    continue
            else:
                n_other += 1
            fo.write(line)
    assert n_fr == len(ids), (n_fr, len(ids))
    cp = od / "candidate_pairs.tsv"
    if cp.exists():
        cp.unlink()
    os.link(src / "candidate_pairs.tsv", cp)
    info["rows_changed"] = n_chg
    info |= {"rows_france": n_fr, "rows_other": n_other, "runtime_s": round(time.time() - t0), "peak_gb": round(peak(), 2)}
    json.dump(info, open(od / "p39_build.json", "w"), indent=1)
    print(info)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "check":
        check(*a[1:])
    else:
        make(a[1], a[2], float(a[3]), a[4], *a[5:])
