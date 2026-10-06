"""P67 France file: swap the final chain's P55 band veto (P50's p55n: ref & own & c < .6 & band, rescue stage kept) from
P55's combiner scores (cf_plain) to P67's (cf_<mode>, P55's features + the synthetic-tuned CE column), on a base whose
France lines already carry the old veto (output/sub66_final).
  restore  old-vetoed pairs the new combiner keeps (c_new >= .6) and that the chain really removed (in sub_iu54_fr38's
           France lines, the state just before p55n: 619 of the 3,044 listed pairs): put back when the SX is not listed for any France
           S1, the S1's caps 5 / 6 hold, P64's legal-form veto would not drop it, and it is a candidate of both
           <base>/candidate_pairs.tsv and <small>/candidate_pairs.tsv (the final zip's pruned set)
  remove   pairs the new combiner vetoes that the old did not (c_new < .6), wherever they still are in the lines
India / US lines byte-identical; candidate_pairs.tsv + code.zip hard links of the base.
    .venv/bin/python probes/P67-fr-synth/build67.py sub66_final sub67 p67bge sub66_final_small 0.95 noswap   # adds only
    .venv/bin/python probes/P67-fr-synth/build67.py qgate sub66_final sub67 sub67q                            # Qwen gate
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P55-fr-combiner"), str(ROOT / "code/business_entity_resolution/src")]
import p55  # noqa: E402

CAP = p55.CUR["cap"]


def cands(path, keys):
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            k, v = line.rstrip("\n").split("\t", 1)
            if k in keys:
                out[k] = set(v.split(",")) if v else set()
    return out


def main(base, out, mode, small=None, u=None, swap="swap"):
    import p42
    import france as FR
    t0 = time.time()
    z = p42.fr_frame()
    f = p55.france_frame()
    rows = f["rows"]
    own = p55.fr_own(z)[rows]
    st, rg = f["X"]["stage"].to_numpy(), f["X"]["region"].to_numpy()
    scope = f["ref"] & own & (rg == 1) & (st != 2)
    c_old = np.load(ROOT / "data/work/p55/cf_plain.npy")
    c_new = np.load(ROOT / f"data/work/p55/cf_{mode}.npy")
    old, new = scope & (c_old < 0.6), scope & (c_new < 0.6)
    i1, ix = p42.fr_ids()
    a1, ax = i1[z["s1"][rows]], ix[z["sx"][rows]]
    lines = p42.read_fr(ROOT / "output" / base)
    cur = {k: set(x.split(",")) if x else set() for k, x in lines.items()}
    del lines
    present = np.array([b in cur.get(a, ()) for a, b in zip(a1, ax)])
    info = {"base": base, "mode": mode, "old_veto": int(old.sum()), "new_veto": int(new.sum()), "both": int((old & new).sum()),
            "old_veto_present_in_base": int((old & present).sum())}
    if swap == "noswap":                               # adds only: keep the chain's p55n veto as it is
        new = old
    rm = np.flatnonzero(new & ~old & present)
    minus = {}
    for i in rm:
        minus.setdefault(a1[i], set()).add(ax[i])
    for k, s in minus.items():
        cur[k] = cur[k] - s
    # restores: best new c first
    fb = pd.read_parquet(ROOT / "data/work/p67/sm/ch/fb_France.parquet", columns=["s1", "ta", "tb"]).set_index("s1")
    pre = p42.read_fr(ROOT / "data/work/p67/fr38")            # sub_iu54_fr38 = the chain's France lines just before p55n
    pre = {k: set(x.split(",")) if x else set() for k, x in pre.items()}
    was = np.array([b in pre.get(a, ()) for a, b in zip(a1, ax)])
    del pre
    info["old_veto_removed_by_chain"] = int((old & was).sum())       # SUBMISSIONS #iu54_fr38_55n: 619
    cand = np.flatnonzero(old & ~new & ~present & was)
    assert np.isin(cand, fb.index).all()                       # every restore is a band row (P3 cleaned names)
    nm = lambda s: pd.Series(s, dtype="str").str.partition(" | ")[0].to_numpy() if len(s) else np.array([], object)
    n1, nx = nm(fb.ta.reindex(cand).to_numpy()), nm(fb.tb.reindex(cand).to_numpy())
    lf4 = FR.lf_class(FR.lf_sig(n1), FR.lf_sig(nx)) == 4
    del fb
    keys = set(a1[cand])
    cb = cands(ROOT / "output" / base / "candidate_pairs.tsv", keys)
    cs = cands(ROOT / "output" / small / "candidate_pairs.tsv", keys) if small else cb
    used = set().union(*cur.values())
    plus, skip = {}, {"lf": 0, "used": 0, "cap": 0, "notcand": 0}
    for j in np.argsort(-c_new[cand], kind="stable"):
        i = cand[j]
        a, b = a1[i], ax[i]
        s2 = sum(x.startswith("S2-") for x in cur[a])
        s3 = len(cur[a]) - s2
        why = "lf" if lf4[j] else "used" if b in used else "notcand" if b not in cb.get(a, ()) or b not in cs.get(a, ()) else \
              "cap" if (s2 >= CAP[0] if b.startswith("S2-") else s3 >= CAP[1]) else None
        if why:
            skip[why] += 1
            continue
        cur[a].add(b)
        used.add(b)
        plus.setdefault(a, set()).add(b)
    info |= {"removed": int(len(rm)), "restored": int(sum(map(len, plus.values()))), "restore_candidates": int(len(cand)),
             "restore_skipped": skip}
    if u is not None:                                  # P55's add arm with the new combiner (p55.build's rule + P64 + small)
        import importlib.util as iu
        sp = iu.spec_from_file_location("rr4b", ROOT / "probes/P4b-rerank/rerank.py")      # P4b's (src has a rerank.py too)
        rr = iu.module_from_spec(sp)
        sp.loader.exec_module(rr)
        sup = ~((rg == 2) & (f["p"] > 0))
        ca = np.flatnonzero(f["free"] & ~f["ref"] & own & sup & (c_new > float(u)) & ~present)
        ca = ca[np.argsort(-c_new[ca], kind="stable")]
        q1, qx = rr.rows("test", "France", pd.Series(a1[ca]), pd.Series(ax[ca]))
        FR.C = "France"                                # france.py keys its text tables on the unseen label
        n1, nx = FR.text_names(q1, qx)
        lf4a = FR.lf_class(FR.lf_sig(n1), FR.lf_sig(nx)) == 4
        keys = set(a1[ca])
        cb = cands(ROOT / "output" / base / "candidate_pairs.tsv", keys)
        cs = cands(ROOT / "output" / small / "candidate_pairs.tsv", keys) if small else cb
        sk2, n_add = {"lf": 0, "used": 0, "cap": 0, "notcand": 0}, 0
        for j, i in enumerate(ca):
            a, b = a1[i], ax[i]
            s2 = sum(x.startswith("S2-") for x in cur[a])
            why = "lf" if lf4a[j] else "used" if b in used else "notcand" if b not in cb.get(a, ()) or b not in cs.get(a, ()) else \
                  "cap" if (s2 >= CAP[0] if b.startswith("S2-") else len(cur[a]) - s2 >= CAP[1]) else None
            if why:
                sk2[why] += 1
                continue
            cur[a].add(b)
            used.add(b)
            plus.setdefault(a, set()).add(b)
            n_add += 1
        info |= {"u": float(u), "add_candidates": int(len(ca)), "added": n_add, "add_skipped": sk2}
    src, od = ROOT / "output" / base, ROOT / "output" / out
    od.mkdir(parents=True, exist_ok=True)
    chg = set(minus) | set(plus)
    n_fr = n_empty = 0
    with open(src / "matching_results.tsv", encoding="utf-8", newline="") as fi, \
            open(od / "matching_results.tsv", "w", encoding="utf-8", newline="") as fo:
        fo.write(fi.readline())
        for line in fi:
            k, v = line.rstrip("\n").split("\t", 1)
            if k in chg:
                n_empty += bool(v) and not cur[k]
                fo.write(f"{k}\t{','.join(sorted(cur[k]))}\n")
            else:
                fo.write(line)
            n_fr += k in cur
    assert n_fr == len(i1), (n_fr, len(i1))
    for name in ("candidate_pairs.tsv", "code.zip"):
        (od / name).unlink(missing_ok=True)
        os.link(src / name, od / name)
    info |= {"s1_changed": len(chg), "s1_made_empty": n_empty, "runtime_s": round(time.time() - t0)}
    (od / "p67_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)



def qgate(base, src, out):
    """output/<out> = output/<src> minus its France adds over output/<base> whose P38 France Qwen r < .5 (unscored adds
    kept). LOCO (qgate67.py): the dropped class is 64% / 49% precise; gating lifts the add arm +0.00005 / +0.00039."""
    import p42
    A, S = p42.read_fr(ROOT / "output" / base), p42.read_fr(ROOT / "output" / src)
    q = pd.read_parquet(ROOT / "data/work/p38/r/r_qwen_pool_test_France.parquet")
    q = dict(zip(zip(q.s1, q.sx), q.r))
    new, n_drop = {}, 0
    for k in S.index[S.to_numpy() != A.reindex(S.index).to_numpy()]:
        a, s = set(filter(None, A[k].split(","))), set(filter(None, S[k].split(",")))
        drop = {x for x in s - a if q.get((k, x), 1.0) < 0.5}
        n_drop += len(drop)
        new[k] = s - drop
    od = ROOT / "output" / out
    od.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "output" / src / "matching_results.tsv", encoding="utf-8", newline="") as fi, \
            open(od / "matching_results.tsv", "w", encoding="utf-8", newline="") as fo:
        fo.write(fi.readline())
        for line in fi:
            k = line.split("\t", 1)[0]
            fo.write(f"{k}\t{','.join(sorted(new[k]))}\n" if k in new else line)
    for name in ("candidate_pairs.tsv", "code.zip"):
        (od / name).unlink(missing_ok=True)
        os.link(ROOT / "output" / src / name, od / name)
    info = {"base": base, "src": src, "qwen_dropped": n_drop}
    (od / "p67_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    qgate(*sys.argv[2:]) if sys.argv[1] == "qgate" else main(*sys.argv[1:])
