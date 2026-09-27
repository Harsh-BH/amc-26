"""Build output/<OUT> = output/<BASE> with France re-decided at recipe bars SPEC (P42 levers; P48's rescue/gc band veto
r < 0.2 re-applied to the new recipe's stages). The pair deltas new - current are applied to BASE's own France lists, so
tier pairs (P41 gib / P40 xf / P46 swap) and every non-France line stay byte-identical; candidate_pairs.tsv and
code.zip are hard links (every France recipe pair is a France chain pair, i.e. a candidate).

    .venv/bin/python probes/P52-fr-calib/build52.py BASE OUT t=0.99,r=0.97,gc=0.7,hn=0.8
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT / "probes/P48-fr-ce-all"), str(ROOT)]
import p42  # noqa: E402
from build48 import veto_mask  # noqa: E402

CUR = {"tau": 0.1, "t": 0.99, "cap": [5, 6], "r": 0.9, "gc": 0.7, "hn": 0.8}


def recipe(z, cfg):
    st, _ = p42.fr_decide(z, cfg)
    return st["hn"] & ~veto_mask(z, st, 0.2, ("rescue", "gc"))


def main(base, out, spec):
    t0 = time.time()
    cfg = dict(CUR)
    for kv in spec.split(","):
        k, v = kv.split("=")
        cfg[k] = None if v == "off" else float(v)
    z = p42.fr_frame()
    old, new = recipe(z, CUR), recipe(z, cfg)
    add, rm = new & ~old, old & ~new
    i1, ix = p42.fr_ids()
    plus, minus = {}, {}
    for d, m in ((plus, add), (minus, rm)):
        for a, b in zip(i1[z["s1"][m]], ix[z["sx"][m]]):
            d.setdefault(a, set()).add(b)
    src, od = ROOT / "output" / base, ROOT / "output" / out
    fr = set(i1)
    # guards against BASE's own non-recipe France pairs (tier / P47 class stages): an added SX already listed under
    # another France S1 is skipped (1-to-many), and adds beyond the caps (5 S2 / 6 S3) go lowest p first
    owner = {}
    base_fr = p42.read_fr(src)
    for k, v in base_fr.items():
        for x in (v.split(",") if v else ()):
            owner[x] = k
    pv = dict(zip(zip(i1[z["s1"][add]], ix[z["sx"][add]]), p42.fr_p(z, 0.1)[add]))
    n_owned = n_cap = 0
    for k in list(plus):
        cur = set(base_fr[k].split(",")) if base_fr[k] else set()
        keep = []
        for x in sorted(plus[k], key=lambda x: -pv[(k, x)]):
            if owner.get(x, k) != k:
                n_owned += 1
                continue
            keep.append(x)
        cnt = {"S2": sum(c.startswith("S2-") for c in cur - minus.get(k, set())), "S3": sum(c.startswith("S3-") for c in cur - minus.get(k, set()))}
        plus[k] = set()
        for x in keep:
            if x in cur:
                plus[k].add(x)
                continue
            src_ = x[:2]
            if cnt[src_] >= (5 if src_ == "S2" else 6):
                n_cap += 1
                continue
            cnt[src_] += 1
            plus[k].add(x)
    del base_fr, owner
    od.mkdir(parents=True, exist_ok=True)
    n_fr = n_chg = n_rm = n_add = n_empty = n_fill = 0
    with open(src / "matching_results.tsv", encoding="utf-8", newline="") as fi, \
            open(od / "matching_results.tsv", "w", encoding="utf-8", newline="") as fo:
        fo.write(fi.readline())
        for line in fi:
            k = line.split("\t", 1)[0]
            if k in fr:
                n_fr += 1
                if k in plus or k in minus:
                    cur = line.rstrip("\n").split("\t")[1]
                    cur = set(cur.split(",")) if cur else set()
                    new_l = (cur - minus.get(k, set())) | plus.get(k, set())
                    n_rm += len(cur - new_l)
                    n_add += len(new_l - cur)
                    n_empty += bool(cur) and not new_l
                    n_fill += (not cur) and bool(new_l)
                    n_chg += new_l != cur
                    fo.write(f"{k}\t{','.join(sorted(new_l))}\n" if new_l != cur else line)
                    continue
            fo.write(line)
    assert n_fr == len(i1), (n_fr, len(i1))
    for name in ("candidate_pairs.tsv", "code.zip"):
        (od / name).unlink(missing_ok=True)
        os.link(src / name, od / name)
    info = {"base": base, "spec": spec, "cfg": cfg, "recipe_add": int(add.sum()), "recipe_rm": int(rm.sum()),
            "added_in_file": n_add, "removed_in_file": n_rm, "s1_changed": n_chg, "s1_made_empty": n_empty,
            "s1_filled": n_fill, "skipped_owned": n_owned, "skipped_cap": n_cap, "runtime_s": round(time.time() - t0)}
    (od / "p52_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
