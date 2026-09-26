"""Build output/<OUT> = output/<BASE> with France changed by the P48 arms (India / US lines byte-identical,
candidate_pairs.tsv and code.zip hard links: every France pair touched is a France chain pair, i.e. a candidate).

  veto V    remove France pairs (stages: default rescue+gc+hn) the current recipe (france.npz, P42 decide t .99 / r .9 / gc .7 / hn .8) added in the
            rescue / gc / hn stages that sit in the band (0.02 < g < 0.98), were not moved by a tier / bxneg rule
            (p0 = the OOD blend), and have CE r < V.  Tier pairs (P41 gib / P40 xf / P46 swap) are never touched.
  rescue U  (optional, needs data/work/p48/r_lo_France.parquet) add free argmax non-sibling g < 0.02 pairs with CE > U
            to one-sided / empty S1 per SCOPE, up to the caps.

    .venv/bin/python probes/P48-fr-ce-all/build48.py BASE OUT veto=0.2[,stages=rescue+gc][,rescue=0.9,scope=onesided]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT)]
import p42  # noqa: E402
import solution_p42 as S  # noqa: E402

CUR = {"tau": 0.1, "t": 0.99, "cap": [5, 6], "r": 0.9, "gc": 0.7, "hn": 0.8}


def veto_mask(z, st, v, stages=("rescue", "gc", "hn")):
    g, r, p0 = z["g"], z["r"], z["p0"]
    bl = 1 / (1 + np.exp(-(S.W_FR[0] * S.logit(g) + S.W_FR[1] * S.logit(np.nan_to_num(r, nan=0.5)) + S.W_FR[2])))
    prev = {"rescue": "base", "gc": "rescue", "hn": "gc"}
    added = np.any([st[s] & ~st[prev[s]] for s in stages], 0)
    return added & (g > S.LO) & (g < S.HI) & (np.abs(p0 - bl) <= 1e-4) & (np.nan_to_num(r, nan=1.0) < v)


def main(base, out, spec):
    t0 = time.time()
    cfg = dict(kv.split("=") for kv in spec.split(","))
    z = p42.fr_frame()
    st, p = p42.fr_decide(z, CUR)
    i1, ix = p42.fr_ids()
    stages = tuple(cfg.get("stages", "rescue+gc+hn").split("+"))
    rm = veto_mask(z, st, float(cfg["veto"]), stages) if "veto" in cfg else np.zeros(len(p), bool)
    add = np.zeros(len(p), bool)
    if "rescue" in cfg:
        import pandas as pd
        lo = pd.read_parquet(ROOT / "data/work/p48/r_lo_France.parquet")        # s1, sx (ids), r
        k = pd.DataFrame({"s1": i1[z["s1"]], "sx": ix[z["sx"]]}).reset_index().merge(lo, on=["s1", "sx"])
        rl = z["r"].copy()                                  # CE r where France already has it, the GPU job's elsewhere
        ki = k["index"].to_numpy()
        rl[ki] = np.where(np.isnan(rl[ki]), k.r.to_numpy(), rl[ki])
        pred = st["hn"] & ~rm
        s1, is2, n = z["s1"].astype(np.int64), z["is2"], int(z["n_s1"])
        _, free = S.base(s1, is2, p, S.argmax_sx(z["sx"].astype(np.int64), p), z["sib"], CUR["t"], tuple(CUR["cap"]))
        n2, n3 = S.counts(s1, is2, pred, n)
        m2, m3 = n2[s1], n3[s1]
        ok = {"empty": m2 + m3 == 0, "onesided": np.where(is2, (m2 == 0) & (m3 > 0), (m3 == 0) & (m2 > 0)),
              "any": np.ones(len(p), bool)}[cfg.get("scope", "onesided")]
        c = np.flatnonzero(free & ~pred & ok & (z["g"] < S.LO) & (np.nan_to_num(rl, nan=0.0) > float(cfg["rescue"])))
        if cfg.get("scope") == "empty":
            o, rk = S.rank_in(s1, rl, c)
            take = o[rk == 0]
        else:
            o, rk = S.rank_in(s1 * 2 + is2, rl, c)
            take = o[rk < np.where(is2[o], CUR["cap"][0] - m2[o], CUR["cap"][1] - m3[o])]
        add[take] = True
    minus, plus = {}, {}
    for a, b in zip(i1[z["s1"][rm]], ix[z["sx"][rm]]):
        minus.setdefault(a, set()).add(b)
    for a, b in zip(i1[z["s1"][add]], ix[z["sx"][add]]):
        plus.setdefault(a, set()).add(b)
    src, od = ROOT / "output" / base, ROOT / "output" / out
    od.mkdir(parents=True, exist_ok=True)
    fr = set(i1)
    n_fr = n_chg = n_rm = n_add = n_empty = 0
    with open(src / "matching_results.tsv", encoding="utf-8", newline="") as fi, \
            open(od / "matching_results.tsv", "w", encoding="utf-8", newline="") as fo:
        fo.write(fi.readline())
        for line in fi:
            k = line.split("\t", 1)[0]
            if k in fr:
                n_fr += 1
                if k in minus or k in plus:
                    cur = line.rstrip("\n").split("\t")[1]
                    cur = set(cur.split(",")) if cur else set()
                    new = (cur - minus.get(k, set())) | plus.get(k, set())
                    n_rm += len(cur - new)
                    n_add += len(new - cur)
                    n_empty += bool(cur) and not new
                    n_chg += new != cur
                    fo.write(f"{k}\t{','.join(sorted(new))}\n")
                    continue
            fo.write(line)
    assert n_fr == len(i1), (n_fr, len(i1))
    for name in ("candidate_pairs.tsv", "code.zip"):
        (od / name).unlink(missing_ok=True)
        os.link(src / name, od / name)
    info = {"base": base, "spec": spec, "veto_pairs": int(rm.sum()), "rescue_pairs": int(add.sum()), "removed_in_file": n_rm,
            "added_in_file": n_add, "s1_changed": n_chg, "s1_made_empty": n_empty,
            "veto_by_stage": {s: int((rm & st[s] & ~st[prev]).sum()) for prev, s in (("base", "rescue"), ("rescue", "gc"), ("gc", "hn"))},
            "runtime_s": round(time.time() - t0)}
    (od / "p48_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
