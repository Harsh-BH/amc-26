"""P42 local driver (the grid itself runs on SageMaker: solution_p42.py).

    .venv/bin/python probes/P42-fr-joint-tune/p42.py grid                 # France densities -> data/work/p42/sm/grid.json
    .venv/bin/python probes/P42-fr-joint-tune/p42.py repro                # LOCO: P39's ontop steps; France: sub37_gc70 rebuilt
    .venv/bin/python probes/P42-fr-joint-tune/p42.py local [units]        # a few grid units locally (timing / smoke)
    .venv/bin/python probes/P42-fr-joint-tune/p42.py build SPEC OUT       # SPEC tau=..,t=..,cap=5/6,r=..,gc=..,hn=..|off
    .venv/bin/python probes/P42-fr-joint-tune/p42.py stack TOP EXTRA OUT  # TOP's lists + EXTRA's adds over sub37_gc70
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]
import solution_p42 as S  # noqa: E402

W = ROOT / "data/work/p42"
SM = W / "sm"
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
LEVERS = {"tau": [0.0, 0.02, 0.05, 0.1, 0.2, 0.3],
          "t": [0.97, 0.98, 0.983, 0.985, 0.9868, 0.988, 0.99, 0.9925],
          "cap": [[5, 6], [4, 5], [3, 4]],
          "rescue": [None, 0.97, 0.95, 0.9, 0.8, 0.7, 0.5],
          "gc": [None, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
          "hn": [None, 0.97, 0.95, 0.9, 0.8, 0.7, 0.5]}
CURRENT = {"tau": 0.1, "t": 0.9868, "cap": [5, 6], "r": 0.9, "gc": 0.7, "hn": None}    # sub37_gc70, LB 0.986304
HN90 = dict(CURRENT, hn=0.9)                                                            # sub37_gc70_hn90 (staged)


def fr_frame():
    z = np.load(W / "france.npz")
    return {k: z[k] for k in z.files}


def fr_p(z, tau):
    """France p at veto tau. Exact for tau <= 0.1 (the rule-exempt pairs are known there: vetoable at 0.1 but kept in
    the 0.1 file). Above 0.1 the exemption of the newly vetoable pairs is a proxy (tier / bxneg changed p): it misses
    bxneg pairs with g >= 0.99, so it over-vetoes."""
    g, r = z["g"], np.nan_to_num(z["r"], nan=1.0)
    cond = lambda t: (g >= S.HI) & (r < t)
    ex = cond(0.1) & (z["p10"] > 0)
    if tau > 0.1:
        ex = ex | (cond(tau) & ~cond(0.1) & (z["p0"].astype(np.float32) != g.astype(np.float32)))
    return np.where(cond(tau) & ~ex, 0.0, z["p0"])


def grid():
    z = fr_frame()
    n, sx = int(z["n_s1"]), z["sx"].astype(np.int64)
    xs = sorted({x for k in ("t", "rescue", "gc", "hn") for x in LEVERS[k] if x is not None})
    dens = {}
    for tau in LEVERS["tau"]:
        p = fr_p(z, tau)
        am = S.argmax_sx(sx, p)
        dens[str(tau)] = {str(x): S.sdens(p, am, S.T_FR, x, n) for x in xs}
        print(f"tau {tau}: accepted/S1 at t {(am & (p >= S.T_FR)).sum() / n:.4f}; dens", {k: round(v, 4) for k, v in dens[str(tau)].items()}, flush=True)
    G = {"levers": LEVERS, "fr_dens": dens, "analogues": ["chain", "m2"], "directions": ["US_India", "India_US"], "current": CURRENT}
    SM.mkdir(parents=True, exist_ok=True)
    (SM / "grid.json").write_text(json.dumps(G))
    for dr in G["directions"]:
        (SM / f"loco_{dr}.npz").unlink(missing_ok=True)
        (SM / f"loco_{dr}.npz").hardlink_to(W / f"loco_{dr}.npz")
    print(f"{len(S.unit_list(G))} units x {len(S.cells(G)) // len(S.unit_list(G)) * len(G['analogues']) * len(G['directions'])} "
          f"= {len(S.cells(G)) * 4:,} evaluations ({len(S.cells(G)):,} cells x 2 analogues x 2 directions)")
    return G


def loco_steps(analogue, dr, G, cfg):
    z = S._frame(W, dr)
    s1, is2, y, n = z["s1"].astype(np.int64), z["is2"], z["y"], len(z["ntrue"])
    p = S.loco_p(z["g"], z["r"], cfg["tau"], analogue)
    am = np.ones(len(s1), bool)
    tm = S.fr_to_loco(G, analogue, dr, cfg["tau"], p, am, n)
    mp = lambda x: None if x is None else tm[str(x)]
    st = S.decide(s1, is2, p, am, z["hn_rel"] == 2, z["hn_rel"] == 1, n, mp(cfg["t"]), tuple(cfg["cap"]), mp(cfg["r"]),
                  mp(cfg["gc"]), mp(cfg["hn"]), steps=True)
    F = {k: S.fbeta(np.bincount(s1[v & y], minlength=n), np.bincount(s1[v], minlength=n), z["ntrue"].astype(np.int64)) for k, v in st.items()}
    return st, F, tm, y


def repro():
    """P39 ontop (chain analogue): rescue90 vs chain -0.0037 / +0.0010; gc70 vs rescue -0.0010 (prec 0.62) / +0.0022
    (0.87); hn90 extras -0.0018 (0.54) / +0.0018 (0.91). France: rebuild sub37_gc70's lists from france.npz."""
    G = json.loads((SM / "grid.json").read_text())
    out = {}
    for an in ("chain", "m2"):
        for dr in ("US_India", "India_US"):
            st, F, tm, y = loco_steps(an, dr, G, HN90)
            r = {"t_loco": round(tm["0.9868"], 4), "t_r": round(tm["0.9"], 4), "t_gc": round(tm["0.7"], 4),
                 "base_F": round(F["base"].mean(), 5)}
            for a, b in (("base", "rescue"), ("rescue", "gc"), ("gc", "hn")):
                add = st[b] & ~st[a]
                r[f"{b}_vs_{a}"] = (round(float((F[b] - F[a]).mean()), 5), int(add.sum()), round(float(y[add].mean()), 3) if add.any() else None)
            out[f"{an}:{dr}"] = r
            print(an, dr, r, flush=True)
    z = fr_frame()
    n = int(z["n_s1"])
    p = fr_p(z, 0.1)
    s1, sx, is2 = z["s1"].astype(np.int64), z["sx"].astype(np.int64), z["is2"]
    am = S.argmax_sx(sx, p)
    for name, cfg in (("sub37_gc70", CURRENT), ("sub37_gc70_hn90", HN90)):
        pred = S.decide(s1, is2, p, am, z["sib"], z["hn_eq"], n, cfg["t"], tuple(cfg["cap"]), cfg["r"], cfg["gc"], cfg["hn"])
        mine = fr_lists(z, pred)
        ref = read_fr(ROOT / "output" / name)
        diff = int((mine.reindex(ref.index).fillna("") != ref).sum())
        out[name] = {"rows_differ": diff, "pairs_mine": int(pred.sum()), "pairs_file": int(ref.str.len().gt(0).sum())}
        print(name, out[name], flush=True)
    (HERE / "repro.json").write_text(json.dumps(out, indent=1))
    print(f"peak {peak():.2f} GB")


_IDS = {}


def fr_ids():
    if "s1" not in _IDS:
        import pandas as pd
        sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
        import features as F
        _IDS["s1"] = pd.read_parquet(ROOT / "data/work/rf_test_s1_France.parquet", columns=["id"]).id.to_numpy()
        _IDS["sx"] = F.ids("test", "France")[1].to_numpy(zero_copy_only=False)
    return _IDS["s1"], _IDS["sx"]


def fr_lists(z, pred):
    import pandas as pd
    i1, ix = fr_ids()
    return pd.Series(ix[z["sx"][pred]]).groupby(i1[z["s1"][pred]]).agg(lambda x: ",".join(sorted(x)))


def read_fr(base):
    """France rows of a submission, member lists sorted (streamed: the whole file does not fit the memory cap)."""
    import pandas as pd
    i1, _ = fr_ids()
    fr, d = set(i1), {}
    with open(base / "matching_results.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            k, v = line.rstrip("\n").split("\t")
            if k in fr:
                d[k] = ",".join(sorted(v.split(","))) if v else ""
    return pd.Series(d)


def parse_cfg(spec):
    """'tau=0.1,t=0.99,cap=5/6,r=0.97,gc=0.7,hn=0.8' ('off' = stage off) -> a recipe dict (unset keys = CURRENT)."""
    c = dict(CURRENT)
    for kv in spec.split(","):
        k, v = kv.split("=")
        c[k] = [int(x) for x in v.split("/")] if k == "cap" else (None if v == "off" else float(v))
    return c


def fr_decide(z, cfg):
    p = fr_p(z, cfg["tau"])
    am = S.argmax_sx(z["sx"].astype(np.int64), p)
    st = S.decide(z["s1"].astype(np.int64), z["is2"], p, am, z["sib"], z["hn_eq"], int(z["n_s1"]), cfg["t"], tuple(cfg["cap"]),
                  cfg["r"], cfg["gc"], cfg["hn"], steps=True)
    return st, p


def build(spec, out, base="sub37_gc70"):
    """sub37_gc70 with France re-decided at recipe `spec`: the recipe's pair deltas vs the uploaded recipe (both from
    france.npz) are applied to the base file's own France lists; India / US lines byte-identical, candidate_pairs.tsv and
    code.zip hard links (every France pair is a candidate pair of the base)."""
    import os
    t0, cfg, z = time.time(), parse_cfg(spec), fr_frame()
    new, _ = fr_decide(z, cfg)
    ref, _ = fr_decide(z, CURRENT)
    pn, pr = new["hn"], ref["hn"]
    add, rm = pn & ~pr, pr & ~pn
    i1, ix = fr_ids()
    plus = {}
    for a, b in zip(i1[z["s1"][add]], ix[z["sx"][add]]):
        plus.setdefault(a, set()).add(b)
    minus = {}
    for a, b in zip(i1[z["s1"][rm]], ix[z["sx"][rm]]):
        minus.setdefault(a, set()).add(b)
    src, od = ROOT / "output" / base, ROOT / "output" / out
    od.mkdir(parents=True, exist_ok=True)
    fr = set(i1)
    n_fr = n_chg = n_miss = 0
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
                    n_miss += len(minus.get(k, set()) - cur) + len(plus.get(k, set()) & cur)
                    fo.write(f"{k}\t{','.join(sorted(cur - minus.get(k, set()) | plus.get(k, set())))}\n")
                    n_chg += 1
                    continue
            fo.write(line)
    assert n_fr == len(i1), (n_fr, len(i1))
    for name in ("candidate_pairs.tsv", "code.zip"):
        (od / name).unlink(missing_ok=True)
        os.link(src / name, od / name)
    n = int(z["n_s1"])
    s1, is2 = z["s1"].astype(np.int64), z["is2"]
    sh = {}
    for k, m in (("gc70", pr), ("new", pn)):
        n2, n3 = S.counts(s1, is2, m, n)
        sh[k] = {"both": round(float(((n2 > 0) & (n3 > 0)).mean()), 4), "s3only": round(float(((n2 == 0) & (n3 > 0)).mean()), 4),
                 "s2only": round(float(((n2 > 0) & (n3 == 0)).mean()), 4), "empty": round(float(((n2 + n3) == 0).mean()), 4),
                 "per_s1": round(float((n2 + n3).mean()), 4)}
    stages = {k: int((new[k] & ~new[j]).sum()) for j, k in (("base", "rescue"), ("rescue", "gc"), ("gc", "hn"))}
    info = {"spec": spec, "cfg": cfg, "base": base, "added": int(add.sum()), "removed": int(rm.sum()), "s1_changed": n_chg,
            "inconsistent_with_base_lists": n_miss, "accepted_at_t": int(new["base"].sum()), "stage_adds": stages, "shape": sh,
            "runtime_s": round(time.time() - t0), "peak_gb": round(peak(), 2)}
    (od / "p42_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    return info


def stack(top, extra, out, base="sub37_gc70"):
    """out = top's lists plus every pair `extra` adds over `base` (all three built line-for-line from base, so the files
    are zipped). Both add only SX-argmax pairs, so a shared SX is the same pair. Reports S1 over the 5 S2 / 6 S3 caps."""
    import os
    src = [ROOT / "output" / x / "matching_results.tsv" for x in (base, extra, top)]
    od = ROOT / "output" / out
    od.mkdir(parents=True, exist_ok=True)
    n_add = n_chg = n_over = 0
    fs = [open(x, encoding="utf-8", newline="") for x in src]
    with open(od / "matching_results.tsv", "w", encoding="utf-8", newline="") as fo:
        for lb, le, lt in zip(*fs):
            k = lt.split("\t", 1)[0]
            assert k == lb.split("\t", 1)[0] == le.split("\t", 1)[0]
            if le == lb:
                fo.write(lt)
                continue
            sp = lambda x: set(x.rstrip("\n").split("\t")[1].split(",")) - {""}
            plus = sp(le) - sp(lb)
            cur = sp(lt)
            new = cur | plus
            n_add += len(new - cur)
            n_chg += new != cur
            n_over += sum(x.startswith("S2-") for x in new) > 5 or sum(x.startswith("S3-") for x in new) > 6
            fo.write(f"{k}\t{','.join(sorted(new))}\n")
        assert all(next(x, None) is None for x in fs)
    for x in fs:
        x.close()
    for name in ("candidate_pairs.tsv", "code.zip"):
        (od / name).unlink(missing_ok=True)
        os.link(ROOT / "output" / base / name, od / name)
    info = {"top": top, "extra": extra, "base": base, "added_over_top": n_add, "s1_changed_vs_top": n_chg, "s1_over_cap": n_over}
    (od / "p42_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    return info


def local(k="2"):
    G = json.loads((SM / "grid.json").read_text())
    t0 = time.time()
    n = S.run_grid(SM, G | {"analogues": ["chain"]}, 0, len(S.unit_list(G)) // int(k), W / "local_cells.jsonl", procs=1)
    print(f"{n} cells in {time.time() - t0:.0f}s, peak {peak():.2f} GB")


if __name__ == "__main__":
    {"grid": grid, "repro": repro, "local": local, "build": build, "stack": stack}[sys.argv[1]](*sys.argv[2:])
