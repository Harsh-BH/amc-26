"""Pareto sweep: candidates/S1 (holdout + test) vs final F0.5 on the #5 yardstick, for the learned pruner and the
simple alternatives (P3's cheaper config, a per-S1 top-k cap by block_score). One ledger row per variant.

    sweep.py [model=pruner.txt] [tag=] [only=<variant prefix>]     # needs prep.py hold + holdq <model> <tag>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]
from prep import OUT, XDIR, peak_gb, xfiles
from prune import load_model, score
from evalp import Bench

TEST_COLS = ["via_r", "block_rank_sx", "block_rank_s1", "addr_empty_x", "rule_acc"]


def sx_rel(g: np.ndarray, q: np.ndarray):
    """(is the SX's top-1 by q, q / the SX's max q)."""
    mx = pd.Series(q).groupby(g).transform("max").to_numpy()
    top = np.zeros(len(q), bool)
    idx = np.lexsort((-q, g))
    top[idx[np.r_[True, g[idx][1:] != g[idx][:-1]]]] = True
    return top, q / np.maximum(mx, 1e-9)


def variants(F: pd.DataFrame) -> dict:
    """name -> kept mask over the rows of F (cols: q, g, rule_acc, via_r, block_rank_sx, block_rank_s1, addr_empty_x)."""
    q, ra = F.q.to_numpy(), F.rule_acc.to_numpy() > 0
    rs1, rsx = F.block_rank_s1.to_numpy(), F.block_rank_sx.to_numpy()
    top, rel = sx_rel(F.g.to_numpy(), q)
    V = {"none": np.ones(len(F), bool)}
    for tau in (0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1):
        V[f"prune-tau{tau}"] = (q >= tau) | ra
    for tau in (0.005, 0.01, 0.02):
        V[f"prune-tau{tau}-noguard"] = q >= tau
    for tau in (0.005, 0.01, 0.02):
        V[f"prune-tau{tau}+sxtop1"] = (q >= tau) | ra | top
    for tau in (0.002, 0.005):
        for rho in (0.01, 0.03, 0.1):
            V[f"prune-tau{tau}-rel{rho}"] = ((q >= tau) & (top | (rel >= rho))) | ra
    V["p3-cheap-kr1-kf5"] = ((F.via_r.to_numpy() > 0) & ((rsx == 0) | (F.addr_empty_x.to_numpy() > 0))) | (rs1 < 5) | ra
    for k in (3, 4, 5, 6, 8):
        V[f"topk{k}-block_score"] = rs1 < k
        V[f"topk{k}-block_score+ra"] = (rs1 < k) | ra
    return V


def test_frames(m):
    """Per country: the variant columns for every test pair (X-file order), q recomputed with model m."""
    for c in ("France", "India", "US"):
        parts = []
        for ci, p in enumerate(xfiles("test", c)):
            X = pd.read_parquet(p, columns=sorted(set(m.feature_name()) | set(TEST_COLS) | {"qx"}))
            F = X[TEST_COLS].astype(np.float32)
            F["q"], F["g"] = score(m, X), X.qx.to_numpy(np.int64)
            parts.append(F)
            del X
        n1 = len(pd.read_parquet(ROOT / f"data/work/p3/txt_test/{c}_s1.parquet", columns=["entity_id"]))
        yield c, pd.concat(parts, ignore_index=True), n1


def main(model="pruner.txt", tag="", only=""):
    from ledger import log
    t0 = time.time()
    m = load_model(OUT / model)
    tc = {}                                         # test candidates per S1 by variant and country
    for c, F, n1 in test_frames(m):
        for k, v in variants(F).items():
            if only and not k.startswith(only) and k != "none":
                continue
            tc.setdefault(k, {})[c] = (int(v.sum()), n1)
        del F
    print(f"test counts done {time.time() - t0:.0f}s, peak {peak_gb()} GB", flush=True)
    b = Bench(cols=TEST_COLS)
    H = b.h
    F = H[TEST_COLS].copy()
    F["q"] = np.concatenate([pd.read_parquet(OUT / f"hold_q{tag}_{c}.parquet").q.to_numpy() for c in ("India", "US")])
    F["g"] = H.g.to_numpy()
    V = variants(F)
    del F
    base = b.baseline()
    rows = []
    for k, keep in V.items():
        if k not in tc:
            continue
        mt, _ = (base, None) if k == "none" else b.run(keep, name=k)
        n_all = sum(a for a, _ in tc[k].values()), sum(n for _, n in tc[k].values())
        mt = mt | {"test_cand_per_s1": n_all[0] / n_all[1]} | {f"test_cand_per_s1_{c}": a / n for c, (a, n) in tc[k].items()}
        rows.append({"variant": k} | mt)
        log("exp", probe="P10-candidates", subproblem="pruning", name=f"{k}{tag}", world="holdout-dense-half1+test",
            metrics={kk: round(float(v), 5) for kk, v in mt.items() if not isinstance(v, list)},
            config={"pruner": model, "w": mt.get("w"), "yardstick": "P4 big.py _blend_eval (w + D3 refit on half 0), "
                    "am recomputed over kept pairs"},
            hypothesis="pruning to <= 6 candidates/S1 loses <= 0.0005 F0.5 vs #5",
            verdict="inconclusive", runtime_s=round(time.time() - t0), peak_gb=peak_gb())
    pd.DataFrame(rows).to_csv(OUT / f"pareto{tag}.csv", index=False)
    print(f"done {time.time() - t0:.0f}s, peak {peak_gb()} GB", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
