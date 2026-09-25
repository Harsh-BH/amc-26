"""Test deliverable at the chosen operating point: the pruned rows of #5's probability files.

    final.py [tau]      # -> data/work/p10/prob_test_{France,India,US}.parquet + a ledger row (profile vs #5)

Each output row keeps #5's p (India / US: CE-blended; France: m2_big) and p_gbdt; `am` / `sx_best_p` are recomputed
over the KEPT pairs on the GBDT p (the #5 file semantic -- submit_probs.py recomputes am on p anyway). France's
threshold is P6's label-free rule re-run on the pruned GBDT p (t* 0.69), as for #5.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "probes/P6-france")]
from prep import OUT, WORK, peak_gb
from prune import TAU, prune
import fr_diag as D                 # P6 (read-only)

TXT = WORK / "p3/txt_test"
T_STAR = 0.69                       # #5's India / US threshold
TS = np.round(np.arange(0.5, 0.99, 0.01), 2)


def ids(c):
    s1 = pq.read_table(TXT / f"{c}_s1.parquet", columns=["entity_id"]).column(0).combine_chunks()
    sx = pa.concat_arrays([pq.read_table(f, columns=["entity_id"]).column(0).combine_chunks()
                           for f in sorted(TXT.glob(f"{c}_sx_*.parquet"))])
    return s1, sx


def codes(t: pa.Table, col: str) -> np.ndarray:
    return t.column(col).combine_chunks().dictionary_encode().indices.to_numpy()


def answer(q1, qx, p, t):
    """submit_probs.py's rule: each SX to its argmax-p S1, kept if p >= t."""
    return D._argmax(q1, qx, p) & (p >= t)


def main(tau=TAU):
    from ledger import log
    tau, t0 = float(tau), time.time()
    m, E, keepd, prof = {}, {}, {}, {}
    for c in ("France", "India", "US"):
        q = pd.read_parquet(OUT / f"q_test_{c}.parquet")
        keep = prune(q, q.q.to_numpy(), tau)
        t = pq.read_table(WORK / f"sub5/prob_{c}.parquet")
        assert t.num_rows == len(q), (c, t.num_rows, len(q))
        a, b = ids(c)                                   # row alignment: X-file order == prob-file order
        chk = np.random.default_rng(0).choice(len(q), 20_000, replace=False)
        assert (a.take(pa.array(q.q1.to_numpy()[chk])).to_numpy(zero_copy_only=False)
                == t.column("s1").take(pa.array(chk)).to_numpy(zero_copy_only=False)).all()
        assert (b.take(pa.array(q.qx.to_numpy()[chk])).to_numpy(zero_copy_only=False)
                == t.column("sx").take(pa.array(chk)).to_numpy(zero_copy_only=False)).all()
        n1 = len(a)
        q1a, qxa, pa_ = codes(t, "s1"), codes(t, "sx"), t.column("p").to_numpy()
        del q, a, b
        kt = t.filter(pa.array(keep))
        del t
        g = kt.column("p_gbdt").to_numpy() if "p_gbdt" in kt.column_names else kt.column("p").to_numpy()
        q1, qx = q1a[keep], qxa[keep]
        am = D._argmax(q1, qx, g)
        best = pd.Series(g).groupby(qx).transform("max").to_numpy(np.float32)
        kt = kt.set_column(kt.column_names.index("am"), "am", pa.array(am))
        kt = kt.set_column(kt.column_names.index("sx_best_p"), "sx_best_p", pa.array(best))
        pq.write_table(kt, OUT / f"prob_test_{c}.parquet")
        E[c] = (D.empty_curve(q1, qx, g, n1, TS), n1)
        keepd[c] = (q1a, qxa, pa_, keep, q1, qx, kt.column("p").to_numpy(), n1)
        m |= {f"pairs_{c}": int(keep.sum()), f"cand_per_s1_{c}": round(keep.sum() / n1, 4),
              f"pairs_5_{c}": int(len(keep))}
        print(c, f"{keep.sum():,} of {len(keep):,} pairs kept ({keep.sum() / n1:.2f}/S1), {time.time() - t0:.0f}s, "
              f"peak {peak_gb()} GB", flush=True)
        del kt
    w = E["US"][1] / (E["US"][1] + E["India"][1])
    ref = w * E["US"][0].loc[T_STAR] + (1 - w) * E["India"][0].loc[T_STAR]
    t_fr = D.france_threshold(E["France"][0], ref, T_STAR)
    m |= {"t_france": t_fr, "ref_empty": round(float(ref), 4)}
    for c, (q1a, qxa, pa_, keep, q1, qx, p, n1) in keepd.items():
        t = t_fr if c == "France" else T_STAR
        s5 = answer(q1a, qxa, pa_, 0.78 if c == "France" else T_STAR)   # #5 as shipped (France 0.78)
        sp = np.zeros(len(keep), bool)
        sp[np.flatnonzero(keep)] = answer(q1, qx, p, t)
        k5, kp = np.bincount(q1a[s5], minlength=n1), np.bincount(q1a[sp], minlength=n1)
        diff = np.zeros(n1, bool)
        diff[q1a[s5 ^ sp]] = True
        m |= {f"matches_per_s1_{c}": round(float(kp.mean()), 4), f"s1_empty_{c}": round(float((kp == 0).mean()), 4),
              f"matches_per_s1_5_{c}": round(float(k5.mean()), 4), f"s1_empty_5_{c}": round(float((k5 == 0).mean()), 4),
              f"answer_lost_{c}": int((s5 & ~sp).sum()), f"answer_new_{c}": int((sp & ~s5).sum()),
              f"s1_changed_{c}": int(diff.sum()), f"s1_changed_share_{c}": round(float(diff.mean()), 5)}
    n = sum(v[7] for v in keepd.values())
    m["cand_per_s1"] = round(sum(m[f"pairs_{c}"] for c in keepd) / n, 4)
    print(m, flush=True)
    fr_ok = 0.045 <= m["s1_empty_France"] <= 0.075 and m["matches_per_s1_France"] >= 2.9
    log("exp", probe="P10-candidates", subproblem="pruning", name=f"test-pruned-tau{tau}", world="test", metrics=m,
        config={"tau": tau, "pruner": "data/work/p10/pruner.txt", "src": "data/work/sub5/prob_{c}.parquet",
                "decision": "submit_probs: argmax per SX on p, p >= t (India/US 0.69, France P6 rule on pruned GBDT p)"},
        hypothesis="the pruned test answers keep #5's profile; France stays in P6's bands (empty 4.5-7.5%, >= 2.9/S1)",
        verdict="keep" if fr_ok else "drop", runtime_s=round(time.time() - t0), peak_gb=peak_gb())


if __name__ == "__main__":
    main(*sys.argv[1:])
