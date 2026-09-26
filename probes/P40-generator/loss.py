"""P40 step 2: generator op -> share of the #34 India/US holdout loss (out of fold, P36's state + error pairs).

Per error pair e of S1 s: w_e = L_s * g_e / sum(g), where L_s = 1 - F0.5_s and g_e = F0.5 gain of fixing e alone
(FN -> accepted, FP -> removed). Additive table: each e goes to ONE primary op, the op on it with the highest FN rate
among holdout true pairs (precedence learned from the data, printed). Oracle column: F0.5 gain of fixing every error
pair that carries the op (ops overlap, so oracles do not add up).
    probes/P40-generator/py.sh probes/P40-generator/loss.py tag      # tag error pairs -> data/work/p40/err_{c}.parquet
    probes/P40-generator/py.sh probes/P40-generator/loss.py table    # tables -> stdout + data/work/p40/loss_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "probes/P36-iu-residual")]
import ops as O  # noqa: E402
from cat import OUT, W, tag_pairs, world  # noqa: E402

LB_W = {"India": 0.468, "US": 0.383}
SIB3 = O.SIB3


def bits(m: np.ndarray) -> pd.DataFrame:
    m = m.astype(np.uint64)
    return pd.DataFrame({o: (m & np.uint64(O.BIT[o])) > 0 for o in O.OPS})


def tag_errors():
    """Tag P36's error pairs: FN against the true S1, FP against the S1 it was given to; for FP also the SX's
    true owner (full train truth) and the sibling offset vs the accepted S1."""
    s1, sx, t = world()
    gt = pd.read_parquet(W / "gt_pairs.parquet")
    owner = pd.Series(gt.s1.to_numpy(), index=gt.sx.to_numpy())
    world_owner = pd.Series(t.s1.to_numpy(), index=t.sx.to_numpy())
    import pyarrow as pa
    import pyarrow.parquet as pq
    need = set()
    for c in LB_W:
        need |= set(pd.read_parquet(W / f"p36/err_{c}.parquet", columns=["sx"]).sx) - set(sx.index)
    v = pa.array(sorted(need), pa.large_string())
    extra = [pq.read_table(W / f"train_source{i}.parquet", filters=[("entity_id", "in", v)]).to_pandas() for i in (2, 3)]
    sx = pd.concat([sx, pd.concat(extra).set_index("entity_id")])
    assert need <= set(sx.index)
    for c in LB_W:
        e = pd.read_parquet(W / f"p36/err_{c}.parquet")
        e["ops"] = tag_pairs(s1, sx, e.s1.to_numpy(), e.sx.to_numpy())
        e["owner"] = owner.reindex(e.sx).fillna("").to_numpy()
        e["owner_in_world"] = world_owner.reindex(e.sx).notna().to_numpy()
        e.to_parquet(OUT / f"err_{c}.parquet", index=False)
        print(c, len(e), e.type.value_counts().to_dict())


def pair_gain(tp, npred, nt, g, fn):
    """g_e for each error pair (vectorised)."""
    base = O_fb(tp[g], npred[g], nt[g])
    new = np.where(fn, O_fb(tp[g] + 1, npred[g] + 1, nt[g]), O_fb(tp[g], npred[g] - 1, nt[g]))
    return new - base


def O_fb(tp, npred, nt):
    tp, npred, nt = (np.asarray(x, float) for x in (tp, npred, nt))
    with np.errstate(divide="ignore", invalid="ignore"):
        P, R = tp / npred, tp / nt
        f = 1.25 * P * R / (0.25 * P + R)
    f = np.where(tp == 0, 0.0, f)
    return np.where(nt == 0, (npred == 0).astype(float), f)


def main():
    from base import state
    tp_all = pd.read_parquet(OUT / "tp.parquet")
    rows, per_pair = [], []
    for c in LB_W:
        s = state(c)
        pr, n, nt = s["pairs"], len(s["ids"]), s["n_true"]
        tp = np.bincount(pr.g[pr.acc & pr.y], minlength=n)
        npred = np.bincount(pr.g[pr.acc], minlength=n)
        e = pd.read_parquet(OUT / f"err_{c}.parquet")
        fn = ~e.type.isin(["fp_ns", "fp_s"]).to_numpy()
        g = e.g.to_numpy()
        ge = pair_gain(tp, npred, nt, g, fn)
        L = 1 - s["f34"]
        sg = np.bincount(g, weights=ge, minlength=n)
        k = np.bincount(g, minlength=n)
        e["w"] = np.where(sg[g] > 0, L[g] * ge / np.where(sg[g] > 0, sg[g], 1), L[g] / k[g])   # all-zero gains: equal split
        assert abs(e.w.sum() - L.sum()) < 1e-6, (e.w.sum(), L.sum())
        e["lb"] = e.w / n * LB_W[c]
        e["c"] = c
        e["fn"] = fn
        e["n"] = n
        # true-pair op rates for the S1 of this holdout (world truth), FN flag from the error file
        t = tp_all[tp_all.c == c]
        fnk = set(zip(e.s1[fn], e.sx[fn]))
        t = t.assign(err=[(a, b) in fnk for a, b in zip(t.s1, t.sx)])
        B = bits(t.ops.to_numpy())
        for o in O.OPS:
            m = B[o].to_numpy()
            rows.append(dict(c=c, op=o, n_true=int(m.sum()), rate=m.mean(), fn_rate=t.err.to_numpy()[m].mean() if m.any()
                             else np.nan))
        rows.append(dict(c=c, op="ALL", n_true=len(t), rate=1.0, fn_rate=t.err.mean()))
        per_pair.append(e)
    R = pd.DataFrame(rows)
    E = pd.concat(per_pair, ignore_index=True)
    R.to_csv(OUT / "op_rates.csv", index=False)
    E.to_parquet(OUT / "err_w.parquet", index=False)
    return R, E


def fp_kind(E: pd.DataFrame) -> np.ndarray:
    B = bits(E.ops.to_numpy())
    k = np.where(E.owner.to_numpy() == "", "orphan", np.where(E.owner_in_world.to_numpy(), "owned_in_world",
                                                                "owned_other_fold"))
    sib = B.A_HN_SIB.to_numpy()
    return np.where(sib, "sib_" + k, k)


def table():
    R, E = main()
    pooled = R.groupby("op").apply(lambda d: pd.Series(dict(n_true=d.n_true.sum(),
                                                             fn_rate=(d.fn_rate * d.n_true).sum() / max(d.n_true.sum(), 1))),
                                   include_groups=False)
    prec = pooled.drop("ALL").query("n_true >= 200").fn_rate.sort_values(ascending=False)
    B = bits(E.ops.to_numpy())
    prim = np.full(len(E), "NONE", object)
    for o in prec.index[::-1]:                            # highest FN rate written last = wins
        prim[B[o].to_numpy()] = o
    prim = np.where(E.fn, prim, "FP:" + fp_kind(E))
    E["primary"] = prim
    out = E.groupby(["primary", "c"]).agg(pairs=("lb", "size"), lb=("lb", "sum")).unstack("c").fillna(0)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["lb"] = out.lb_India + out.lb_US
    out = out.sort_values("lb", ascending=False)
    out["share"] = out.lb / out.lb.sum()
    out = out.join(prec.rename("fn_rate_true"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print("total LB-eq lost", round(out.lb.sum(), 6))
    print(out.round(6).to_string())
    out.to_csv(OUT / "loss_primary.csv")
    # oracle per op (errors carrying the op, both FN and FP pairs), per country
    orc = []
    for o in O.OPS:
        m = B[o].to_numpy()
        orc.append(dict(op=o, pairs=int(m.sum()), fn_pairs=int((m & E.fn).sum()), fp_pairs=int((m & ~E.fn).sum()),
                        lb_sum=E.lb[m].sum()))
    orc = pd.DataFrame(orc).set_index("op").join(pooled).sort_values("lb_sum", ascending=False)
    print(orc.round(6).to_string())
    orc.to_csv(OUT / "loss_oracle.csv")
    # by stage x primary
    st = E.pivot_table(index="primary", columns="type", values="lb", aggfunc="sum").fillna(0)
    st = st.loc[out.index]
    print((st * 1e5).round(1).to_string())
    return E


if __name__ == "__main__":
    {"tag": tag_errors, "table": table}[sys.argv[1]]()
