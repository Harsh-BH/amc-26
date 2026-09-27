"""P59 India/US gate: sub44r's combiner (P44 FINAL + l_r_all, 5-seed bag, t 0.71) with the OLD r_all (ce_bge_pool.pt,
folds 4-9) vs the NEW r_all (P59 CE, folds 1-9), fit on one holdout half, reported on the other, both directions.
Paired per-S1 F0.5 difference new - old; LB-eq = E.LB_W (0.47 India + 0.38 US).

    eval59.py <r_new.parquet> [arms=new,newband,both] [zoo|nozoo]   # r_new: (s1, sx, r) for every holdout argmax pair
    zoo (default) = the shipped India/US combiner of #54 / sub_clean (P50 fullzoo: + P45's 16 zoo logits); nozoo = sub44r's
    eval59.py selftest                                   # r_new := the old p_r_all -> every paired diff must be 0
Arms (each vs 'old' = sub44r's columns):
  new      l_r_all from the new CE
  newband  + the band r columns a10 reads (r, lr, r0) from the new CE where the old band r exists
  both     old and new l_r_all side by side
"""
import json, os, sys, time
os.environ["P44_VARIANT"] = "r"
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P44-meta-stack"), str(ROOT / "probes/P50-idea-batch"), str(ROOT)]
import ev as E, meta as MT, te44  # noqa: E402
from best import bag  # noqa: E402

OUT = ROOT / "data/work/p59"


def zc():
    import zoo50 as Z
    return [f"z_{m}" for m in Z.MEM + Z.RANK]


def load(r_new, zoo=True):
    D = MT.prep(te44.MODELS)
    if zoo:
        import zoo50 as Z
        Z.attach(D)
    for c in E.C:
        A = D[c][0]
        if r_new == "selftest":
            A["p_r_new"] = A.p_r_all.to_numpy()
        else:
            r = pd.read_parquet(r_new).drop_duplicates(["s1", "sx"])
            A["p_r_new"] = A[["s1", "sx"]].merge(r, on=["s1", "sx"], how="left").r.to_numpy(np.float64)
        assert A.p_r_new.notna().all(), f"{c}: {A.p_r_new.isna().sum()} argmax pairs without new r"
        A["l_r_new"] = E.lg(A.p_r_new.to_numpy())
        band = A.r.notna().to_numpy()
        A["r_nb"] = np.where(band, A.p_r_new.to_numpy(), np.nan)          # new CE on the old band rows
        A["lr_nb"] = E.lg(A.r_nb.to_numpy())
        A["r0_nb"] = A.r_nb.fillna(0.0)
        cc = np.corrcoef(A.l_r_all.to_numpy(), A.l_r_new.to_numpy())[0, 1]
        print(f"{c}: {len(A):,} argmax pairs; logit corr old/new r_all {cc:.4f}; band rows {band.sum():,}", flush=True)
    return D


def cols(arm, zoo=True):
    return _cols(arm) + (zc() if zoo else [])


def _cols(arm):
    F = te44.FINAL
    if arm == "old":
        return F
    if arm == "new":
        return [x if x != "l_r_all" else "l_r_new" for x in F]
    if arm == "newband":
        sw = {"l_r_all": "l_r_new", "r": "r_nb", "lr": "lr_nb", "r0": "r0_nb"}
        return [sw.get(x, x) for x in F]
    if arm == "both":
        return F + ["l_r_new"]
    raise ValueError(arm)


def run(r_new, arms=("new", "newband", "both"), zoo=True):
    t0 = time.time()
    D = load(r_new, zoo)
    print("band cols swapped by newband:", [x for x in te44.FINAL if x in ("r", "lr", "r0")], flush=True)
    res = []
    for fit in (0, 1):
        rep = 1 - fit
        tr = {c: D[c][0].half.to_numpy() == fit for c in E.C}
        F = {a: None for a in ("old",) + tuple(arms)}
        for a in F:
            p = bag(cols(a, zoo))(tr, D)
            F[a] = {c: E.per_s1(D[c][0], D[c][1], p[c] >= E.T) for c in E.C}
        for a in arms:
            row = {"arm": a, "fit": fit, "rep": rep}
            for c in E.C:
                S = D[c][1]
                m = S.half.to_numpy() == rep
                d = (F[a][c] - F["old"][c])[m]
                v34 = (F[a][c] - S.f34.to_numpy())[m]
                o34 = (F["old"][c] - S.f34.to_numpy())[m]
                row[c] = (d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), int((d < 0).sum()))
                row[f"{c}_vs34"], row[f"{c}_old_vs34"] = v34.mean(), o34.mean()
            lb, se = E.lbeq(row)
            row["lbeq"], row["se"] = lb, se
            row["lbeq_vs34"] = sum(E.LB_W[c] * row[f"{c}_vs34"] for c in E.C)
            row["lbeq_old_vs34"] = sum(E.LB_W[c] * row[f"{c}_old_vs34"] for c in E.C)
            res.append(row)
            print(f"{a:8s} h{fit}->h{rep}: new-old LB-eq {lb:+.6f} ± {se:.6f} ({lb / se if se else 0:+.1f} SE) | "
                  + " | ".join(f"{c} {row[c][0]:+.6f} ± {row[c][1]:.6f} (+{row[c][2]}/-{row[c][3]})" for c in E.C)
                  + f" | vs #34: arm {row['lbeq_vs34']:+.6f}, old {row['lbeq_old_vs34']:+.6f}", flush=True)
    # cross-fit: each S1 reported once (h0 S1 from the h1 fit and vice versa) -> pooled mean / SE per arm
    summ = {}
    for a in arms:
        r0, r1 = [x for x in res if x["arm"] == a]
        cf = {}
        for c in E.C:
            n0 = int((D[c][1].half == r0["rep"]).sum()); n1 = int((D[c][1].half == r1["rep"]).sum())
            m = (r0[c][0] * n0 + r1[c][0] * n1) / (n0 + n1)
            se = np.sqrt((r0[c][1] * n0) ** 2 + (r1[c][1] * n1) ** 2) / (n0 + n1)
            cf[c] = (m, se)
        lb, se = E.lbeq(cf)
        summ[a] = {"crossfit_lbeq": lb, "crossfit_se": se, "h0h1": r0["lbeq"], "h0h1_se": r0["se"], "h1h0": r1["lbeq"],
                   "h1h0_se": r1["se"], **{f"{c}_cf": cf[c][0] for c in E.C},
                   "pass": bool(lb >= 1e-4 and lb >= 3 * se and r0["lbeq"] > 0 and r1["lbeq"] > 0)}
        print(f"{a:8s} CROSS-FIT new-old LB-eq {lb:+.6f} ± {se:.6f} ({lb / se if se else 0:+.1f} SE); halves "
              f"{r0['lbeq']:+.6f} / {r1['lbeq']:+.6f} -> {'PASS' if summ[a]['pass'] else 'fail'}", flush=True)
    print(f"{time.time() - t0:.0f}s", flush=True)
    return res, summ


if __name__ == "__main__":
    src = sys.argv[1]
    arms = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] not in ("zoo", "nozoo") else ("new", "newband", "both")
    zoo = "nozoo" not in sys.argv
    if src == "selftest":
        res, summ = run("selftest", ("new",), zoo="nozoo" not in sys.argv)
        assert all(abs(x[c][0]) < 1e-12 for x in res for c in E.C), "old == new must give a zero paired diff"
        print("selftest ok", flush=True)
    else:
        res, summ = run(src, arms, zoo)
        (OUT / f"eval59_{Path(src).stem}_{'zoo' if zoo else 'nozoo'}.json").write_text(json.dumps({"rows": res, "summary": summ}, default=float, indent=1))
