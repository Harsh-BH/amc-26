"""P76 step 1: which KINDS of S1 carry the India/US loss under sub54 (p54 >= 0.71, out of fold, P57 full holdout world).

Per segment value: S1 count/share, loss share (1 - F0.5), mean F0.5, FP, FN split into FN_below (true argmax pair with
p54 < t: a threshold can reach it) and FN_up (true SX not in the S1's argmax pairs: blocking / argmax / stolen), lift =
loss share - S1 share. Pooled rows weight each country by its LB share (0.468 India, 0.383 US).
    probes/P76-segment-hill/py.sh probes/P76-segment-hill/dive.py     -> data/work/p76/dive.json, dive_top.md,
                                                                        examples.md (raw rows: gitignored data/ only)
"""
import csv
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "probes/P76-segment-hill")
W = "data/work/p76"
T = 0.71
LB_W = {"India": 0.468, "US": 0.383}
FAM = ["lf", "org", "ntok", "twin", "hn", "unit", "floor", "state", "gsize", "n_am", "noaddr"]


def fb(tp, npred, nt):
    tp, npred, nt = (np.asarray(x, float) for x in (tp, npred, nt))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 1.25 * tp / (0.25 * nt + npred)
    f = np.where(tp == 0, 0.0, f)
    return np.where(nt == 0, np.where(npred == 0, 1.0, 0.0), f)


def load(c):
    F = pd.read_parquet(f"{W}/F_{c}.parquet")
    S = pd.read_parquet(f"{W}/S_{c}.parquet")
    n, g, y = len(S), F.g.to_numpy(), F.y.to_numpy()
    acc = F.p54.to_numpy() >= T
    tp, npred, nt = np.bincount(g[acc & y], minlength=n), np.bincount(g[acc], minlength=n), S.n_true.to_numpy()
    S["f"] = fb(tp, npred, nt)
    S["fp"] = npred - tp
    S["fn_below"] = np.bincount(g[~acc & y], minlength=n)
    S["fn_up"] = nt - np.bincount(g[y], minlength=n)
    S["npred"] = npred
    # label-free prediction-side segments
    S["gsize"] = pd.Series(np.minimum(npred, 5)).map(lambda k: "5+" if k == 5 else str(k)).to_numpy()
    nam = np.bincount(g, minlength=n)
    S["n_am"] = pd.Series(np.minimum(nam, 6)).map(lambda k: "6+" if k == 6 else str(k)).to_numpy()
    na = np.bincount(g[F.noaddr.to_numpy()], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = na / nam
    S["noaddr"] = np.select([nam == 0, na == 0, sh < 0.5], ["no_pairs", "0", "<.5"], ">=.5")
    return F, S


def table(S, c):
    L = 1 - S.f.to_numpy()
    tot, n = L.sum(), len(S)
    rows = []
    for fam in FAM:
        for v, ix in S.groupby(fam).indices.items():
            s = S.iloc[ix]
            rows.append(dict(country=c, fam=fam, val=v, n=len(ix), s1_share=len(ix) / n, loss_share=L[ix].sum() / tot,
                             f05=float(s.f.mean()), fp=int(s.fp.sum()), fn_below=int(s.fn_below.sum()),
                             fn_up=int(s.fn_up.sum()), s1_fp=int((s.fp > 0).sum()), s1_fn=int(((s.fn_below + s.fn_up) > 0).sum()),
                             loss=float(L[ix].sum())))
    t = pd.DataFrame(rows)
    t["lift"] = t.loss_share - t.s1_share
    t["ratio"] = t.loss_share / t.s1_share
    return t


def pooled(tabs, Ss):
    """LB-weighted pool: a segment value's loss = sum_c w_c * loss_c / n_c (the LB-eq loss it carries)."""
    tot = sum(LB_W[c] * (1 - Ss[c].f).sum() / len(Ss[c]) for c in Ss)
    t = pd.concat(tabs.values())
    t["lb_loss"] = [LB_W[c] * l / len(Ss[c]) for c, l in zip(t.country, t.loss)]
    t["lb_s1"] = [LB_W[c] * k / len(Ss[c]) for c, k in zip(t.country, t.n)]
    agg = t.groupby(["fam", "val"]).agg(n=("n", "sum"), lb_loss=("lb_loss", "sum"), lb_s1=("lb_s1", "sum"), fp=("fp", "sum"),
                                         fn_below=("fn_below", "sum"), fn_up=("fn_up", "sum")).reset_index()
    wt = sum(LB_W.values())
    agg["s1_share"], agg["loss_share"] = agg.lb_s1 / wt, agg.lb_loss / tot
    agg["lift"] = agg.loss_share - agg.s1_share
    return agg, tot


def sx_text(ids: set) -> pd.DataFrame:
    out = []
    for s in (2, 3):
        for ch in pd.read_csv(f"data/student_resource/dataset/train/train_source{s}.tsv", sep="\t", quoting=csv.QUOTE_NONE,
                              dtype=str, keep_default_na=False, chunksize=1_000_000):
            out.append(ch[ch.entity_id.isin(ids)])
    return pd.concat(out).set_index("entity_id")


def examples(D, top, k=2, seed=0):
    """2 FP + 2 FN_below + 2 FN_up examples per top segment (raw rows -> data/work/p76/examples.md only)."""
    import pyarrow.parquet as pq
    gt = pq.read_table("data/work/gt_pairs.parquet", columns=["s1", "sx"]).to_pandas()
    picks, need = [], set()
    rng = np.random.default_rng(seed)
    for _, r in top.iterrows():
        for c in (["India", "US"] if r.get("country") is None else [r.country]):
            F, S = D[c]
            m = S[r.fam].to_numpy() == r.val
            g = np.flatnonzero(m)
            acc = F.p54.to_numpy() >= T
            inseg = m[F.g.to_numpy()]
            for kind, rowm in (("FP", inseg & acc & ~F.y.to_numpy()), ("FN_below", inseg & ~acc & F.y.to_numpy())):
                ix = np.flatnonzero(rowm)
                for i in rng.choice(ix, min(k, len(ix)), replace=False) if len(ix) else []:
                    picks.append((r.fam, r.val, c, kind, S.id.iloc[F.g.iloc[i]], F.sx_id.iloc[i], float(F.p54.iloc[i])))
                    need.add(F.sx_id.iloc[i])
            up = g[S.fn_up.to_numpy()[g] > 0]
            for gi in rng.choice(up, min(k, len(up)), replace=False) if len(up) else []:
                s1 = S.id.iloc[gi]
                have = set(F.sx_id[F.g.to_numpy() == gi])
                miss = [x for x in gt.sx[gt.s1 == s1] if x not in have][:1]
                for x in miss:
                    picks.append((r.fam, r.val, c, "FN_up", s1, x, float("nan")))
                    need.add(x)
    del gt
    sx = sx_text(need)
    lines = ["# P76 raw error examples (gitignored: quotes Amazon rows)", ""]
    for fam, val, c, kind, s1, x, p in picks:
        S = D[c][1].set_index("id")
        r1 = S.loc[s1]
        rx = sx.loc[x] if x in sx.index else None
        lines.append(f"- **{fam}={val}** {c} {kind} p54={p:.3f} | S1 `{r1['name']}` / `{r1.addr}` | SX {x} "
                     f"`{rx.business_name if rx is not None else '?'}` / `{rx.business_address if rx is not None else '?'}`")
    return "\n".join(lines) + "\n"


def main():
    D = {c: load(c) for c in LB_W}
    Ss = {c: D[c][1] for c in D}
    tabs = {c: table(Ss[c], c) for c in D}
    agg, tot = pooled(tabs, Ss)
    res = {"pooled": agg.to_dict("records"), **{c: tabs[c].to_dict("records") for c in tabs},
           "base": {c: dict(f05=float(Ss[c].f.mean()), fp=int(Ss[c].fp.sum()), fn_below=int(Ss[c].fn_below.sum()),
                            fn_up=int(Ss[c].fn_up.sum()), n=len(Ss[c])) for c in Ss}, "lb_loss": float(tot)}
    json.dump(res, open(f"{W}/dive.json", "w"), indent=0, default=float)
    def fmt(t, cols):
        cell = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
        return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                         + ["| " + " | ".join(cell(v) for v in r) + " |" for r in t[cols].itertuples(index=False)])
    out = [f"base: " + ", ".join(f"{c} F0.5 {v['f05']:.5f} FP {v['fp']} FN_below {v['fn_below']} FN_up {v['fn_up']} "
                                 f"(n {v['n']})" for c, v in res["base"].items()), ""]
    big = agg[agg.n >= 500].sort_values("lift", ascending=False)
    out += ["## pooled (LB-weighted), n >= 500, top 15 by lift", fmt(big.head(15), ["fam", "val", "n", "s1_share",
            "loss_share", "lift", "fp", "fn_below", "fn_up"]), "", "## pooled, bottom 8 by lift (under-represented)",
            fmt(big.tail(8), ["fam", "val", "n", "s1_share", "loss_share", "lift", "fp", "fn_below", "fn_up"]), ""]
    for c in tabs:
        t = tabs[c][tabs[c].n >= 300].sort_values("lift", ascending=False)
        out += [f"## {c}, n >= 300, top 12 by lift", fmt(t.head(12), ["fam", "val", "n", "s1_share", "loss_share", "lift",
                "f05", "fp", "fn_below", "fn_up", "s1_fp", "s1_fn"]), ""]
    for fam in FAM:
        t = agg[agg.fam == fam].sort_values("n", ascending=False)
        out += [f"### family {fam} (pooled)", fmt(t, ["val", "n", "s1_share", "loss_share", "lift", "fp", "fn_below",
                                                      "fn_up"]), ""]
    open(f"{W}/dive_top.md", "w").write("\n".join(out))
    print("\n".join(out[:40]))
    top = pd.concat([tabs[c][tabs[c].n >= 300].sort_values("lift", ascending=False).head(6) for c in tabs])
    open(f"{W}/examples.md", "w").write(examples(D, top))


if __name__ == "__main__":
    main()
