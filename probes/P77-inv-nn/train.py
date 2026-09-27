"""P77 (from P57): neural set-matcher; inv=1 keeps only country-invariant inputs (France transfer, LOCO-gated) +
the S1's same-source structure (the generator gives a company ~one copy per source). Otherwise P57's docstring:
P57: neural set-matcher -- a residual transformer over each S1's candidate list, on top of #7's blended p.

Every candidate row is a token: M2's 72 features + M2 logit, CE r, blended logit, P17's house-number columns, SX-side
competition stats (build.py). L self-attention layers run across the S1's own candidates (the decision is per S1:
F0.5 rewards picking the right SUBSET), and a zero-initialised head adds a residual to the blended logit:
p' = sigmoid(logit p + f(set)). L=0 is the per-row MLP ablation (same inputs, no cross-candidate attention).

Fit on the holdout S1 of run.half 0 (both countries in one model; country is never a feature), early-stopped on a 10%
S1 slice of half 0. Report = half 1 against #7 at t 0.71 with P17's yardstick (argmax_sx over every row incl. the
competing train S1, per-S1 F0.5, blocking misses count), next to P17's hn stacker (arm sup) fit on the same half.

    .venv/bin/python probes/P57-neural/train.py [L=2] [d=128] [ep=12] [fit=India,US] [swap] [tag=...]
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
from scipy.special import ndtri
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
_s = importlib.util.spec_from_file_location("p17", ROOT / "probes/P17-hn-stack/stack.py")
P17 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(P17)
import model as M  # noqa: E402
from worlds import WORK  # noqa: E402

IN, T7, CS = WORK / "p57", 0.71, ("India", "US")   # P57's built inputs; results go there too
NOTX = ["q1", "qx", "p", "y", "pos"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, f: int, d: int, L: int):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(f, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True,
                                                                    norm_first=True), L, enable_nested_tensor=False) if L else None
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        nn.init.zeros_(self.out[1].weight), nn.init.zeros_(self.out[1].bias)   # starts as the base p

    def forward(self, x, pad):
        h = self.inp(x)
        if self.enc is not None:
            h = self.enc(h, src_key_padding_mask=pad)
        return self.out(h).squeeze(-1)


def batches(gid: np.ndarray, groups: np.ndarray, rows: int, shuffle: bool, rng=None) -> list:
    """Row-index matrices [g, K] (-1 = pad) over the chosen groups, similar sizes per batch, ~rows real rows each."""
    order = np.argsort(gid, kind="stable")
    start = np.searchsorted(gid[order], groups)
    size = np.searchsorted(gid[order], groups, side="right") - start
    o = np.argsort(size + (rng.random(len(size)) if shuffle else 0), kind="stable")   # bucket by size
    out, i = [], 0
    while i < len(o):
        k = int(size[o[i]])
        j = min(len(o), i + max(1, rows // k))
        sel = o[i:j]
        K = int(size[sel].max())
        m = np.full((len(sel), K), -1, np.int64)
        for a in range(K):
            ok = size[sel] > a
            m[ok, a] = order[start[sel][ok] + a]
        out.append(m)
        i = j
    if shuffle:
        rng.shuffle(out)
    return out


def run(net, X: torch.Tensor, lp: torch.Tensor, m: np.ndarray):
    idx = torch.from_numpy(m)
    pad = idx < 0
    xb = X[idx.clamp(min=0)].to(DEV, non_blocking=True).float()
    return net(xb, pad.to(DEV)), pad


ENC = ["lp", "cls", "d", "eq", "n_same", "p_same", "n_s1hn"]      # also kept raw: P17's stacker control
REF = np.linspace(0, 1, 1001)


TAG = ""                                                  # "_m2": France's regime (build.py base=m2), no CE r
INV = False                                               # inv=1: country-invariant inputs only
INVC = ["lp", "cls", "d", "eq", "n_same", "p_same", "n_s1hn", "c_sxcnt", "c_sxrk", "c_sxother", "c_sxgap", "is_s3",
        "s1_n", "s1_gap", "s1_margin", "sx_n", "sx_gap", "sx_margin", "block_rank_s1", "block_rank_sx", "blocked",
        "rule_acc", "g_src_n", "g_src_max"]
EXTRA = {}


def src_feats():
    """Per row: how many OTHER candidates of the S1 from the same source (S2 / S3) have p > 0.5, and their best logit."""
    q = pd.DataFrame({"g": np.concatenate([(ci << 32) | pq.read_table(IN / f"holdout{TAG}_{c}.parquet", columns=["q1"])
                                           .column(0).to_numpy().astype(np.int64) for ci, c in enumerate(CS)]),
                      "s": col("is_s3"), "lp": col("lp")})
    q["hi"] = (q.lp > 0).astype(np.float32)
    g = q.groupby(["g", "s"])
    n = g.hi.transform("sum").to_numpy() - q.hi.to_numpy()
    top = g.lp.transform("max").to_numpy()
    q["r"] = g.lp.rank(method="first", ascending=False).to_numpy()
    sec = q[q.r == 2].set_index(["g", "s"]).lp
    s2 = sec.reindex(pd.MultiIndex.from_arrays([q.g, q.s])).to_numpy()
    EXTRA["g_src_n"], EXTRA["g_src_max"] = n.astype(np.float32), np.where(q.lp.to_numpy() >= top, s2, top).astype(np.float32)


def col(name: str) -> np.ndarray:
    if name in EXTRA:
        return EXTRA[name]
    return np.concatenate([pq.read_table(IN / f"holdout{TAG}_{c}.parquet", columns=[name]).column(0)
                           .to_numpy(zero_copy_only=False) for c in CS])


def rank_gauss(v: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """sklearn QuantileTransformer(normal)'s map for one column: ties averaged both ways, then the normal quantile."""
    u = 0.5 * (np.interp(v, qs, REF) - np.interp(-v, -qs[::-1], -REF[::-1]))
    return np.clip(ndtri(np.clip(u, 1e-7, 1 - 1e-7)), -5.2, 5.2)


def load(fit, fh):
    """Inputs as ONE preallocated float16 matrix, filled column by column (RAM ~ rows x inputs x 2 B), + meta arrays."""
    meta, M_ = {}, {k: col(k) for k in NOTX + ENC}
    gid, hvr, fr = [], [], []
    for ci, c in enumerate(CS):
        q1 = pq.read_table(IN / f"holdout{TAG}_{c}.parquet", columns=["q1", "pos"])
        mz = np.load(IN / f"holdout{TAG}_{c}_meta.npz")
        hv = mz["half"].astype(np.int64)
        meta[c] = (mz["ntrue"], hv)
        pos = q1.column("pos").to_numpy()
        gid.append((ci << 32) | q1.column("q1").to_numpy().astype(np.int64))
        hvr.append(np.where(pos >= 0, hv[np.maximum(pos, 0)], -1))
        fr.append(np.full(len(pos), c in fit))
    gid, hvr = np.concatenate(gid), np.concatenate(hvr)
    tr = (hvr == fh) & np.concatenate(fr)
    names = [n for n in pq.read_schema(IN / f"holdout{TAG}_{CS[0]}.parquet").names if n not in NOTX + ["r"] * bool(TAG)]
    if INV:
        src_feats()
        names = [n for n in names if n in INVC] + ["g_src_n", "g_src_max"]
    hasnan = [n for n in names if np.isnan(col(n).astype(np.float32)).any()]
    X = np.empty((len(gid), len(names) + len(hasnan) + 5), np.float16)
    enc = {}
    for j, n in enumerate(names):
        v = col(n).astype(np.float32)
        m = np.isnan(v)
        med = float(np.nanmedian(v[tr]))
        v[m] = med
        qs = np.quantile(v[tr][::2], REF).astype(np.float64)
        X[:, j] = rank_gauss(v, qs)
        enc[n] = (med, qs)
        if n in hasnan:
            X[:, len(names) + hasnan.index(n)] = m
    X[:, -5:] = np.eye(5, dtype=np.float16)[np.nan_to_num(M_["cls"]).astype(np.int64).clip(0, 4)]
    return torch.from_numpy(X), M_, gid, hvr, tr, meta, {"names": names, "hasnan": hasnan, "enc": enc}


def main(L=2, d=128, ep=12, fit=CS, fh=0, tag="", seed=0):
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X, x, gid, hvr, tr, meta, spec = load(fit, fh)
    lp, y = torch.from_numpy(x["lp"].astype(np.float32)), torch.from_numpy(x["y"].astype(np.float32))
    g_tr = np.unique(gid[tr])
    va = (g_tr * 2654435761 % 1000) < 100                 # 10% of the fit S1 for early stopping
    g_fit, g_va = g_tr[~va], g_tr[va]
    print(f"{len(gid):,} rows, {X.shape[1]} inputs, fit groups {len(g_fit):,}, val {len(g_va):,}, L={L} d={d}, "
          f"{time.time() - t0:.0f}s, peak {P17.peak():.2f} GB", flush=True)

    net = Net(X.shape[1], d, L).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    vb = batches(gid, g_va, 8192, False)
    steps = ep * len(batches(gid, g_fit, 4096, True, rng))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)

    def val_loss():
        net.eval()
        tot = n = 0.0
        with torch.no_grad():
            for m in vb:
                o, pad = run(net, X, lp, m)
                ok = ~pad.to(DEV)
                i = torch.from_numpy(m).clamp(min=0).to(DEV)
                z = lp.to(DEV)[i] + o
                tot += bce(z[ok], y.to(DEV)[i][ok]).item()
                n += ok.sum().item()
        return tot / n

    base_vl = val_loss()
    best, state, lpd, yd = base_vl, None, lp.to(DEV), y.to(DEV)
    for e in range(ep):
        net.train()
        tl = n = 0.0
        for m in batches(gid, g_fit, 4096, True, rng):
            o, pad = run(net, X, lp, m)
            ok = ~pad.to(DEV)
            i = torch.from_numpy(m).clamp(min=0).to(DEV)
            loss = bce((lpd[i] + o)[ok], yd[i][ok])
            opt.zero_grad(set_to_none=True)
            (loss / ok.sum()).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(), sch.step()
            tl, n = tl + loss.item(), n + ok.sum().item()
        vl = val_loss()
        if vl < best:
            best, state = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"ep {e}: train {tl / n:.5f} val {vl:.5f} (base {base_vl:.5f}) {time.time() - t0:.0f}s", flush=True)
    if state is not None:
        net.load_state_dict(state)

    net.eval()
    res_all = np.zeros(len(gid), np.float32)
    with torch.no_grad():
        for m in batches(gid, np.unique(gid), 16384, False):
            o, pad = run(net, X, lp, m)
            ok = ~pad
            res_all[m[ok.numpy()]] = o.cpu().numpy()[ok.numpy()]
    s_all = (1 / (1 + np.exp(-(x["lp"].astype(np.float64) + res_all)))).astype(np.float32)

    res, base, new, stk = {"L": L, "d": d, "ep": ep, "fit": list(fit), "fh": fh, "best_val": round(best, 5),
                           "base_val": round(base_vl, 5)}, [], [], []
    for ci, c in enumerate(CS):
        mc = (gid >> 32) == ci
        ntrue, hv = meta[c]
        pos, yy, qx, p, s = x["pos"][mc], x["y"][mc], x["qx"][mc], x["p"][mc], s_all[mc]
        h0, h1 = hv == fh, hv == 1 - fh
        a7, am = M.argmax_sx(qx, p), M.argmax_sx(qx, s)
        fb = P17.f05(a7, p, T7, pos, yy, ntrue)
        tn = max(P17.TS, key=lambda t_: P17.f05(am, s, t_, pos, yy, ntrue)[h0].mean())
        fn, fn7 = P17.f05(am, s, tn, pos, yy, ntrue), P17.f05(am, s, T7, pos, yy, ntrue)
        # control: P17's GBDT stacker (arm sup) fit on the same half of this country
        Xs = pd.DataFrame({k: x[k][mc] for k in ENC})
        trs = (pos >= 0) & (hv[np.maximum(pos, 0)] == fh)
        ss = P17.stacked(P17.fit(Xs, yy, trs), Xs)
        ams = M.argmax_sx(qx, ss)
        ts = max(P17.TS, key=lambda t_: P17.f05(ams, ss, t_, pos, yy, ntrue)[h0].mean())
        fs = P17.f05(ams, ss, ts, pos, yy, ntrue)
        dn, d7, ds = (fn - fb)[h1], (fn7 - fb)[h1], (fs - fb)[h1]
        # label-free t (France's rule): the t whose empty-S1 rate matches the base's at T7; all S1 when c is unseen
        emp = lambda a_, s_, t_: 1 - len(np.unique(pos[a_ & (s_ >= t_) & (pos >= 0)])) / len(ntrue)
        eb = emp(a7, p, T7)
        tf = min(P17.TS, key=lambda t_: abs(emp(am, s, t_) - eb))
        ev = h1 if c in fit else np.ones(len(ntrue), bool)
        dfree = (P17.f05(am, s, tf, pos, yy, ntrue) - fb)[ev]
        se = lambda v: round(float(v.std() / np.sqrt(len(v))), 5)
        res[c] = {"f05_7_h1": round(float(fb[h1].mean()), 5), "f05_nn_h1": round(float(fn[h1].mean()), 5),
                  "t_nn": float(tn), "d_nn": round(float(dn.mean()), 5), "se_nn": se(dn),
                  "d_nn_t071": round(float(d7.mean()), 5), "t_stack": float(ts), "d_stack": round(float(ds.mean()), 5),
                  "se_stack": se(ds), "d_nn_vs_stack": round(float((fn - fs)[h1].mean()), 5), "se_nn_vs_stack": se((fn - fs)[h1]),
                  "up": int((dn > 0).sum()), "down": int((dn < 0).sum()), "t_free": float(tf),
                  "d_nn_free": round(float(dfree.mean()), 5), "se_nn_free": se(dfree), "free_on": "half1" if c in fit else "all"}
        print(c, res[c], flush=True)
        base.append(fb[h1]), new.append(fn[h1]), stk.append(fs[h1])
    fb, fn, fs = (np.concatenate(z) for z in (base, new, stk))
    se = lambda v: round(float(v.std() / np.sqrt(len(v))), 5)
    res["all"] = {"f05_7_h1": round(float(fb.mean()), 5), "f05_nn_h1": round(float(fn.mean()), 5),
                  "d_nn": round(float((fn - fb).mean()), 5), "se_nn": se(fn - fb),
                  "d_stack": round(float((fs - fb).mean()), 5), "se_stack": se(fs - fb),
                  "d_nn_vs_stack": round(float((fn - fs).mean()), 5), "se_nn_vs_stack": se(fn - fs)}
    res["runtime_s"] = round(time.time() - t0)
    print("India+US", res["all"], f"{res['runtime_s']}s", flush=True)
    name = tag or f"L{L}-d{d}-fit{'+'.join(fit)}{'-swap' if fh else ''}{TAG}{'-inv' if INV else ''}"
    (IN / f"res_{name}.json").write_text(json.dumps(res, indent=1))
    torch.save({"state": net.state_dict(), **spec, "L": L, "d": d},
               IN / f"net_{name}.pt")
    from ledger import log
    log("exp", probe="P77-inv-nn", subproblem="matcher", name=f"nn-{name}",
        world="holdout-dense half1 (#7 pruned list, P17 yardstick)" if not fh else "holdout-dense half0 (swap)",
        metrics={k: v for c in ("all", *CS) for k, v in ((f"{c}_{kk}", vv) for kk, vv in res[c].items())},
        config={"L": L, "d": d, "ep": ep, "fit": list(fit), "fh": fh, "inputs": int(X.shape[1])},
        hypothesis="a transformer over each S1's candidate set finds cross-candidate patterns the per-pair GBDT "
                   "and the hand-built hn stacker miss",
        verdict="keep" if res["all"]["d_nn_vs_stack"] > 2 * res["all"]["se_nn_vs_stack"] else "inconclusive",
        runtime_s=res["runtime_s"], peak_gb=round(P17.peak(), 2))
    return res


if __name__ == "__main__":
    kv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    TAG = "_m2" if kv.get("base") == "m2" else ""
    INV = kv.get("inv") == "1"
    main(L=int(kv.get("L", 2)), d=int(kv.get("d", 128)), ep=int(kv.get("ep", 12)),
         fit=tuple(kv.get("fit", "India,US").split(",")), fh=int("swap" in sys.argv[1:]), tag=kv.get("tag", ""))
