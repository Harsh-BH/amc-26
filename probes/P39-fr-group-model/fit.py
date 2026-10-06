"""P39 LOCO gate: train the group-aware second stage on one LOCO frame, apply to the other, decode, score.

  "US>India" = train on loco_India_US (US labels, India-only first stage), apply to loco_US_India (India target,
  US-only first stage); "India>US" the reverse. Nothing is tuned on the target: the decision threshold is the
  F0.5-optimal one on the training frame's out-of-fold predictions (5 folds by S1).
Decode: argmax fixed (frame rows are SX-argmax pairs), accept s >= tau, caps 5 S2 / 6 S3 per S1 (highest s kept).
Modes: full = the model decides every pair; add = chain kept, the model only adds; addrm = add + remove.
Paired dF0.5 per target S1 vs the chain analogue (acc = p >= t), mean ± SE, and LB-eq = 0.1498 * dF (France share).

    .venv/bin/python probes/P39-fr-group-model/fit.py [arm ...]      # arms: see ARMS
"""
import json
import resource
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
W = ROOT / "data/work/p39"
FR_SHARE = 0.1498
CAP = {True: 5, False: 6}
P_ONLY = ["lt", "q"]
GROUP = ["rank_src", "n_acc_src", "n_acc_oth", "n_live_src", "n_live_oth", "best_oth_lt", "best_src_ex_lt", "is2"]
TEXT = ["hn_rel", "kind", "nm_ratio", "nm_tset", "city", "state_eq", "is_web", "addr_eq"]
GTEXT = ["grp_sim", "grp_hn"]
NAME = ["kind", "nm_ratio", "nm_tset", "city", "state_eq", "is_web"]           # TEXT without the hn columns
ARMS = {"p_only": P_ONLY, "p_group": P_ONLY + GROUP, "p_text": P_ONLY + TEXT, "full": P_ONLY + GROUP + TEXT + GTEXT,
        "full_noq": ["lt"] + GROUP + TEXT + GTEXT, "full_nolt": ["q"] + GROUP + TEXT + GTEXT,
        "p_group_gt": P_ONLY + GROUP + GTEXT, "p_group_nm": P_ONLY + GROUP + NAME,
        "full_nohn": P_ONLY + GROUP + NAME + ["grp_sim"], "p_group_mono": P_ONLY + GROUP}
MONO = {"p_group_mono": {"lt": 1, "q": 1, "best_oth_lt": 1}}      # arm -> LightGBM monotone constraints
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, seed=0)
ROUNDS = 400
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
half = lambda ids: (pd.util.hash_pandas_object(pd.Series(list(ids)), index=False).to_numpy() >> 7) % 2   # = run.half


def fb(tp, npred, ntrue):
    tp, npred, ntrue = (np.asarray(x, float) for x in (tp, npred, ntrue))
    with np.errstate(divide="ignore", invalid="ignore"):
        P, R = tp / npred, tp / ntrue
        f = 1.25 * P * R / (0.25 * P + R)
    f = np.where(tp == 0, 0.0, f)
    return np.where(ntrue == 0, np.where(npred == 0, 1.0, 0.0), f)


def load(src, tgt):
    d = pd.read_parquet(W / f"loco_{src}_{tgt}.parquet")
    s = pd.read_parquet(W / f"s1_{src}_{tgt}.parquet")
    return d, s.ntrue.to_numpy(), half(s.id)


def capped(s1, is2, s, keep):
    """keep, limited to the top-CAP by s per (S1, source)."""
    o = np.flatnonzero(keep)
    o = o[np.lexsort((-s[o], s1[o] * 2 + is2[o]))]
    k = s1[o] * 2 + is2[o]
    i = np.arange(len(o))
    rank = i - np.maximum.accumulate(np.where(np.r_[True, k[1:] != k[:-1]], i, 0))
    m = np.zeros(len(s), bool)
    m[o[rank < np.where(is2[o], CAP[True], CAP[False])]] = True
    return m


def decide(d, s, tau, mode="full", tau_rm=None):
    s1, is2, acc = d.s1.to_numpy(), d.is2.to_numpy(), d.acc.to_numpy()
    if mode == "full":
        keep = s >= tau
    elif mode == "add":
        keep = acc | (s >= tau)
    elif mode == "gc":                      # complete one-sided groups only: best missing-source pair per S1
        n = int(s1.max()) + 1
        n2, n3 = np.bincount(s1[acc & is2], minlength=n)[s1], np.bincount(s1[acc & ~is2], minlength=n)[s1]
        miss = np.where(is2, (n2 == 0) & (n3 > 0), (n3 == 0) & (n2 > 0))
        c = np.flatnonzero(miss & ~acc & (s >= tau))
        c = c[np.lexsort((-s[c], s1[c]))]
        keep = acc.copy()
        keep[c[np.r_[True, s1[c][1:] != s1[c][:-1]]] if len(c) else c] = True
    else:                                   # addrm: add at tau, remove chain pairs below tau_rm
        keep = (acc & (s >= tau_rm)) | (s >= tau)
    return capped(s1, is2, s, keep)


def per_s1(d, pred, ntrue):
    s1, y = d.s1.to_numpy(), d.y.to_numpy()
    n = len(ntrue)
    return fb(np.bincount(s1[pred & y], minlength=n), np.bincount(s1[pred], minlength=n), ntrue)


def shape(d, pred, n):
    s1, is2 = d.s1.to_numpy(), d.is2.to_numpy()
    n2, n3 = np.bincount(s1[pred & is2], minlength=n), np.bincount(s1[pred & ~is2], minlength=n)
    ne = (n2 + n3) > 0
    return {"empty": round(float((~ne).mean()), 4), "both": round(float(((n2 > 0) & (n3 > 0)).mean()), 4),
            "s3only": round(float(((n2 == 0) & (n3 > 0)).mean()), 4), "s2only": round(float(((n2 > 0) & (n3 == 0)).mean()), 4),
            "mps": round(float((n2 + n3)[ne].mean()), 3)}


def fit(d, feats, rounds=ROUNDS, mono=None):
    live = d.p.to_numpy() >= 0.02
    prm = PARAMS if not mono else PARAMS | {"monotone_constraints": [mono.get(f, 0) for f in feats]}
    return lgb.train(prm, lgb.Dataset(d.loc[live, feats].to_numpy(np.float32), d.y.to_numpy()[live]), rounds)


def score(m, d, feats):
    s = np.zeros(len(d))
    live = d.p.to_numpy() >= 0.02
    s[live] = m.predict(d.loc[live, feats].to_numpy(np.float32))
    return s


def oof(d, feats, k=5, mono=None):
    fo = (pd.util.hash_array(d.s1.to_numpy().astype(np.int64)) % k).astype(int)
    s = np.zeros(len(d))
    for i in range(k):
        m = fit(d[fo != i], feats, mono=mono)
        s[fo == i] = score(m, d[fo == i], feats)
    return s


TAUS = np.round(np.arange(0.30, 0.96, 0.01), 3)


def best_tau(d, s, ntrue, mode):
    fs = {t: per_s1(d, decide(d, s, t, mode, tau_rm=0.5 if mode == "addrm" else None), ntrue).mean() for t in TAUS}
    return max(fs, key=fs.get), fs


def run(direction, arms):
    t0 = time.time()
    tgt = direction.split(">")[1]
    src = direction.split(">")[0]
    dtr, nt_tr, _ = load(tgt, src)          # train frame: first stage trained on tgt, labels of src
    dte, nt_te, hv = load(src, tgt)         # apply frame: first stage trained on src, labels of tgt
    base_tr = per_s1(dtr, dtr.acc.to_numpy(), nt_tr)
    base = per_s1(dte, dte.acc.to_numpy(), nt_te)
    n = len(nt_te)
    res = {"base_train": round(float(base_tr.mean()), 5), "base": round(float(base.mean()), 5),
           "shape_base": shape(dte, dte.acc.to_numpy(), n)}
    ts = np.zeros(n, int)
    np.add.at(ts, dte.s1.to_numpy()[dte.y.to_numpy()], 1)
    res["frame_recall_ceiling"] = round(float((ts / np.maximum(nt_te, 1))[nt_te > 0].mean()), 4)
    print(f"== {direction}: train {len(dtr):,} rows (base F {base_tr.mean():.5f}), apply {len(dte):,} rows, "
          f"{n:,} S1, base F {base.mean():.5f}, shape {res['shape_base']}", flush=True)
    for arm in arms:
        feats = ARMS[arm]
        so = oof(dtr, feats, mono=MONO.get(arm))
        m = fit(dtr, feats, mono=MONO.get(arm))
        s = score(m, dte, feats)
        out = {}
        for mode in ("full", "add", "addrm"):
            tau, fs = best_tau(dtr, so, nt_tr, mode)
            pred = decide(dte, s, tau, mode, tau_rm=0.5 if mode == "addrm" else None)
            f = per_s1(dte, pred, nt_te)
            dl = f - base
            orc = max(TAUS, key=lambda t: per_s1(dte, decide(dte, s, t, mode, 0.5 if mode == "addrm" else None), nt_te).mean())
            fo_ = per_s1(dte, decide(dte, s, orc, mode, 0.5 if mode == "addrm" else None), nt_te) - base
            acc = dte.acc.to_numpy()
            add, rm = pred & ~acc, acc & ~pred
            y = dte.y.to_numpy()
            out[mode] = {"tau": float(tau), "train_gain": round(float(fs[tau] - base_tr.mean()), 5),
                         "gain": round(float(dl.mean()), 5), "se": round(float(dl.std(ddof=1) / np.sqrt(n)), 5),
                         "gain_h1": round(float(dl[hv == 1].mean()), 5), "lbeq": round(float(dl.mean() * FR_SHARE), 5),
                         "added": int(add.sum()), "added_prec": round(float(y[add].mean()), 3) if add.any() else None,
                         "removed": int(rm.sum()), "removed_prec": round(float(y[rm].mean()), 3) if rm.any() else None,
                         "oracle_tau": float(orc), "oracle_gain": round(float(fo_.mean()), 5), "shape": shape(dte, pred, n)}
            print(f"  {arm:<9} {mode:<5} tau {tau:.2f} (train +{out[mode]['train_gain']:.5f}) gain {dl.mean():+.5f} ± "
                  f"{out[mode]['se']:.5f} h1 {out[mode]['gain_h1']:+.5f} LB-eq {out[mode]['lbeq']:+.5f} | add "
                  f"{out[mode]['added']:,} @ {out[mode]['added_prec']} rm {out[mode]['removed']:,} @ {out[mode]['removed_prec']}"
                  f" | oracle tau {orc:.2f} {fo_.mean():+.5f} | {out[mode]['shape']}", flush=True)
        imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
        out["imp"] = {k: round(float(v / imp.sum()), 3) for k, v in imp.head(8).items()}
        print(f"  {arm} importance: {out['imp']}  ({time.time() - t0:.0f}s, peak {peak():.2f} GB)", flush=True)
        res[arm] = out
    return res


if __name__ == "__main__":
    arms = sys.argv[1:] or ["p_only", "full"]
    res = {dr: run(dr, arms) for dr in ("US>India", "India>US")}
    json.dump(res, open(W / f"fit_{'_'.join(arms)}.json", "w"), indent=1)
    from ledger import log
    for arm in arms:
        for mode in ("full", "add", "addrm"):
            m = {dr: {k: v for k, v in res[dr][arm][mode].items() if k != "shape"} for dr in res}
            ok = all(m[dr]["lbeq"] >= 0.0005 for dr in m)
            log("exp", probe="P39-fr-group-model", subproblem="france-group-model", name=f"loco-{arm}-{mode}",
                world="LOCO (P6 M2 + P30 LOCO bge blend + veto 0.1, fixed t; argmax fixed)",
                metrics={"US>India": m["US>India"], "India>US": m["India>US"],
                         "base": {dr: res[dr]["base"] for dr in res}},
                config={"feats": ARMS[arm], "params": PARAMS, "rounds": ROUNDS, "mode": mode, "tau": "train-frame OOF F0.5-opt",
                        "caps": "5 S2 / 6 S3"},
                hypothesis="a group-aware GBDT on transferable (rank / group / text) columns beats the France chain "
                           "analogue on the unseen country by >= 0.0005 LB-eq both ways",
                verdict="keep" if ok else "drop", runtime_s=0, peak_gb=round(peak(), 2))
