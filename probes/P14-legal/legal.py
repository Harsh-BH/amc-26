"""P14-legal: does canonicalising legal forms ("Private Limited" = "Pvt. Ltd." = "Pirvate Limted" -> pvt ltd) help the
M2 matcher? Adds 7 columns after the 65 cached features and never changes those:
  legal_n1 / legal_nx          canonical legal tokens on the S1 / S2-S3 name (0 = none)
  legal_eq / legal_diff        both names carry a legal form, and the canonical forms agree / differ
  legal_jacc                   token overlap of the two canonical forms (NaN unless both carry one)
  cname_ratio / cname_tset     rapidfuzz ratio / token-set ratio on the names with every legal token canonicalised
Legal tokens are the ones normalize.py already strips from name_core (N._is_legal: its LEGAL set + typo match).
Names come from P3's cleaned texts (data/work/p3/txt, q1 / qx row order). No labels are used.

A/B: the M2 recipe (model.PARAMS, early stop on the fold-1 25% sample) fitted on fold 2, 65 vs 72 features on the
same rows. Scored like run.py --split holdout: every pair of each S2/S3 that a fold-0 S1 competes for, argmax over
all its train S1, one threshold tuned on holdout half 0, macro F0.5 on half 1 (+ a paired SE).

    .venv/bin/python probes/P14-legal/legal.py check     # canonicaliser self-check
    .venv/bin/python probes/P14-legal/legal.py run       # needs build.py's X files; ~1 h -> data/work/p14, ledger
"""
import json
import sys
import time
from functools import lru_cache

sys.path[:0] = ["code/business_entity_resolution/src", "."]

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import Indel
from rapidfuzz.process import cpdist

import block as B
import features as F
import model as M
import normalize as N
from worlds import WORK

OUT = WORK / "p14"
ID = ["q1", "qx", "is_true", "fold"]
EXTRA = ["legal_n1", "legal_nx", "legal_eq", "legal_diff", "legal_jacc", "cname_ratio", "cname_tset"]
CANON = {"private": "pvt", "pvt": "pvt", "prvt": "pvt", "limited": "ltd", "ltd": "ltd", "incorporated": "inc",
         "inc": "inc", "corporation": "corp", "corp": "corp", "company": "co", "co": "co",
         "etablissement": "ets", "etablissements": "ets"}
LONG = {"private": "pvt", "limited": "ltd", "incorporated": "inc", "corporation": "corp", "company": "co"}
CODES = sorted(set(CANON.values()) | {CANON.get(t, t) for t in N.LEGAL})
BIT = {c: 1 << i for i, c in enumerate(CODES)}
assert len(CODES) < 63
def code(t: str, nxt: str) -> str:
    """Canonical code of token t (nxt = the token after it); '' when t is not a legal form. Mirrors N._is_legal's
    cheap tests first, so only long p/l/i/c words reach the (cached) typo match."""
    if t in N.LEGAL:
        return CANON.get(t, t)
    if t == "p":                                           # "P. Ltd."
        return "pvt" if nxt in ("ltd", "limited") else ""
    if len(t) < 5 or t[0] not in "plic" or t.isdigit():
        return ""
    return _typo(t, nxt in ("ltd", "limited"))


@lru_cache(maxsize=1 << 20)
def _typo(t: str, before_ltd: bool) -> str:
    if not N._is_legal(t, "ltd" if before_ltd else ""):
        return ""
    return LONG[max(LONG, key=lambda w: Indel.normalized_similarity(t, w))]   # its closest long form


def canon(names) -> tuple:
    """Cleaned names -> (bitmask of canonical legal codes, name with each legal token canonicalised)."""
    ms, cs = np.zeros(len(names), np.int64), np.empty(len(names), object)
    for k, s in enumerate(names):
        toks = [N._tok_fix(t) for t in N._merge_letters((s or "").split())]
        m, out = 0, []
        for i, t in enumerate(toks):
            c = code(t, toks[i + 1] if i + 1 < len(toks) else "")
            m |= BIT[c] if c else 0
            out.append(c or t)
        ms[k], cs[k] = m, " ".join(out)
    return ms, cs


def recs(c: str) -> dict:
    """{m1, c1, mx, cx} for one country in q1 / qx row order, cached in data/work/p14/rec_<c>_<side>.parquet."""
    out = {}
    for side, files in (("1", [B.TXT["train"] / f"{c}_s1.parquet"]), ("x", F.sx_files("train", c)[0])):
        p = OUT / f"rec_{c}_s{side}.parquet"
        if not p.exists():
            names = [n for f in files for n in pq.read_table(f, columns=["name"]).column(0).to_pylist()]
            m, cn = canon(names)
            pd.DataFrame({"mask": m, "cname": pd.Series(cn, dtype="string")}).to_parquet(p, index=False)
        t = pq.read_table(p)
        out["m" + side] = t.column("mask").to_numpy()
        out["c" + side] = t.column("cname").combine_chunks()          # arrow strings: compact, gathered per pair
    return out


def extras(q1: np.ndarray, qx: np.ndarray, r: dict) -> np.ndarray:
    """The 7 EXTRA columns for pairs (q1, qx) of one country, float32 rounded to 3 like every cached feature."""
    m1, mx = r["m1"][q1], r["mx"][qx]
    n1, nx = np.bitwise_count(m1), np.bitwise_count(mx)
    both = (n1 > 0) & (nx > 0)
    jacc = np.where(both, np.bitwise_count(m1 & mx) / np.maximum(np.bitwise_count(m1 | mx), 1), np.nan)
    a, b = r["c1"].take(pa.array(q1)).to_pylist(), r["cx"].take(pa.array(qx)).to_pylist()
    sim = lambda s: cpdist(a, b, scorer=s, workers=-1, dtype=np.float32) / 100
    cols = [n1, nx, both & (m1 == mx), both & (m1 != mx), jacc, sim(fuzz.ratio), sim(fuzz.token_set_ratio)]
    return np.column_stack(cols).astype(np.float32).round(3)


def countries() -> list:
    return sorted(pq.read_table(WORK / "train_source1.parquet", columns=["country"]).column(0).unique().to_pylist())


def cols65() -> list:
    return M.feats(pd.DataFrame(columns=pq.read_schema(F.xfiles("train", countries()[0])[0]).names))


def load(folds, frac: float = 1.0, seed: int = 0):
    """train_matcher.load's rows (same S1-hash sample, same file order) with the EXTRA columns appended:
    (X float32 [n, 72], y)."""
    cols = cols65()
    keep = lambda t: np.isin(t.column("fold").to_numpy(), folds) & (
        (t.column("q1").to_numpy().astype(np.uint64) * np.uint64(2654435761) + np.uint64(seed)) % np.uint64(1000)
        < frac * 1000)
    files = [(c, p) for c in countries() for p in F.xfiles("train", c)]
    n = sum(int(keep(pq.read_table(p, columns=["q1", "fold"])).sum()) for _, p in files)
    X, y, a, r = np.empty((n, len(cols) + len(EXTRA)), np.float32), np.empty(n, bool), 0, {}
    for c, p in files:
        r = r if r.get("c") == c else {"c": c, **recs(c)}
        t = pq.read_table(p, columns=ID + cols)
        d = t.filter(keep(t)).to_pandas()
        X[a:a + len(d), :len(cols)] = d[cols].to_numpy(np.float32)
        X[a:a + len(d), len(cols):] = extras(d.q1.to_numpy(), d.qx.to_numpy(), r)
        y[a:a + len(d)] = d.is_true.to_numpy()
        a += len(d)
        del t, d
        F.trim()
    assert a == n
    return X, y


def fit(X, y, Xv, yv, names: list, out, rounds: int = 8000):
    import lightgbm as lgb
    d = lgb.Dataset(X, y.astype(np.float32), feature_name=names, free_raw_data=True)
    dv = lgb.Dataset(Xv, yv.astype(np.float32), reference=d)
    m = lgb.train(M.PARAMS, d, rounds, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    m.save_model(str(out))
    return m


def f05(s1, sel, y, nt):
    """Macro-F0.5 term per holdout S1 (train_matcher.f05_per_s1) for the selected argmax pairs."""
    from train_matcher import f05_per_s1
    return f05_per_s1(np.bincount(s1[sel & y], minlength=len(nt)), np.bincount(s1[sel], minlength=len(nt)), nt)


def evaluate(models: dict, max_files=None) -> dict:
    """{model: (booster, n_cols)} -> F0.5 on the full-corpus holdout + the legal-form diagnostics.
    max_files: score only the first files per country (smoke test; the F0.5 is then meaningless)."""
    from run import half
    cnt = pd.read_parquet(WORK / "gt_pairs.parquet", columns=["s1"]).s1.value_counts()
    fo = pq.read_table(WORK / "s1_folds.parquet")
    cols, got = cols65(), {k: [] for k in models}
    nts, ids, grp, off, diag = [], [], [], 0, []
    for c in countries():
        ids1, _ = F.ids("train", c)
        f = fo.column("fold").to_numpy()[pc.index_in(ids1, value_set=fo.column("entity_id").combine_chunks()).to_numpy()]
        rows = np.flatnonzero(f == 0)
        pos = np.full(len(ids1), -1)
        pos[rows] = off + np.arange(len(rows))
        hid = ids1.take(rows).to_pylist()
        nts.append(cnt.reindex(hid).fillna(0).to_numpy())
        ids += hid
        grp += [c] * len(rows)
        r = recs(c)
        for p in F.xfiles("train", c)[:max_files]:
            t = pq.read_table(p, columns=ID + cols)
            qx = t.column("qx").to_numpy()
            t = t.filter(pa.array(np.isin(qx, qx[t.column("fold").to_numpy() == 0])))    # every pair of those SX
            q1, qx = t.column("q1").to_numpy(), t.column("qx").to_numpy()
            h, yt = t.column("fold").to_numpy() == 0, t.column("is_true").to_numpy()
            E, prs = np.empty((len(q1), len(EXTRA)), np.float32), {k: np.empty(len(q1), np.float32) for k in models}
            for a in range(0, len(q1), 400_000):              # the feature matrix one chunk at a time (RAM)
                part = t.slice(a, 400_000)
                A = np.empty((part.num_rows, len(cols) + len(EXTRA)), np.float32)
                for j, col in enumerate(cols):
                    A[:, j] = part.column(col).to_numpy()
                A[:, len(cols):] = E[a:a + part.num_rows] = extras(q1[a:a + part.num_rows], qx[a:a + part.num_rows], r)
                for k, (m, ncol) in models.items():
                    prs[k][a:a + part.num_rows] = m.predict(A[:, :ncol], num_threads=M.PARAMS["num_threads"])
                del A, part
            for k, pr in prs.items():
                sel = M.argmax_sx(qx, pr) & h
                got[k].append((pos[q1[sel]], pr[sel], yt[sel], E[sel, 2:4]))
            diag.append((E[h & yt, 2:4].sum(0), int((h & yt).sum())))
            del t, E, prs
            F.trim()
        off += len(rows)
    nt, grp, hs = np.concatenate(nts), np.array(grp), half(ids)
    ts, res = np.round(np.arange(0.02, 0.99, 0.01), 2), {}
    for k, v in got.items():
        s1, pr, yt, lg = (np.concatenate(z) for z in zip(*v))
        per = {t: f05(s1, pr >= t, yt, nt) for t in ts}
        t0 = max(ts, key=lambda t: per[t][hs == 0].mean())          # tuned on half 0
        ta = max(ts, key=lambda t: per[t].mean())                    # tuned on all fold 0 (train_matcher's number)
        fp, tp = (pr >= t0) & ~yt, (pr >= t0) & yt
        res[k] = {"t_half0": float(t0), "f05_half1": float(per[t0][hs == 1].mean()),
                  "f05_all": float(per[ta].mean()), "t_all": float(ta), "per": per[t0],
                  **{f"f05_half1_{c}": float(per[t0][(hs == 1) & (grp == c)].mean()) for c in np.unique(grp)},
                  "fp": int(fp.sum()), "tp": int(tp.sum()),
                  "fp_legal_eq": int(lg[fp, 0].sum()), "fp_legal_diff": int(lg[fp, 1].sum()),
                  "tp_legal_eq": int(lg[tp, 0].sum()), "tp_legal_diff": int(lg[tp, 1].sum())}
    eq, n_true = sum(d[0] for d in diag), sum(d[1] for d in diag)
    res["true_pairs"] = {"n": n_true, "legal_eq": int(eq[0]), "legal_diff": int(eq[1])}
    res["half1_n"] = int((hs == 1).sum())
    res["_hs"] = hs
    return res


def run(smoke: bool = False):
    """smoke: 2% samples, 50 trees, one X file per country, nothing logged -- exercises every step in minutes."""
    import lightgbm as lgb
    from ledger import log
    OUT.mkdir(parents=True, exist_ok=True)
    t0, cols = time.monotonic(), cols65()
    for c in countries():
        r = recs(c)
        print(f"{c}: legal form on {np.mean(r['m1'] > 0):.1%} of S1, {np.mean(r['mx'] > 0):.1%} of S2/S3 names "
              f"({time.monotonic() - t0:.0f}s)", flush=True)
    X, y = load((2,), frac=0.02 if smoke else 1.0)
    Xv, yv = load((1,), frac=0.02 if smoke else 0.25)
    print(f"train {len(y):,} pairs, val {len(yv):,}, {X.shape[1]} cols, {time.monotonic() - t0:.0f}s, "
          f"peak {B.peak_gb():.2f} GB", flush=True)
    models = {}
    for k, n in (("base", len(cols)), ("legal", len(cols) + len(EXTRA))):
        p = OUT / f"{'smoke_' if smoke else ''}m2_{k}.txt"
        if smoke or not p.exists():
            fit(np.ascontiguousarray(X[:, :n]), y, np.ascontiguousarray(Xv[:, :n]), yv, (cols + EXTRA)[:n], p,
                50 if smoke else 8000)
        models[k] = (lgb.Booster(model_file=str(p)), n)
        print(f"{k}: {models[k][0].num_trees()} trees, {time.monotonic() - t0:.0f}s, peak {B.peak_gb():.2f} GB", flush=True)
    del X, y, Xv, yv
    F.trim()
    res = evaluate(models, 1 if smoke else None)
    hs = res.pop("_hs")
    d = (res["legal"].pop("per") - res["base"].pop("per"))[hs == 1]      # paired: same S1, each model at its own t
    res["ab"] = {"diff_half1": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                 "s1_up": int((d > 0).sum()), "s1_down": int((d < 0).sum())}
    gain = dict(zip(models["legal"][0].feature_name(), models["legal"][0].feature_importance("gain").tolist()))
    tot = sum(gain.values())
    res["extra_gain_share"] = {k: round(gain[k] / tot, 5) for k in EXTRA}
    res["extra_gain_rank"] = {k: sorted(gain, key=gain.get, reverse=True).index(k) + 1 for k in EXTRA}
    res["runtime_s"], res["peak_gb"] = round(time.monotonic() - t0), round(B.peak_gb(), 2)
    if smoke:
        print(json.dumps(res, indent=1), "\nsmoke ok", flush=True)
        return res
    (OUT / "ab.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    cfg = {"train": "fold 2", "val": "fold 1, 25% S1-hash sample", "params": "model.PARAMS", "eval": "full-corpus "
           "holdout, argmax over all train S1, t tuned on half 0, F0.5 on half 1"}
    ok = res["ab"]["diff_half1"] >= 0.001
    for k in ("base", "legal"):
        log("exp", probe="P14-legal", subproblem="matching", name=f"M2-fold2-{k}", world="holdout-full",
            metrics={q: v for q, v in res[k].items()} | ({"diff_vs_base": res["ab"]["diff_half1"],
                                                          "se": res["ab"]["se"]} if k == "legal" else {}),
            config=cfg | {"features": f"{models[k][1]} (65 cached" + (" + 7 legal-form" if k == "legal" else "") + ")"},
            hypothesis="canonical legal forms (pvt ltd = private limited) separate look-alikes / recover spelling swaps",
            verdict="control" if k == "base" else ("keep" if ok else "drop"), runtime_s=res["runtime_s"],
            peak_gb=res["peak_gb"])
    return res


def _check():
    raw = pd.Series(["Shivam Investments Pvt. Ltd.", "SHIVAM INVESTMENTS PRIVATE LIMITED",
                     "Shivam Investments Pirvate Limted", "Shivam Investments LLP", "Metro Holding Co",
                     "Metro Holding Company", "Twisted Deli L.L.C.", "Twisted Deli Inc", "Bien & Frères S.A.R.L.",
                     "Lifewood P. Ltd", "Sunrise 1td", "Delhi Public School"])
    m, cn = canon(list(B.clean(raw, "anyascii")))
    form = [" ".join(c for c in CODES if v & BIT[c]) for v in m]
    assert form == ["ltd pvt"] * 3 + ["llp", "co", "co", "llc", "inc", "sarl", "ltd pvt", "ltd", ""], form
    assert cn[0] == cn[1] == cn[2] == "shivam investments pvt ltd", cn[:3]
    cn = pa.array(list(cn))
    E = extras(np.array([0, 0, 4, 6, 11]), np.array([1, 3, 5, 7, 0]), {"m1": m, "c1": cn, "mx": m, "cx": cn})
    assert E[:, 2].tolist() == [1, 0, 1, 0, 0] and E[:, 3].tolist() == [0, 1, 0, 1, 0], E
    assert E[0, 5] == 1 and E[0, 4] == 1 and np.isnan(E[4, 4]) and E[4, 0] == 0, E
    print("legal self-check ok")


if __name__ == "__main__":
    {"check": _check, "run": run, "smoke": lambda: run(True)}[sys.argv[1]]()
