"""Ship path for the India/US combiner + trace + anchor features (holdout paired test: +0.000046 ± 0.000033 / +0.000056 ±
0.000030 LB-eq, both halves). Same recipe as P50 e_zoo (5 seeds, te44.PRM, 200 rounds, residual on lp, hc guard, t .71).
    ta_ship.py fit                 -> data/work/p70/ta_s{0..4}.txt (full India + US holdout)
    ta_ship.py probs               -> data/work/p70/prob_test_{c}_ta.parquet (needs tr_test_{c} / an_test_{c})
    ta_ship.py build BASE OUT      -> BASE with India/US lines rebuilt from the ta probs. First proves the builder:
                                      fullzoo probs must reproduce BASE's India/US lines (dup copies carried over)."""
import csv, json, os, sys
from pathlib import Path
os.environ["P44_VARIANT"] = "r"
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P44-meta-stack"), str(ROOT / "probes/P50-idea-batch"), str(ROOT / "probes/P70-fr-roles")]
import ev as E, meta as MT, te44  # noqa: E402
import zoo50 as Z  # noqa: E402
import e_zoo  # noqa: E402
from e_ship import P50, attach  # noqa: E402
import traces as TRm, anchors as ANm  # noqa: E402
W = ROOT / "data/work"
COLS = list(e_zoo.COLS) + TRm.TR + ANm.AN
T = 0.71


def add_feats(A, split, c):
    for f, cols in ((W / f"p70/tr_{split}_{c}.parquet", TRm.TR), (W / f"p70/an_{split}_{c}.parquet", ANm.AN)):
        t = pd.read_parquet(f)
        assert len(t) == len(A), (f, len(t), len(A))
        for k in cols:
            A[k] = t[k].to_numpy(np.float32)
    return A


def fit():
    import lightgbm as lgb
    D = MT.prep(te44.MODELS)
    Z.attach(D)
    for c in E.C:
        add_feats(D[c][0], "train", c)
    X = pd.concat([D[c][0][COLS] for c in E.C])
    y = np.concatenate([D[c][0].y.to_numpy() for c in E.C])
    base = np.concatenate([D[c][0].lp.to_numpy() for c in E.C])
    for s in te44.SEEDS:
        b = lgb.train(te44.PRM(s), lgb.Dataset(X, y, init_score=base, categorical_feature=["cat"], free_raw_data=True), 200)
        b.save_model(str(W / f"p70/ta_s{s}.txt"))
        print("booster", s, flush=True)


def probs(chunk=400_000):
    import lightgbm as lgb
    bs = [lgb.Booster(model_file=str(W / f"p70/ta_s{s}.txt")) for s in te44.SEEDS]
    for c in E.C:
        A = te44.derive(attach(c, pd.read_parquet(P50 / f"T_{c}.parquet")))
        A = e_zoo.zcols(A, pd.read_parquet(P50 / f"zoo_test_{c}.parquet"))
        A = add_feats(A, "test", c)
        p = np.empty(len(A))
        for i in range(0, len(A), chunk):
            X = A.iloc[i:i + chunk]
            outs = [MT.lg(np.where(X.hc.to_numpy() == 1, X.p.to_numpy(), MT.sig(b.predict(X[COLS], raw_score=True) + X.lp.to_numpy()))) for b in bs]
            p[i:i + chunk] = MT.sig(np.mean(outs, axis=0))
        pr = pd.read_parquet(P50 / f"prob_test_{c}_fullzoo.parquet")
        out = pr.p.to_numpy().copy()
        old = out[A.row.to_numpy()] >= T
        out[A.row.to_numpy()] = p.astype(out.dtype)
        pr.assign(p=out).to_parquet(W / f"p70/prob_test_{c}_ta.parquet", index=False)
        new = p >= T
        print(c, f"fullzoo accepts {old.sum():,}, ta {new.sum():,}; adds {(new & ~old).sum():,}, drops {(old & ~new).sum():,}", flush=True)
        del A, pr


def lines_from(prob):
    d = pd.read_parquet(prob, columns=["s1", "sx", "p", "am"])
    d = d[d.am & (d.p >= T)]
    return d.groupby("s1").sx.apply(set).to_dict()


def build(base, out):
    base_p, out_p = ROOT / "output" / base, ROOT / "output" / out
    s1c = pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
    m = {r[0]: r[1] for r in csv.reader(open(base_p / "matching_results.tsv", encoding="utf-8"), delimiter="\t")}
    hdr = m.pop("source1_entity_id")
    info = {"base": base}
    new_lines = {}
    for c in E.C:
        ids = set(s1c.index[s1c == c])
        cur = {s: set(filter(None, v.split(","))) for s, v in m.items() if s in ids}
        ref = lines_from(P50 / f"prob_test_{c}_fullzoo.parquet")
        extra = {s: v - ref.get(s, set()) for s, v in cur.items()}          # pairs the combiner rule does not produce (dup copies)
        miss = sum(len(ref.get(s, set()) - v) for s, v in cur.items())
        n_extra = sum(len(v) for v in extra.values())
        info[f"{c}_repro"] = {"base_pairs": sum(len(v) for v in cur.values()), "fullzoo_rule_missing_from_base": miss, "base_extra_pairs": n_extra}
        assert miss == 0 and n_extra <= 50, info[f"{c}_repro"]            # builder must reproduce BASE (dup copies only)
        ta = lines_from(W / f"p70/prob_test_{c}_ta.parquet")
        used = set()
        for s in cur:
            v = ta.get(s, set()) | extra[s]
            new_lines[s] = v
        chg = sum(1 for s in cur if new_lines[s] != cur[s])
        adds = sum(len(new_lines[s] - cur[s]) for s in cur); rms = sum(len(cur[s] - new_lines[s]) for s in cur)
        info[f"{c}_change"] = {"s1_changed": chg, "added": adds, "removed": rms}
    owner = {}
    for s, v in new_lines.items():
        for x in v:
            assert x not in owner, ("SX twice", x)
            owner[x] = s
    out_p.mkdir(parents=True, exist_ok=True)
    with open(out_p / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write(f"source1_entity_id\t{hdr}\n")
        for s, v in m.items():                       # unchanged lines (France and India/US alike) are written byte for byte
            changed = s in new_lines and new_lines[s] != set(filter(None, v.split(",")))
            g.write(f"{s}\t{','.join(sorted(new_lines[s])) if changed else v}\n")
    for name in ("candidate_pairs.tsv", "code.zip"):
        if not (out_p / name).exists() and (base_p / name).exists():
            os.link(base_p / name, out_p / name)
    (out_p / "ta_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    {"fit": fit, "probs": probs, "build": lambda: build(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
