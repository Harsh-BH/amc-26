"""P28: the address-anchor rule on France (test, no labels) on top of #9ab's France file (P19: IDF world + bge CE, t 0.985).
Writes data/work/p28/prob_test_France_9ab_<v>.parquet (p := max(p, 0.99) on rule pairs) and prints, per variant, the
pairs / S1 it changes and France's profile against P6's bands (empty 4.5-7.5%, matches per S1 >= 2.9).
    .venv/bin/python probes/P28-addr-anchor/fr.py --blocked bxneg cxneg     # then probes/P14-france-shift/combine.py
    ... fr.py --blocked --src data/work/p20/prob_test_France_idf_noisesib.parquet --t 0.975 bxneg   # any France file
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P28-addr-anchor"), str(ROOT)]
from anchor import boosted, decide, rule_masks  # noqa: E402

W = ROOT / "data/work"
_arg = lambda k, v: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else v
SRC = ROOT / _arg("--src", "data/work/p19/prob_test_France_idf144_bge.parquet")
IS_9AB = SRC.name == "prob_test_France_idf144_bge.parquet"
T = float(_arg("--t", 0.985))
LEARN_BLOCKED = "--blocked" in sys.argv          # learn the sibling-word guard on P3's blocked France list (4.3M pairs)
SUB9AB = ROOT.parent.parent / "amc-26/output/sub9ab/matching_results.tsv"


def profile(d: pd.DataFrame, pred: np.ndarray, n_s1: int) -> dict:
    k = d.s1[pred].value_counts()
    return {"empty": round(1 - len(k) / n_s1, 4), "mps": round(len(k.index.repeat(k)) / n_s1, 4)}


if __name__ == "__main__":
    from ledger import log
    t0 = time.time()
    d = pd.read_parquet(SRC)
    n_s1 = int((pd.read_parquet(W / "test_source1.parquet", columns=["country"]).country == "France").sum())
    p = d.p.to_numpy(np.float64)
    base = decide(d.sx, p, T)
    if SUB9AB.exists() and IS_9AB:                            # the file + t reproduce the uploaded #9ab France rows
        got = d[base].groupby("s1").sx.agg(lambda s: ",".join(sorted(s))).to_dict()
        ids = set(d.s1)
        diff = 0
        with open(SUB9AB, encoding="utf-8") as f:
            next(f)
            for line in f:
                s, _, rest = line.rstrip("\n").partition("\t")
                if s in ids:
                    diff += got.get(s, "") != rest
        print(f"#9ab France rows reproduced: {diff} S1 differ")
    learn = pd.read_parquet(W / "p3/cands_test_France.parquet", columns=["s1", "sx"]) if LEARN_BLOCKED else None
    m = rule_masks(d[["s1", "sx", "p"]], "test", "France", T, learn=learn)
    del learn
    b = profile(d, base, n_s1)
    print(f"France {len(d):,} pairs, {n_s1:,} S1; base t {T}: {b}; {m['info']}")
    print("sibling words:", sorted(m["sibw"]))
    res = {"base_empty": b["empty"], "base_mps": b["mps"], **m["info"]}
    tag = "_blk" if LEARN_BLOCKED else ""
    stem = "prob_test_France_9ab" if IS_9AB else SRC.stem
    opts = {sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a in ("--src", "--t")}
    for v in [a for a in sys.argv[1:] if not a.startswith("--") and a not in opts] or ["bxneg", "cxneg"]:
        q = boosted(p, m[v])
        pred = decide(d.sx, q, T)
        add, drop = pred & ~base, base & ~pred
        pr = profile(d, pred, n_s1)
        s1c = len(set(d.s1[add]) | set(d.s1[drop]))
        kinds = pd.Series(m["kind"][add]).value_counts().to_dict()
        r = {"rule_pairs": int(m[v].sum()), "rule_already_acc": int((m[v] & base).sum()), "added": int(add.sum()),
             "dropped": int(drop.sum()), "s1_changed": s1c, **{f"added_{k}": int(n) for k, n in kinds.items()}, **pr,
             "in_p6_bands": bool(0.045 <= pr["empty"] <= 0.075 and pr["mps"] >= 2.9)}
        print(f"  {v}: {r}", flush=True)
        res |= {f"{v}{tag}_{k}": x for k, x in r.items()}
        out = W / f"p28/{stem}_{v}{tag}.parquet"
        d.assign(p=q.astype(d.p.dtype)).to_parquet(out, index=False)
        print(f"    -> {out}")
    log("exp", probe="P28-addr-anchor", subproblem="decide", name=f"addr-anchor-France-on-{'9ab' if IS_9AB else SRC.stem}{tag}", world="test-France",
        metrics=res, config={"src": str(SRC), "t": T, "boost": 0.99, "sib_words": sorted(m["sibw"]),
                             "guard_learned_on": "p3 blocked list" if LEARN_BLOCKED else "the file's pairs"},
        hypothesis="France's matcher under-accepts insert/delete same-address pairs like an unseen-country model",
        verdict="inconclusive", runtime_s=round(time.time() - t0), peak_gb=0)
