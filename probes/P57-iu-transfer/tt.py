"""P57 label-free transfer check: do the relative-feature combiners flip test the way they flip the holdout, better than the
absolute-count ones? Each arm's half-0 orig boosters (ev57 cache) score P50's test frames (chunked by row group; r_all_test and
zoo_test rows are positional with T's), and its test adds / drops per #34-accepted pair are divided by the same boosters'
rates on the holdout report half 1 (full world and sw).     .venv/bin/python probes/P57-iu-transfer/tt.py -> data/work/p57/tt.json"""
import json
import os
import sys

os.environ.setdefault("P57_ARMS", "sub44r,sub54,rel44r,rel54")
sys.path.insert(0, "probes/P57-iu-transfer")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import ev57 as V  # noqa: E402

T = 0.71
W = V.E.W


def test_scores(c: str) -> dict:
    need = sorted(set(sum((v[0] for v in V.ARMS.values()), [])) | set(V.ABS) | {"p", "lp", "hc", "pa", "hn_lev", "q", "n_twins",
                                                                              "p_m2hn", "p_hn21"})
    f = pq.ParquetFile(W / f"p50/T_{c}.parquet")
    have = set(f.schema_arrow.names)
    cols = [k for k in need if k in have]
    r_all = pq.read_table(W / f"p45/oof/r_all_test_{c}.parquet", columns=["r"]).column(0).to_numpy()
    zt = pq.ParquetFile(W / f"p50/zoo_test_{c}.parquet")
    bs = {name: V.boosters(name, 0) for name in V.ARMS}
    out, pa_all, i0 = {name: [] for name in V.ARMS}, [], 0
    for rg in range(f.num_row_groups):
        A = f.read_row_group(rg, columns=cols).to_pandas()
        n = len(A)
        A["p_r_all"] = r_all[i0:i0 + n]
        z = pq.read_table(W / f"p50/zoo_test_{c}.parquet").slice(i0, n).to_pandas() if zt.num_row_groups != f.num_row_groups \
            else zt.read_row_group(rg).to_pandas()
        assert len(z) == n
        for m in V.Z.MEM:
            A[f"z_{m}"] = V.MT.lg(z[f"p_{m}"].to_numpy())
        for m in V.Z.RANK:
            A[f"z_{m}"] = z[f"p_{m}"].to_numpy(np.float64)
        A = V.relcols(V.te44.derive(A))
        for name, (cc, _) in V.ARMS.items():                   # 250k-row slices: one float64 copy of 55 cols is 110 MB
            out[name].append(np.concatenate([V.score(bs[name], A.iloc[j:j + 250_000], cc) for j in range(0, n, 250_000)]))
        pa_all.append(A.pa.to_numpy())
        i0 += n
        del A, z
    return {name: np.concatenate(v) for name, v in out.items()}, np.concatenate(pa_all)


def main():
    res = {}
    for c in V.E.C:
        s, pa = test_scores(c)
        old = pa >= T
        ref = pd.read_parquet(W / f"p50/prob_test_{c}_fullzoo.parquet", columns=["p"]).p.to_numpy()[
            pq.read_table(W / f"p50/T_{c}.parquet", columns=["row"]).column(0).to_numpy()]
        for name, p in list(s.items()) + [("shipped_fullzoo", ref)]:
            new = p >= T
            res[f"test|{c}|{name}"] = {"adds_per_acc": float((new & ~old).sum() / old.sum()),
                                       "drops_per_acc": float((old & ~new).sum() / old.sum()), "adds": int((new & ~old).sum()),
                                       "drops": int((old & ~new).sum())}
    for w in ("full", "sw"):
        R = json.load(open(W / f"p57/ev57_{w}.json"))
        acc = {c: R[f"{w}|orig|rel54|h0"][c]["acc34"] for c in V.E.C}
        for name in V.ARMS:
            for c in V.E.C:
                r = R[f"{w}|orig|{name}|h0"][c]
                res[f"{w}|{c}|{name}"] = {"adds_per_acc": r["adds"] / acc[c], "drops_per_acc": r["drops"] / acc[c]}
    json.dump(res, open(W / "p57/tt.json", "w"), indent=1)
    print("test / holdout-h1 flip-rate ratios, same half-0 boosters (1.0 = the arm flips test like it flips the holdout)")
    for name in V.ARMS:
        for c in V.E.C:
            t = res[f"test|{c}|{name}"]
            print(f"  {name:8s} {c:5s} test adds/acc {t['adds_per_acc']:.5f} drops/acc {t['drops_per_acc']:.5f} (n {t['adds']}/{t['drops']}) | "
                  + " | ".join(f"{w}: adds x{t['adds_per_acc'] / res[f'{w}|{c}|{name}']['adds_per_acc']:.2f} drops "
                               f"x{t['drops_per_acc'] / res[f'{w}|{c}|{name}']['drops_per_acc']:.2f}" for w in ("full", "sw")))
    for c in V.E.C:
        t = res[f"test|{c}|shipped_fullzoo"]
        print(f"  shipped sub54 (full-holdout refit) {c}: adds/acc {t['adds_per_acc']:.5f} drops/acc {t['drops_per_acc']:.5f}")


if __name__ == "__main__":
    main()
