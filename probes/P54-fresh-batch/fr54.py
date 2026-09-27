"""P54 idea 1 on test France: the in-band CE rescue on sub48's France recipe (P42 recipe t .99 / r .9 / gc .7 / hn .8 from
france.npz minus P48's veto), label-free counts + the adds list.

Rescue: free (argmax, non-sibling, not capped out) unaccepted pairs with 0.02 < g < 0.98, p0 = the OOD blend (no tier /
bxneg rule moved p), CE r >= u, whose SX sits in no line of the base file; best r first, up to the caps counted on the
base file's own line (so tier pairs count toward 5 / 6).

    probes/P54-fresh-batch/py.sh probes/P54-fresh-batch/fr54.py BASE U [U ...]   # -> data/work/p54/fr_rescue_<BASE>_u<U>.parquet
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P48-fr-ce-all"), str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT)]
import build48  # noqa: E402
import p42  # noqa: E402
import solution_p42 as S  # noqa: E402

CUR = build48.CUR
OUT = ROOT / "data/work/p54"


def base_lines(base, i1):
    """France lines of the base file: sx -> taken, per-S1 (n2, n3)."""
    fr = set(i1)
    taken, n2, n3 = set(), {}, {}
    with open(ROOT / "output" / base / "matching_results.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            k, v = line.rstrip("\n").split("\t")
            if k in fr and v:
                xs = v.split(",")
                taken.update(xs)
                n2[k] = sum(x.startswith("S2-") for x in xs)
                n3[k] = len(xs) - n2[k]
    return taken, n2, n3


def main(base, us):
    t0 = time.time()
    z = p42.fr_frame()
    st, p = p42.fr_decide(z, CUR)
    veto = build48.veto_mask(z, st, 0.2, ("rescue", "gc"))
    pred = st["hn"] & ~veto
    i1, ix = p42.fr_ids()
    s1, is2, n = z["s1"].astype(np.int64), z["is2"], int(z["n_s1"])
    g, r, p0 = z["g"], z["r"], z["p0"]
    _, free = S.base(s1, is2, p, S.argmax_sx(z["sx"].astype(np.int64), p), z["sib"], CUR["t"], tuple(CUR["cap"]))
    bl = 1 / (1 + np.exp(-(S.W_FR[0] * S.logit(g) + S.W_FR[1] * S.logit(np.nan_to_num(r, nan=0.5)) + S.W_FR[2])))
    band = (g > S.LO) & (g < S.HI)
    unmoved = np.abs(p0 - bl) <= 1e-4
    taken, f2, f3 = base_lines(base, i1)
    ids1, idsx = i1[z["s1"]], ix[z["sx"]]
    ok_sx = ~pd.Series(idsx).isin(taken).to_numpy()
    b2 = pd.Series(f2).reindex(i1).fillna(0).to_numpy(np.int64)       # per S1 index, base-file counts
    b3 = pd.Series(f3).reindex(i1).fillna(0).to_numpy(np.int64)
    m2, m3 = b2[s1], b3[s1]
    info = {"base": base, "recipe_pairs": int(pred.sum()), "veto": int(veto.sum()),
            "band_rows": int(band.sum()), "band_r_nan": int((band & np.isnan(r)).sum()),
            "pool_free_unacc_band": int((free & ~pred & band).sum()), "pool_unmoved": int((free & ~pred & band & unmoved).sum()),
            "pool_sx_free_in_file": int((free & ~pred & band & unmoved & ok_sx).sum()),
            "file_pairs": int(b2.sum() + b3.sum())}
    rr = np.nan_to_num(r, nan=0.0)
    for u in us:
        c = np.flatnonzero(free & ~pred & band & unmoved & ok_sx & (rr >= u))
        o, rk = S.rank_in(s1 * 2 + is2, rr, c)
        take = o[rk < np.where(is2[o], CUR["cap"][0] - m2[o], CUR["cap"][1] - m3[o])]
        e = (m2 + m3)[take] == 0
        one = np.where(is2[take], (m2[take] == 0) & (m3[take] > 0), (m3[take] == 0) & (m2[take] > 0))
        info[f"u{u}"] = {"cand": len(c), "add": len(take), "rate_per_recipe_pair": round(len(take) / int(pred.sum()), 5),
                         "s1": int(len(np.unique(s1[take]))), "empty": int(e.sum()), "onesided": int(one.sum()),
                         "nonempty_twosided": int((~e & ~one).sum()), "g_q": np.round(np.quantile(g[take], [.1, .5, .9]), 3).tolist() if len(take) else None,
                         "p_q": np.round(np.quantile(p[take], [.1, .5, .9]), 3).tolist() if len(take) else None,
                         "hn_eq": round(float(z["hn_eq"][take].mean()), 3) if len(take) else None,
                         "hn_eq_recipe": round(float(z["hn_eq"][pred].mean()), 3), "is2": round(float(is2[take].mean()), 3) if len(take) else None}
        pd.DataFrame({"s1": ids1[take], "sx": idsx[take], "g": g[take], "r": r[take]}).to_parquet(OUT / f"fr_rescue_{base}_u{u}.parquet")
        print(u, info[f"u{u}"], flush=True)
    info["runtime_s"] = round(time.time() - t0)
    (OUT / f"fr_rescue_{base}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if not k.startswith("u")}))


if __name__ == "__main__":
    main(sys.argv[1], [float(x) for x in sys.argv[2:]])
