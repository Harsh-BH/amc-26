"""P72: audit output/<out>'s France adds over output/<base> (label-free): pooled Qwen r, P64 legal-form class, both
combiners' c, and the Qwen r of base's own accepted France band pairs as the reference.
    .venv/bin/python probes/P72-corrupt-marks/audit72.py sub66_final sub72   -> data/work/p72/audit_<out>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P64-legal-form-feat"), str(ROOT / "probes/P72-corrupt-marks")]
import lf  # noqa: E402
from fr72 import split_txt  # noqa: E402

W = ROOT / "data/work"


def lines(path, keep):
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for ln in f:
            k, v = ln.rstrip("\n").split("\t", 1)
            if k in keep:
                out[k] = v
    return out


def main(base, out):
    fr = set(pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "country"]).query("country == 'France'").entity_id)
    a, b = lines(ROOT / "output" / base / "matching_results.tsv", fr), lines(ROOT / "output" / out / "matching_results.tsv", fr)
    add, rm = [], 0
    for k in fr:
        x, y = set(filter(None, a[k].split(","))), set(filter(None, b[k].split(",")))
        add += [(k, s) for s in y - x]
        rm += len(x - y)
    d = pd.DataFrame(add, columns=["s1", "sx"])
    f = pd.read_parquet(W / "p65/sm/frame_France.parquet", columns=["row", "s1", "sx", "g", "s1_txt", "sx_txt"])
    d = d.merge(f, on=["s1", "sx"], how="left")
    assert d.row.notna().all(), "every add is a frame row"
    rows = d.row.to_numpy(np.int64)
    d["c_num"], d["c_plain"] = np.load(W / "p55/cf_p72num.npy")[rows], np.load(W / "p55/cf_plain.npy")[rows]
    n1, _ = split_txt(d.s1_txt.to_numpy())
    nx, _ = split_txt(d.sx_txt.to_numpy())
    d["lf"] = lf.pair_class(lf.sig(pd.Series(n1)), lf.sig(pd.Series(nx)))
    q = pd.read_parquet(W / "p38/r/r_qwen_pool_test_France.parquet").drop_duplicates(["s1", "sx"])
    d = d.merge(q, on=["s1", "sx"], how="left")
    qs = lambda r: r.dropna().quantile([.1, .25, .5, .75, .9]).round(3).tolist()
    both = d.c_plain > 0.99
    # reference: base's accepted France pairs that Qwen scored (the band), a sample of 200k lines
    acc = pd.DataFrame([(k, s) for k in list(fr)[:200_000] for s in filter(None, a[k].split(","))], columns=["s1", "sx"])
    acc = acc.merge(q, on=["s1", "sx"], how="inner")
    res = {"base": base, "out": out, "adds": len(d), "removed": rm, "s1": int(d.s1.nunique()),
           "empties_filled": int(sum(1 for k in set(d.s1) if not a[k])),
           "lf_conflict_cls4": int((d.lf == 4).sum()), "lf_counts": d.lf.value_counts().sort_index().to_dict(),
           "g_quantiles": d.g.quantile([.1, .5, .9]).round(3).tolist(), "band_share": float(((d.g > .02) & (d.g < .98)).mean()),
           "also_plain_gt99": int(both.sum()),
           "qwen_cov": float(d.r.notna().mean()), "qwen_q": qs(d.r), "qwen_ge05": float((d.r.dropna() >= .5).mean()),
           "qwen_q_numonly": qs(d.r[~both]), "qwen_ge05_numonly": float((d.r[~both].dropna() >= .5).mean()),
           "qwen_q_both": qs(d.r[both]), "qwen_ge05_both": float((d.r[both].dropna() >= .5).mean()),
           "ref_accepted_band_qwen_q": qs(acc.r), "ref_accepted_band_qwen_ge05": float((acc.r >= .5).mean()), "ref_n": len(acc)}
    (W / f"p72/audit_{out}.json").write_text(json.dumps(res, indent=1, default=int))
    print(json.dumps(res, default=int))
    return d



def drop_conflicts(base, out):
    """Remove out's adds (over base) that P64's veto would remove (legal-form class 4), in place."""
    d = main(base, out)
    bad = set(zip(d.s1[d.lf == 4], d.sx[d.lf == 4]))
    p = ROOT / "output" / out / "matching_results.tsv"
    tmp = p.with_suffix(".tmp")
    n = 0
    with open(p, encoding="utf-8") as fi, open(tmp, "w", encoding="utf-8", newline="") as fo:
        fo.write(fi.readline())
        for ln in fi:
            k, v = ln.rstrip("\n").split("\t", 1)
            xs = v.split(",") if v else []
            keep = [x for x in xs if (k, x) not in bad]
            n += len(xs) - len(keep)
            fo.write(f"{k}\t{','.join(keep)}\n" if len(keep) != len(xs) else ln)
    tmp.replace(p)
    assert n == len(bad), (n, len(bad))
    print("dropped", n, "legal-form conflicts")


if __name__ == "__main__":
    (drop_conflicts if sys.argv[3:] == ["--drop-conflicts"] else main)(*sys.argv[1:3])
