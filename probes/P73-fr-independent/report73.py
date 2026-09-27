"""P73 veto report for the lead (27 Sep 20:00): (1) the stage each removed France pair came from, (2) a label-backed reading
of the removed set's profile — on both LOCO frames, the vetoed pairs at the same tau: true share, hn-equal share (hn_rel,
both sides numbered) and Qwen median for TRUE vs FALSE vetoed pairs, then France's removed hn-equal share → implied true share.
    .venv/bin/python probes/P73-fr-independent/report73.py OUT TAU LOCO_JOBS... -- FRANCE_JOBS...
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "code/business_entity_resolution/src")]
import ev73  # noqa: E402
import vt73  # noqa: E402


def removed(out, base="sub82"):
    gone = []
    with open(ROOT / "output" / base / "matching_results.tsv") as fa, open(ROOT / "output" / out / "matching_results.tsv") as fb:
        next(fa); next(fb)
        for la, lb in zip(fa, fb):
            if la != lb:
                a, _, x0 = la.rstrip("\n").partition("\t"); _, _, x1 = lb.rstrip("\n").partition("\t")
                gone += [(a, x) for x in set(x0.split(",")) - set(x1.split(",")) - {""}]
    return gone


def stages(gone):
    ch = pd.read_parquet(ROOT / "data/work/p78/final_changes.parquet")
    src = [(f"p78:{st}", set(zip(ch.s1[ch.step == st], ch.sx[ch.step == st]))) for st in ch.step.unique()]
    src += [("sub79q set", vt73._adds("sub79q", "sub77")), ("xlf", vt73._adds("sub77_xlf", "sub77")),
            ("groupe", vt73._adds("sub78g", "sub77")), ("p80", vt73._adds("sub82", "sub81_r3"))]
    s66 = set()
    with open(ROOT / "output/sub66_final/matching_results.tsv") as f:
        next(f)
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            s66 |= {(a, x) for x in b.split(",") if x}
    cnt = {}
    for p in gone:
        k = next((n for n, s in src if p in s), "sub66_final chain (P55)" if p in s66 else "other")
        cnt[k] = cnt.get(k, 0) + 1
    return dict(sorted(cnt.items(), key=lambda kv: -kv[1]))


def loco(tau, jobs):
    import p55
    import p61
    L = p55.L
    fr = {dr: p61.load(dr) for dr in ("US_India", "India_US")}
    inc = {dr: pd.DataFrame(ev73.columns(vt73.INC, dr, len(fr[dr]["X"]))) for dr in fr}
    out = {}
    for name, trn, tst in p55.DIRN:
        a, b = fr[trn], fr[tst]
        bs = p55.bag(ev73.design(a["X"], inc[trn], "p67+ele4n"), a["z"]["y"], None)
        sel0 = p61.arm_a(b, p55.bag_score(bs, ev73.design(b["X"], inc[tst], "p67+ele4n"), None))
        r, ns = vt73.fc_logit(jobs, tst, len(b["X"]))
        veto = sel0 & (b["z"]["g"] >= 0.98) & ~np.isnan(r) & (r < tau)
        y = b["z"]["y"].astype(bool); hr = b["X"]["hn_rel"].to_numpy(); q = b["q"]
        row = {"seeds": ns, "veto": int(veto.sum()), "true_share": round(float(y[veto].mean()), 3)}
        for lab, m in (("true", veto & y), ("false", veto & ~y), ("all_sel_true", sel0 & (b["z"]["g"] >= 0.98) & y)):
            both = m & (hr > 0)
            row[lab] = {"n": int(m.sum()), "hn_equal_given_both": round(float((hr[both] == 1).mean()), 3) if both.any() else None,
                        "qwen_n": int((m & np.isfinite(q)).sum()),
                        "qwen_median": round(float(np.nanmedian(q[m])), 3) if (m & np.isfinite(q)).any() else None}
        out[tst] = row
    return out


if __name__ == "__main__":
    out, tau = sys.argv[1], float(sys.argv[2])
    k = sys.argv.index("--")
    lj, fj = sys.argv[3:k], sys.argv[k + 1:]
    gone = removed(out)
    rep = {"out": out, "tau": tau, "removed": len(gone), "by_stage": stages(gone), "loco": loco(tau, lj),
           "france_profile": json.loads((ROOT / "output" / out / "vt73_build.json").read_text()).get("removed_profile")}
    from france import hn_rel                                # same hn_rel as the LOCO frames' column
    g = pd.DataFrame(gone, columns=["s1", "sx"])
    t1 = pd.read_parquet(ROOT / "data/work/rf_test_s1_France.parquet", columns=["id", "hn"]).set_index("id").hn
    tx = pd.read_parquet(ROOT / "data/work/rf_test_sx_France.parquet", columns=["id", "hn"]).set_index("id").hn
    hr = hn_rel(t1.reindex(g.s1).fillna("").astype(str).to_numpy(object), tx.reindex(g.sx).fillna("").astype(str).to_numpy(object))
    h = round(float((hr[hr > 0] == 1).mean()), 3)
    rep["france_hn_rel"] = {"both": round(float((hr > 0).mean()), 3), "equal_given_both": h,
                            "classes": {int(c): int(n) for c, n in zip(*np.unique(hr, return_counts=True))}}
    for dr, r in rep["loco"].items():                       # implied true share of France's removed set via hn-equal mixture
        ht, hf = r["true"]["hn_equal_given_both"], r["false"]["hn_equal_given_both"]
        r["france_implied_true_share"] = round((h - hf) / (ht - hf), 3) if ht is not None and hf is not None and ht != hf else None
    (ROOT / "output" / out / "report73.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
