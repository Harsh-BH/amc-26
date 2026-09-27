"""P50 C on test: output/<out> = output/<base> minus the France exclusivity veto. A base France pair is vetoed when the
P42 recipe itself accepts it (t .99 / rescue .9 / gc .7 / hn .8, caps 5/6: the pairs the LOCO analogue scored, so the
P40/P41/P46 tier pairs and P47 class adds are exempt), its SX's other-S1 c_hn p mass >= M and its post-rules p < BAR.
India/US lines byte-identical; candidate_pairs.tsv / code.zip hard links.
    .venv/bin/python probes/P50-idea-batch/c_build.py BASE OUT [M BAR] [ADDS_FROM]   (ADDS_FROM: report overlap with
    the pairs BASE has and ADDS_FROM lacks, e.g. P47's cls80 adds)"""
import os, sys, json, time
from pathlib import Path
import numpy as np
ROOT = Path(".").resolve()
sys.path[:0] = [str(ROOT / "probes/P47-fr-shape-match"), str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT),
                str(ROOT / "probes/P50-idea-batch")]
import p47, p42  # noqa: E402
import solution_p42 as S  # noqa: E402
from c_excl import other_mass  # noqa: E402

base, out = sys.argv[1], sys.argv[2]
M = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
BAR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.99
t0 = time.time()
st = p47.fr_state(base)
R = p47.RECIPE
rec = S.decide(st["s1"], st["is2"], st["p"], st["am"], st["sib"], st["hn_eq"], st["n"], R["t"], p47.CAP, R["r"], R["gc"], R["hn"])
o = other_mass(st["sx"], st["g"])
raw = st["pred"] & (o >= M) & (st["p"] < BAR)
rm = raw & rec
info = {"base": base, "m": M, "bar": BAR, "pred": int(st["pred"].sum()), "recipe_pred": int(rec.sum()),
        "base_not_recipe": int((st["pred"] & ~rec).sum()), "raw_rule": int(raw.sum()), "vetoed": int(rm.sum()),
        "raw_rule_on_non_recipe": int((raw & ~rec).sum()), "vetoed_s1": int(len(np.unique(st["s1"][rm]))),
        "vetoed_p_quant": [round(float(x), 4) for x in np.quantile(st["p"][rm], [0, .25, .5, .75, 1])] if rm.any() else []}
if len(sys.argv) > 5:                                    # overlap with the pairs base added on top of ADDS_FROM
    prev = p47.fr_state(sys.argv[5])["pred"]
    added = st["pred"] & ~prev
    info |= {"adds_vs_" + sys.argv[5]: int(added.sum()), "raw_rule_on_adds": int((raw & added).sum()),
             "vetoed_on_adds": int((rm & added).sum())}
n2, n3 = S.counts(st["s1"], st["is2"], st["pred"], st["n"])
m2, m3 = S.counts(st["s1"], st["is2"], st["pred"] & ~rm, st["n"])
info["s1_made_empty"] = int((((n2 + n3) > 0) & ((m2 + m3) == 0)).sum())
i1, ix = p42.fr_ids()
minus = {}
for a, b in zip(i1[st["s1"][rm]], ix[st["sx"][rm]]):
    minus.setdefault(a, set()).add(b)
src, od = ROOT / "output" / base, ROOT / "output" / out
od.mkdir(parents=True, exist_ok=True)
fr = set(i1)
n_fr = n_bad = n_chg = 0
with open(src / "matching_results.tsv", encoding="utf-8", newline="") as fi, \
        open(od / "matching_results.tsv", "w", encoding="utf-8", newline="") as fo:
    fo.write(fi.readline())
    for line in fi:
        k = line.split("\t", 1)[0]
        if k in fr:
            n_fr += 1
            if k in minus:
                cur = line.rstrip("\n").split("\t")[1]
                cur = set(cur.split(",")) if cur else set()
                n_bad += len(minus[k] - cur)
                fo.write(f"{k}\t{','.join(sorted(cur - minus[k]))}\n")
                n_chg += 1
                continue
        fo.write(line)
assert n_fr == len(i1) and n_bad == 0, (n_fr, len(i1), n_bad)
for name in ("candidate_pairs.tsv", "code.zip"):
    (od / name).unlink(missing_ok=True)
    if (src / name).exists():
        os.link(src / name, od / name)
info |= {"s1_changed": n_chg, "runtime_s": round(time.time() - t0)}
(od / "p50_info.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info), flush=True)
