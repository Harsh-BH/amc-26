"""P50 integrator: assemble a submission from any subset of France rule stages + India/US lines from any folder.

    asm50.py ledger                      -> data/work/p50/fr_ledger.npz: one stage label per France chain pair (P42
                                            france.npz rows), attributed by diffing the lineage folders
    asm50.py build OUT --iu FOLDER --fr STAGES [--dup] [--tmpl FOLDER]
    asm50.py test                        rebuilds sub48 / sub52_max / sub54 / sub_all_v3 and compares pair sets

France stages (the base = P42 recipe at t .99 is always on):
  rescue, gc, hn   the recipe's later stages (P42 decide steps: rescue .9, gc .7, hn .8); hn90 / hn95 = hn-equal
                   extras at a stricter bar instead of hn (the tiers stay exempt: they are their own stages)
  gib, xf, tier46  P41 gib / P40 xf_noaddr / P46 noise tier adds (tier_moved: recipe pairs these tiers displaced)
  veto48           P48's CE veto (recipe rescue/gc pairs sub48 dropped)
  cls80            P47 hn-change class adds (sub52_max - sub48)
  resc54           P54 in-band CE rescue u .99 (sub54 - sub52_max - dup); with p55 the u .995 set built on P55's decision
  p55              P55 combiner: veto (sub52_max - sub_all_v3) + adds
  p52              P52 lower rescue / gc bars (sub52cal - sub52_max)
--dup runs P54's duplicate-consistency step last (all countries), as P54 prescribes.
Dropping an early stage does not let later stages reuse the SX / caps it frees (a subset of the lineage, not a rebuild):
ponytail: subset assembly, rerun the stage builders in order if a dropped stage should free capacity.
"""
import os, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P47-fr-shape-match"), str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT)]
import p42, p47  # noqa: E402
import solution_p42 as S  # noqa: E402

W = ROOT / "data/work"
LED = W / "p50/fr_ledger.npz"
OUT = ROOT / "output"
ORDER = ["rescue", "gc", "hn", "hn90", "hn95", "gib", "xf", "tier46", "veto48", "cls80", "resc54", "p55", "p52"]


def fr_pairs(folder):
    """France (s1, sx) pairs of a folder, as a set."""
    s = set()
    for a, v in p42.read_fr(OUT / folder).items():
        if v:
            s.update((a, x) for x in v.split(","))
    return s


def mask(keys, pairs):
    return np.fromiter(((a, b) in pairs for a, b in keys), bool, len(keys))


def ledger():
    st = p47.fr_state("sub48")
    i1, ix = p42.fr_ids()
    keys = list(zip(i1[st["s1"]], ix[st["sx"]]))
    R = p47.RECIPE
    rs = S.decide(st["s1"], st["is2"], st["p"], st["am"], st["sib"], st["hn_eq"], st["n"], R["t"], p47.CAP, R["r"], R["gc"],
                  R["hn"], steps=True)
    in48 = st["pred"]
    in52, in54, inv3, incal = (mask(keys, fr_pairs(f)) for f in ("sub52_max", "sub54", "sub_all_v3", "sub52cal"))
    rec = rs["hn"]
    g, r = st["g"], np.nan_to_num(st["r"], nan=1.0)
    band = (g > S.LO) & (g < S.HI)
    rm48 = rec & ~in48
    v48 = rm48 & (rs["gc"] & ~rs["base"]) & band & (r < 0.2)      # P48: rescue / gc stage band pairs with bge r < .2
    t48 = in48 & ~rec
    lens = pd.read_parquet(W / "p46/flips_lens.parquet", columns=["s1", "sx"])
    tier46 = t48 & mask(keys, set(zip(lens.s1, lens.sx)))
    sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
    import france
    france.C = "France"
    t = np.flatnonzero(t48 & ~tier46)
    xfm = np.zeros(len(keys), bool)
    xfm[t] = france.noaddr_mask(i1[st["s1"][t]], ix[st["sx"][t]])       # P40 xf_noaddr's guard; the rest = P41 gib
    gibm = t48 & ~tier46 & ~xfm
    r995 = pd.read_parquet(W / "p54/fr_rescue__v3a_u0.995.parquet", columns=["s1", "sx"])
    r99 = pd.read_parquet(W / "p54/fr_rescue_sub52_max_u0.99.parquet", columns=["s1", "sx"])
    hnb = {}
    for b in (0.9, 0.95):                                    # hn-equal extras at a stricter bar (same earlier stages)
        q = S.decide(st["s1"], st["is2"], st["p"], st["am"], st["sib"], st["hn_eq"], st["n"], R["t"], p47.CAP, R["r"],
                     R["gc"], b, steps=True)
        assert (q["gc"] == rs["gc"]).all()
        hnb[f"hn{int(b * 100)}"] = q["hn"] & ~q["gc"]
    # P55's exact change set on sub52_max (rerun its builder into a scratch folder)
    subprocess.run([sys.executable, str(ROOT / "probes/P55-fr-combiner/p55.py"), "build", "sub52_max", "_p55tmp",
                    "v=0.6,scope=band,u=0.95"], check=True, stdout=subprocess.DEVNULL)
    p55 = mask(keys, fr_pairs("_p55tmp"))
    subprocess.run(["rm", "-rf", str(OUT / "_p55tmp")], check=True)
    n2, n3 = S.counts(st["s1"], st["is2"], in52, st["n"])
    two = ((n2 > 0) & (n3 > 0))[st["s1"]]
    L = {"rescue": rs["rescue"] & ~rs["base"], "gc": rs["gc"] & ~rs["rescue"], "hn": rs["hn"] & ~rs["gc"], "base": rs["base"],
         **hnb, "gib": gibm, "xf": xfm, "tier46": tier46, "tier_moved": rm48 & ~v48, "veto48": v48,
         "cls80": in52 & ~in48, "resc54": in54 & ~in52 & mask(keys, set(zip(r99.s1, r99.sx))),
         "p55veto": in52 & ~inv3, "resc995": inv3 & ~in52 & mask(keys, set(zip(r995.s1, r995.sx)))}
    L["p55veto"] = in52 & ~p55
    L["p55add"] = p55 & ~in52
    L["p52"] = incal & ~in52
    # dup copies inside the chain file (the rest are candidate appends): they are the dup step's, never a stage's
    L["dup54_in_file"] = in54 & ~in52 & ~L["resc54"]
    L["dupv3_in_file"] = inv3 & ~p55 & ~L["resc995"]
    L["dupv3_readds_p55veto"] = inv3 & L["p55veto"]
    L["resc54_hnx"] = L["resc54"] & st["hn_eq"] & two        # P54 rescue pairs of the hn-extras type (hn-equal, 2-sided)
    L["resc995_hnx"] = L["resc995"] & st["hn_eq"] & two
    np.savez_compressed(LED, s1=st["s1"], sx=st["sx"], is2=st["is2"], n=st["n"], **L)
    for k, v in L.items():
        print(f"{k:14s} {int(v.sum()):>8,}", flush=True)
    return L


def france_mask(stages):
    z = dict(np.load(LED))
    m = z["base"].copy()
    for s in ("rescue", "gc", "hn", "hn90", "hn95"):
        if s in stages:
            m |= z[s]
    for s in ("gib", "xf", "tier46"):
        if s in stages:
            m |= z[s]
    if {"gib", "xf", "tier46"} & stages:
        m &= ~z["tier_moved"]
    if "veto48" in stages:
        m &= ~z["veto48"]
    if "cls80" in stages:
        m |= z["cls80"]
    if "p55" in stages:
        m &= ~z["p55veto"]
        m |= z["p55add"]
        if "resc54" in stages:
            m |= z["resc995"]
    elif "resc54" in stages:
        m |= z["resc54"]
    if "p52" in stages:
        m |= z["p52"]
    sx = z["sx"][m]
    assert len(np.unique(sx)) == len(sx), f"{len(sx) - len(np.unique(sx))} SX at two S1: incompatible stage set"
    return z, m


def build(out, iu, stages, dup=False, tmpl=None):
    """output/<out>: France lines = the stage subset; India/US (and every non-France) line verbatim from output/<iu>;
    France lines whose set equals output/<tmpl or iu>'s are copied verbatim (member order), others sorted."""
    unknown = stages - set(ORDER)
    assert not unknown, unknown
    z, m = france_mask(stages)
    i1, ix = p42.fr_ids()
    new = {}
    for a, b in zip(i1[z["s1"][m]], ix[z["sx"][m]]):
        new.setdefault(a, set()).add(b)
    fr = set(i1)
    tm = p42.read_fr(OUT / (tmpl or iu)) if tmpl else None
    od = OUT / (out + "_nodup" if dup else out)
    od.mkdir(parents=True, exist_ok=True)
    n_fr = 0
    with open(OUT / iu / "matching_results.tsv", encoding="utf-8", newline="") as fi, \
            open(od / "matching_results.tsv", "w", encoding="utf-8", newline="") as fo:
        fo.write(fi.readline())
        for line in fi:
            k, v = line.rstrip("\n").split("\t", 1)
            if k in fr:
                n_fr += 1
                s = new.get(k, set())
                ref = tm[k] if tm is not None else v
                if set(ref.split(",")) - {""} == s:
                    fo.write(f"{k}\t{ref}\n" if tm is not None else line)
                else:
                    fo.write(f"{k}\t{','.join(sorted(s))}\n")
            else:
                fo.write(line)
    assert n_fr == len(i1)
    for name in ("candidate_pairs.tsv", "code.zip"):
        (od / name).unlink(missing_ok=True)
        if (OUT / iu / name).exists():
            os.link(OUT / iu / name, od / name)
    print(f"{od.name}: France {int(m.sum()):,} pairs on {len(new):,} S1 from stages {sorted(stages, key=ORDER.index)}; "
          f"India/US from {iu}", flush=True)
    if dup:
        subprocess.run([sys.executable, str(ROOT / "probes/P54-fresh-batch/build54.py"), od.name, out], check=True,
                       stdout=subprocess.DEVNULL)
        (OUT / out / "code.zip").unlink(missing_ok=True)
        if (od / "code.zip").exists():
            os.link(od / "code.zip", OUT / out / "code.zip")
        subprocess.run(["rm", "-rf", str(od)], check=True)
        print(f"{out}: + P54 dup step", flush=True)


def diff_lines(a, b):
    """Streamed compare of two folders' matching files (same S1 row order): (lines differing as sets, +pairs, -pairs)."""
    n = add = rm = 0
    with open(OUT / a / "matching_results.tsv", encoding="utf-8") as fa, open(OUT / b / "matching_results.tsv", encoding="utf-8") as fb:
        next(fa), next(fb)
        for la, lb in zip(fa, fb):
            ka, va = la.rstrip("\n").split("\t", 1)
            kb, vb = lb.rstrip("\n").split("\t", 1)
            assert ka == kb, "row order differs"
            if va != vb:
                x, y = set(va.split(",")) - {""}, set(vb.split(",")) - {""}
                if x != y:
                    n, add, rm = n + 1, add + len(x - y), rm + len(y - x)
    return n, add, rm


def test():
    base = {"rescue", "gc", "hn", "gib", "xf", "tier46", "veto48"}
    cases = [("sub48", "sub48", base, False), ("sub52_max", "sub52_max", base | {"cls80"}, False),
             ("sub54", "sub52_max", base | {"cls80", "resc54"}, True),
             ("sub_all_v3", "sub52_max", base | {"cls80", "p55", "resc54"}, True)]
    for target, iu, st, dup in cases:
        build("_asm_test", iu, st, dup)
        n, a, r = diff_lines("_asm_test", target)
        print(f"TEST {target}: {n} lines differ, +{a} / -{r} pairs vs target", flush=True)
        subprocess.run(["rm", "-rf", str(OUT / "_asm_test")], check=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ledger":
        ledger()
    elif cmd == "test":
        test()
    elif cmd == "build":
        a = sys.argv[3:]
        get = lambda k: a[a.index(k) + 1] if k in a else None
        build(sys.argv[2], get("--iu"), set(get("--fr").split(",")) if get("--fr") else set(), "--dup" in a, get("--tmpl"))
