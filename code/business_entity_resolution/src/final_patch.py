"""The final submission #90u (LB 0.989382): #77 (final_build.py) + the evening change sets, as attributed change rows.

    python code/business_entity_resolution/src/final_build.py output/sub66_final_small output/sub77 --upto 41
    python code/business_entity_resolution/src/final_patch.py output/sub77 output/sub90u
    python code/business_entity_resolution/src/final_patch.py --summary        # what each change set did
    python code/business_entity_resolution/src/final_patch.py --selftest

final_patch.parquet holds one row per id on every line of the two TSVs that differs from #77's: (file, s1, pos, sx,
step, op). op 'keep' = already on #77's line; 'add' = new, and step names the change set after which the pair appears
and stays; 'remove' = a #77 pair that is gone, step = the change set that removed it for good. Counts are net: a pair
one set added and a later one removed appears in neither. A changed line is rebuilt from its keep / add rows in pos
order, byte for byte. The rows were derived from the staged chain by experiments/P91-patch-provenance/provenance.py;
STEPS below maps each step to the experiment that made it (details: src/experiments/INDEX.md).
"""
import argparse
import tempfile
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
PATCH = SRC / "final_patch.parquet"
NAMES = ("matching_results.tsv", "candidate_pairs.tsv")
STEPS = {   # step -> (experiment, what it does), in chain order
    "xlf": ("P70 xlf_build.py", "address-less names equal to a unique S1 core up to the legal form"),
    "groupe": ("P71 build71.py b", "one S1 word swapped for 'groupe' at the identical house number"),
    "p79q_ce": ("P73 / P79 build79.py", "France-independent CE: its vetoes and its Qwen-gated adds"),
    "p76_r3": ("P76 build_r3.py", "two-member frames lose base-stage pairs with p in [.9868, .9925)"),
    "p80_lf_kept": ("P80 build_lf.py", "address-less names that keep the legal form, unique full-name match"),
    "type_word_veto": ("P70 tw_audit.py", "pairs that swap in a broad type word (compagnie / institut / pharmacie)"),
    "sub68fr_keys": ("P68 fr_key.py", "(core, house number, city)-unique pairs the blocker never proposed"),
    "nc_noise_swaps": ("P70 tw_audit.py", "same-house-number noise-suffix swaps the blocker never proposed"),
    "p73_narrow_veto": ("P73 vt73.py", "sibling-step pairs the France-independent CE rejects (logit < -4.5)"),
    "iu_trace_anchor": ("P70 ta_ship.py", "India / US lines re-decided with generator-trace + anchor columns"),
    "p88_rules_pass12": ("P81 chainA.sh + P88 profile.py", "credited third-party ADD rules, passes 1-2, filtered"),
    "p88_rules_pass3": ("P81 + P88", "the same rules' pass 3, same filter"),
    "p80_us_redecision": ("P80 stagec.py + build_sub.py", "US owners re-decided by the cross-fitted stage C"),
}


def lines(d: pd.DataFrame, name: str) -> dict:
    """{s1: rebuilt line} for every S1 the rows touch in file `name`: its keep / add ids joined in pos order."""
    f = d[d.file == name]
    k = f[f.op != "remove"].sort_values(["s1", "pos"]).groupby("s1").sx.agg(",".join).to_dict()
    return {s: f"{s}\t{k.get(s, '')}" for s in f.s1.unique()}


def main(base, out, patch=PATCH) -> None:
    """Copy base/{matching_results,candidate_pairs}.tsv to out/, rebuilding every S1 line the change rows touch."""
    d = pd.read_parquet(patch)
    assert set(d.step) <= {"sub77", *STEPS} and set(d.op) <= {"keep", "add", "remove"}, "unknown step / op"
    Path(out).mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        rep, seen = lines(d, name), 0
        with open(Path(base) / name, encoding="utf-8") as f, open(Path(out) / name, "w", encoding="utf-8") as g:
            g.write(f.readline())
            for line in f:
                s1 = line.split("\t", 1)[0]
                seen += s1 in rep
                g.write(rep[s1] + "\n" if s1 in rep else line)
        assert seen == len(rep), f"{name}: {len(rep) - seen} patched S1 are not in the base"


def summary(patch=PATCH) -> pd.DataFrame:
    """Per change set: experiment, what it does, and the pairs it added / removed in each file (net)."""
    d = pd.read_parquet(patch)
    c = d[d.op != "keep"].groupby(["step", "file", "op"]).size().unstack(["file", "op"], fill_value=0)
    t = pd.DataFrame({"experiment": [STEPS[s][0] for s in STEPS], "what": [STEPS[s][1] for s in STEPS]}, index=list(STEPS))
    cols = {("matching_results.tsv", "add"): "match +", ("matching_results.tsv", "remove"): "match -",
            ("candidate_pairs.tsv", "add"): "cand +"}
    for k, v in cols.items():
        t[v] = c[k].reindex(t.index, fill_value=0) if k in c else 0
    return t


def _selftest() -> None:
    """Self-check: keep / add rows rebuild a line in pos order, removals empty it, untouched lines are copied."""
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "b").mkdir()
        for n in NAMES:
            (t / "b" / n).write_text("h\nS1-a\tS2-1\nS1-b\tS2-2\nS1-c\t\n", encoding="utf-8")
        rows = [(n, *r) for n in NAMES for r in (("S1-a", 1, "S3-9", "groupe", "add"), ("S1-a", 0, "S2-1", "sub77", "keep"),
                                                  ("S1-b", -1, "S2-2", "p76_r3", "remove"))]
        pd.DataFrame(rows, columns=["file", "s1", "pos", "sx", "step", "op"]).to_parquet(t / "p.parquet", index=False)
        main(t / "b", t / "o", t / "p.parquet")
        for n in NAMES:
            assert (t / "o" / n).read_text(encoding="utf-8") == "h\nS1-a\tS2-1,S3-9\nS1-b\t\nS1-c\t\n", n
        assert summary(t / "p.parquet").loc["groupe", "match +"] == 1
    print("final_patch self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="apply the attributed evening change sets: #77 -> #90u")
    ap.add_argument("base", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.summary:
        print(summary().to_string())
    else:
        main(a.base, a.out)
