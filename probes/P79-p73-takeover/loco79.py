"""P79 gate: P73's ev73.loco (P55 combiner + column, LOCO, shipped cell) on a trimmed FSETS: base, p67, the incumbent
ele4n / p67+ele4n (P73's best, loco73_ep4), and each new arm alone and with p67. Same bag seeds -> arm-vs-incumbent is paired.
    .venv/bin/python probes/P79-p73-takeover/loco79.py TAG ARM[,ARM] JOB [JOB ...]   -> probes/P79-p73-takeover/loco73_TAG.json
  ARM = the run key m of the r_ind_<m>_* files (e.g. mmln, bgbn, ele4r); JOB = their result folders (both LOCO countries).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P73-fr-independent"), str(ROOT)]
import ev73  # noqa: E402
import ledger  # noqa: E402

BASE_JOBS = ["p67-bge-loco-0927-1405", "p73b-both"]      # r_p67 (P67 v1 synthetic CE), r_ele4n (the incumbent)
_log = ledger.log
ledger.log = lambda kind, **k: _log(kind, **(k | {"probe": "P79-p73-takeover"}))

def main(tag, arms, jobs):
    ev73.FSETS = {"base": [], "p67": ["r_p67"], "ele4n": ["r_ele4n"], "p67+ele4n": ["r_p67", "r_ele4n"]} | \
        {k: v for a in arms for k, v in ((a, [f"r_{a}"]), (f"p67+{a}", ["r_p67", f"r_{a}"]),
                                        (f"p67+ele4n+{a}", ["r_p67", "r_ele4n", f"r_{a}"]))}
    ev73.HERE = HERE
    ev73.loco(*BASE_JOBS, *jobs, tag=f"_{tag}")
    vs_incumbent(tag, arms, jobs)


def vs_incumbent(tag, arms, jobs):
    """arm minus the incumbent p67+ele4n, paired (same bags) -> ledger row p79-gate-vs-incumbent-<tag>."""
    R = json.loads((HERE / f"loco73_{tag}.json").read_text())
    inc = {n: R[n]["p67+ele4n-base"]["dF"] for n in ("India->US", "US->India")}
    vs = {fs: {n: round(R[n][f"{fs}-base"]["dF"] - inc[n], 6) for n in inc} for a in arms for fs in (a, f"p67+{a}", f"p67+ele4n+{a}")}
    ok = {fs: d["India->US"] >= 0.0003 and d["US->India"] >= 0 for fs, d in vs.items()}
    print("VS_INCUMBENT", json.dumps(vs), json.dumps(ok), flush=True)
    ledger.log("exp", subproblem="france-combiner", name=f"p79-gate-vs-incumbent-{tag}", world="holdout-loco",
               metrics={f"{n}|{fs}-inc": v for fs, d in vs.items() for n, v in d.items()},
               config={"incumbent": "p67+ele4n (loco73_ep4)", "arms": arms, "jobs": jobs,
                       "rule": "fs minus incumbent: India->US >= +0.0003 and US->India >= 0"},
               hypothesis=f"arm(s) {arms} beat P73's best France-independent column set p67+ele4n",
               verdict="keep" if any(ok.values()) else "drop")


if __name__ == "__main__":
    vs = sys.argv[1] == "vs"                       # loco79.py vs TAG ARMS: redo the incumbent comparison from the JSON
    a = sys.argv[1 + vs:]
    (vs_incumbent if vs else main)(a[0], a[1].split(","), a[2:])
