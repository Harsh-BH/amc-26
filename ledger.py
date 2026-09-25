"""Append-only ledgers: `exp` = probe experiments, `pipe` = end-to-end pipeline runs.

    python ledger.py show exp --sort f05          python ledger.py show pipe
    python ledger.py selftest
    from ledger import log
    log("exp", probe="P3-block", subproblem="blocking", name="tfidf-char3-k20", world="dev",
        metrics={"pair_recall": 0.962, "cands_per_s1": 20}, config={"k": 20, "ngram": 3})
"""
import argparse, fcntl, json, os, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# AMC_LEDGER_DIR: probes in git worktrees log to the main checkout's ledger, so there is one ledger and
# branch merges never conflict on it (the probe runner sets it).
LDIR = Path(os.environ.get("AMC_LEDGER_DIR", ROOT / "ledger"))
FILES = {"exp": LDIR / "experiments.jsonl", "pipe": LDIR / "pipelines.jsonl"}
REQUIRED = {"exp": {"probe", "subproblem", "name", "world", "metrics", "config"},
            "pipe": {"version", "world", "metrics", "config"}}


def _git() -> str:
    run = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    sha = run("rev-parse", "--short", "HEAD") or "nogit"
    return sha + ("-dirty" if run("status", "--porcelain", "--untracked-files=no") else "")


def log(kind: str, **row) -> dict:
    missing = REQUIRED[kind] - row.keys()
    if missing:
        raise ValueError(f"{kind} ledger row missing {sorted(missing)}")
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": _git(), **row}
    FILES[kind].parent.mkdir(parents=True, exist_ok=True)
    with open(FILES[kind], "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)          # released on close
        f.write(json.dumps(row, default=str) + "\n")
    return row


def rows(kind: str) -> list:
    p = FILES[kind]
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def show(kind: str, sort: str | None = None, probe: str | None = None):
    rs = [r for r in rows(kind) if probe is None or r.get("probe") == probe]
    if sort:
        rs.sort(key=lambda r: r["metrics"].get(sort, float("-inf")), reverse=True)
    for r in rs:
        who = f'{r["probe"]}/{r["name"]}' if kind == "exp" else r["version"]
        m = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r["metrics"].items())
        print(f'{r["ts"][5:16]}  {r["git"]:<14} {who:<40} {r["world"]:<8} {m}  {r.get("verdict", "")}')


def _selftest():
    from multiprocessing import Pool
    tmp = Path(tempfile.mkdtemp()) / "exp.jsonl"
    FILES["exp"] = tmp
    with Pool(8) as p:                        # fork start method: children inherit the patched FILES
        p.map(_write_many, range(8))
    lines = tmp.read_text().splitlines()
    assert len(lines) == 1600 and all(json.loads(l)["probe"] == "t" for l in lines)
    try:
        log("exp", probe="t")
        raise AssertionError("missing fields must raise")
    except ValueError:
        pass
    print("ledger self-test ok")


def _write_many(i):
    for j in range(200):
        log("exp", probe="t", subproblem="t", name=f"{i}-{j}", world="t", metrics={"x": j}, config={})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["show", "selftest"])
    ap.add_argument("kind", nargs="?", default="exp", choices=["exp", "pipe"])
    ap.add_argument("--sort"); ap.add_argument("--probe")
    a = ap.parse_args()
    _selftest() if a.cmd == "selftest" else show(a.kind, a.sort, a.probe)
