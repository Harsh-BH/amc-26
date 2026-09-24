# AMC-26 Master-Agent Probe Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reach the strongest private-leaderboard macro F0.5 by 27 Sep 22:00 IST. Get there by splitting the
entity-resolution problem into sub-problems, attacking each with a parallel Opus 5.5 "probe" session in tmux,
merging their evidence in one master session, then building and iterating one end-to-end pipeline, with every
experiment and every pipeline run recorded in a ledger.

**Architecture:** This Claude Code session is the **master**. Probes are headless `claude -p --model
claude-opus-5-5` sessions, each in its own tmux window in session `amc`, memory-capped with `systemd-run`. They
talk to the master only through files: `probes/<id>/BRIEF.md` goes in; `RESULT.md`, ledger rows and a `DONE`
marker come out. The master merges results after each wave into `probes/MERGE-<n>.md` and decides the next
wave. The pipeline lives from day one where the final zip expects it: `code/business_entity_resolution/src/`.

**Tech Stack:** Python 3.12 venv (pandas 3, pyarrow, scikit-learn, lightgbm, torch+CUDA) plus licence-checked
additions: rapidfuzz, jellyfish, anyascii, indic-transliteration, ai4bharat-transliteration, sparse-dot-topn,
faiss-cpu, polars, duckdb, splink. Models: multilingual-e5-small/base (MIT), LaBSE (Apache-2.0),
xlm-roberta-base (MIT), muril-base-cased (Apache-2.0). Orchestration: tmux 3.7, Claude Code 2.1.282,
systemd user scopes, jq.

**Spec:** `docs/PROBLEM.md` (rules, metric, data findings). Evidence behind every decision here is in
`data/notes/`: `train_report.md`, `test_report.md`, `research_licences.md`, `research_techniques.md`.

## Global Constraints

- Hard deadline **27 Sep 2026 23:59:49 IST**; plan to have the final upload done by **22:00 IST**.
- **5 leaderboard submissions per day**; assume the count resets at midnight IST.
- Final model **MIT or Apache-2.0, ≤ 8B params**. `unidecode` (GPL) and `aksharamukha` (AGPL) are banned. Qwen2.5-3B is **not** Apache; Qwen2.5-1.5B and Qwen3 dense models are.
- **No external lookup**: no geocoding, registries, hosted LLM/embedding APIs. Probes may WebSearch *docs and papers* only.
- Every one of the **1,732,544 test S1 rows** is in every output, France included. Country is an open label set.
- Use `.venv/bin/python`; read TSVs with `sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False`.
- Never commit `data/`, `*.tsv` or `output/`. Ledgers and RESULT.md files are committed and must not quote raw rows.
- **Every probe and sub-agent runs on Opus 5.5** (`--model claude-opus-5-5 --effort xhigh`).
- **RAM: 15.7 GB total.** Each probe's CLI costs about 0.7 GB, plus whatever its Python work needs. The sum of MemoryMax caps must stay below MemAvailable − 1 GB.
- Unsupervised statistics (TF-IDF vocab, IDF) are fitted on **train only** until the Google Form answers open question 4.
- **pandas 3 string regex runs on RE2, where `\w` is ASCII-only.** `str.replace(r"[^\w ]", …)` silently deletes every
  Devanagari and accented letter. Strip punctuation with an explicit class (see `s0_exact.py`) or use Python `re` on lists.

## Review Focus

1. **Unseen country label in the output.** France never appears in train. `run.py` must pass through any country
   string. Test: inject fake rows with `country="Xland"` into the dev world; they must get output rows (Task 2.1).
2. **Holdout optimism.** A 220k-S1 mini-world has far fewer confusable records than the 10M-record test corpus.
   Test: P3 reports blocking recall on holdout S1 against the **full** train S2/S3 corpus, not just the mini-world.
3. **Concurrent ledger writes.** Several probes log at once. Test: 8 processes × 200 appends must give
   1,600 parseable lines (Task 0.4 self-check).
4. **Silent probe failure.** A probe can exit (OOM, crash, time-box) without a RESULT.md. `DONE` records the exit
   code, and the master rejects any RESULT.md missing template sections, then resumes that session (Task 0.5).
5. **Matches outside candidates.** A final match that was never a candidate is a pipeline bug and gets flagged in
   the audit. Test: `run.py` asserts matches ⊆ candidates for every S1 before writing (Task 2.1).

---

## How the master operates (applies to every phase)

```
                     ┌───────────── master (this session, Opus 5.5) ─────────────┐
                     │ writes BRIEF.md · launches · watches DONE · reads RESULT  │
                     │ merges → MERGE-n.md · decides next wave · submits         │
                     └──┬──────────────┬──────────────┬──────────────┬───────────┘
          probes/P1/BRIEF.md    probes/P2/…      probes/P3/…      ledger/*.jsonl  (file bus)
                        ▼              ▼              ▼
   tmux session "amc":  [P1-taxonomy]  [P2-script]   [P3-block]  …  each: claude -p --model claude-opus-5-5
                        systemd scope MemoryMax=…  → RESULT.md + ledger rows + DONE(exit code)
```

1. **Brief.** The master writes `probes/<id>/BRIEF.md` from the roster below. `launch.sh` prepends `probes/_COMMON.md`.
2. **Launch.** `probes/launch.sh <id> <MemoryMax>` refuses to start if RAM is short, opens a tmux window and records the session id.
3. **Watch.** The master runs one background watcher that exits when any new `DONE` appears; the harness wakes the master. You can follow any probe live with `tmux attach -t amc`.
4. **Collect.** The master reads `RESULT.md`, checks it has every template section, and pulls its ledger rows with `ledger.py show exp --probe <id>`.
5. **Follow up.** If something is unclear, the master resumes that probe instead of starting over: `claude -p --resume $(cat probes/<id>/session_id) --model claude-opus-5-5 "…"`.
6. **Merge.** When a wave is done, the master writes `probes/MERGE-<n>.md` (template in Step 4), sends you a 5-line checkpoint, and briefs the next wave.

---

## Phase 0 — Foundation (master, 25 Sep 04:00–07:00 IST)

### Task 0.0: Free RAM and install licence-checked dependencies

**Files:** Create `code/business_entity_resolution/requirements.txt`

- [ ] **Step 1: Ask the user to close Firefox/Chrome tabs.** Browsers hold ~8 GB right now (Firefox 6.2 GB); MemAvailable is 4.9 GB. Target ≥ 11 GB.
- [ ] **Step 2: Install**

```bash
VIRTUAL_ENV=.venv uv pip install rapidfuzz jellyfish anyascii indic-transliteration \
    ai4bharat-transliteration sparse-dot-topn faiss-cpu polars duckdb splink sentence-transformers
.venv/bin/python -c "import rapidfuzz, jellyfish, anyascii, indic_transliteration, sparse_dot_topn, faiss, polars, duckdb, splink; print('ok')"
```

Expected: `ok`. If `ai4bharat-transliteration` fails to build, drop it — P2 falls back to `indic-transliteration`.

- [ ] **Step 3: Pin exact versions:** `VIRTUAL_ENV=.venv uv pip freeze > code/business_entity_resolution/requirements.txt`, then trim it to the packages the pipeline imports.

### Task 0.1: TSV → parquet, once

**Files:** Create `code/business_entity_resolution/src/io_tsv.py`

**Interfaces — Produces:** `read_tsv(path) -> pd.DataFrame` (all columns `str`); `data/work/{train,test}_source{1,2,3}.parquet`, `data/work/train_ground_truth.parquet`.

- [ ] **Step 1: Write the module with its self-check**

```python
"""TSV <-> parquet for the 2026 data. QUOTE_NONE because names and addresses contain literal quotes.

    python src/io_tsv.py            # convert data/student_resource/dataset -> data/work/*.parquet
    python src/io_tsv.py --check    # self-check only
"""
import csv, os, sys, tempfile
from pathlib import Path
import pandas as pd

DATA = Path(os.environ.get("ER_DATA", "data/student_resource/dataset"))
WORK = Path(os.environ.get("ER_WORK", "data/work"))


def read_tsv(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
                       keep_default_na=False, engine="c")


def convert():
    WORK.mkdir(parents=True, exist_ok=True)
    for f in sorted(DATA.glob("*/*.tsv")):
        df = read_tsv(f)
        with open(f, encoding="utf-8") as fh:
            n = sum(1 for _ in fh) - 1
        assert len(df) == n, f"{f.name}: parsed {len(df)} rows but file has {n}"
        df.to_parquet(WORK / f"{f.stem}.parquet", index=False)
        print(f"{f.name}: {len(df):,} rows")
        del df


def _check():
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as t:
        t.write('entity_id\tbusiness_name\tbusiness_address\tcountry\n'
                'S1-1\tJoe\'s "Best" Pizza\t12 "A" Rd, Pune\tIndia\n'
                'S1-2\tNA\t\tUS\n')
    df = read_tsv(t.name)
    assert len(df) == 2, "quotes must not merge lines"
    assert df.business_name[1] == "NA", "keep_default_na=False keeps the literal string NA"
    assert df.business_address[1] == "", "empty stays empty, not NaN"
    print("io_tsv self-check ok")


if __name__ == "__main__":
    _check()
    if "--check" not in sys.argv:
        convert()
```

- [ ] **Step 2: Run the check:** `.venv/bin/python code/business_entity_resolution/src/io_tsv.py --check`. Expected: `io_tsv self-check ok`.
- [ ] **Step 3: Convert:** run it again without `--check`. Expected: 7 lines of counts matching `docs/PROBLEM.md` (2,206,821 / 5,034,616 / 5,285,603 / 2,206,821 / 1,732,544 / 4,887,273 / 5,082,316).
- [ ] **Step 4: Commit:** `git add code/business_entity_resolution/src/io_tsv.py && git commit -m "Add TSV->parquet loader with quote-safe parsing"`

### Task 0.2: The metric, and remove the stale SMAPE scaffold

**Files:** Create `code/business_entity_resolution/src/metric.py`; delete `metrics.py`, `baseline.py`, `submit.py`

**Interfaces — Produces:** `f_beta(pred: set, true: set, beta=0.5) -> float`; `score(pred: dict, true: dict, groups: dict | None) -> dict` (keys `f05`, `n`, `singleton_acc`, `f05_<group>`); `read_sets(path) -> dict[str, set]`.

- [ ] **Step 1: Write the module with its self-check**

```python
"""Macro F0.5 per Source-1 entity, exactly as the 2026 statement defines it.

    python src/metric.py                               # self-check
    python src/metric.py pred.tsv truth.tsv            # score a matching_results file
"""
import sys


def f_beta(pred: set, true: set, beta: float = 0.5) -> float:
    if not true:                      # singleton: empty is perfect, anything else is 0
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:                       # includes an empty prediction on a non-singleton
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def score(pred: dict, true: dict, groups: dict | None = None) -> dict:
    """Mean F0.5 over every S1 in `true` (a missing prediction counts as empty); optional per-group means."""
    per = {s: f_beta(pred.get(s, set()), t) for s, t in true.items()}
    out = {"f05": sum(per.values()) / len(per), "n": len(per)}
    single = [per[s] for s, t in true.items() if not t]
    if single:
        out["singleton_acc"] = sum(single) / len(single)
    for g in sorted(set(groups.values())) if groups else []:
        vals = [v for s, v in per.items() if groups.get(s) == g]
        out[f"f05_{g}"] = sum(vals) / len(vals)
    return out


def read_sets(path) -> dict:
    """matching_results / ground-truth / candidate TSV -> {s1: set(ids)}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            out[s1] = set(rest.split(",")) if rest else set()
    return out


def _check():
    ex = f_beta({"S2-47", "S2-193", "S3-812"}, {"S2-47", "S3-812"})
    assert round(ex, 3) == 0.714, ex                                  # the PDF's worked example
    assert f_beta(set(), set()) == 1.0 and f_beta({"S2-1"}, set()) == 0.0   # singleton rules
    assert f_beta(set(), {"S2-1", "S2-2"}) == 0.0                     # missed non-singleton
    assert f_beta({"S2-1", "S2-9"}, {"S2-1", "S2-2"}) == 0.5          # 1 right + 1 wrong
    assert round(f_beta({"S2-1"}, {f"S2-{i}" for i in range(5)}), 3) == 0.556   # precision wins
    s = score({"a": set(), "b": {"x"}}, {"a": set(), "b": {"x"}, "c": {"y"}}, {"a": "US", "b": "US", "c": "India"})
    assert s["f05"] == 2 / 3 and s["singleton_acc"] == 1.0 and s["f05_India"] == 0.0
    print("metric self-check ok")


if __name__ == "__main__":
    _check()
    if len(sys.argv) == 3:
        print(score(read_sets(sys.argv[1]), read_sets(sys.argv[2])))
```

- [ ] **Step 2: Run:** `.venv/bin/python code/business_entity_resolution/src/metric.py`. Expected: `metric self-check ok`.
- [ ] **Step 3: Delete the stale files and their references:** `git rm metrics.py baseline.py submit.py`; drop their three rows from the README Files table and the "stale" line in CLAUDE.md.
- [ ] **Step 4: Commit:** `git commit -am "Replace SMAPE scaffold with the 2026 macro F0.5 metric"`

### Task 0.3: Folds and test-like mini-worlds

**Files:** Create `code/business_entity_resolution/src/worlds.py`

**Interfaces — Consumes:** Task 0.1 parquet. **Produces:** `data/work/gt_pairs.parquet` (`s1`, `sx`); `data/work/s1_folds.parquet` (`entity_id`, `country`, `fold`); `data/work/worlds/<name>/{s1,sx,truth}.parquet`; `load_world(name) -> (s1_df, sx_df, truth: dict[str, set])`.

Why a mini-world, not a random row sample: in test, every S2/S3 record either has its parent in the S1 file or is a
distractor. A world takes a set of S1 records, **all** of their children, and extra parentless records per country
until (S2+S3)/S1 matches the test density (US 5.75, India 5.82; train is only 4.67). Records whose real parent is
outside the world are parentless inside it, exactly like test distractors. So the 1-to-many exclusivity step and
the singleton decisions behave as they will on test.

| World | S1 | Use |
|---|---|---|
| `dev` | 30k from fold 1, stratified by country | fast probe iteration (~200k records) |
| `holdout` | all of fold 0 (~220k) | the number every pipeline decision is made on |
| `train` | 200k from folds 2–9 | candidate pairs to fit matchers |

- [ ] **Step 1: Write the module**

```python
"""Deterministic folds + test-like mini-worlds (see plan Task 0.3 for the why).

    python src/worlds.py            # build gt_pairs, folds, and the dev/holdout/train worlds
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path(os.environ.get("ER_WORK", "data/work"))
TEST_DENSITY = {"US": 5.75, "India": 5.82}          # (S2+S3)/S1 measured on test_source*.tsv
WORLDS = {"dev": ([1], 30_000), "holdout": ([0], None), "train": ([2, 3, 4, 5, 6, 7, 8, 9], 200_000)}


def folds(ids: pd.Series, k: int = 10) -> np.ndarray:
    return (pd.util.hash_pandas_object(ids, index=False).to_numpy() % k).astype(np.int8)


def build_base():
    gt = pd.read_parquet(WORK / "train_ground_truth.parquet")
    pairs = (gt.assign(sx=gt.matched_entity_ids.str.split(",")).explode("sx")
               .query("sx.notna() and sx != ''", engine="python")[["source1_entity_id", "sx"]]
               .rename(columns={"source1_entity_id": "s1"}))
    pairs.to_parquet(WORK / "gt_pairs.parquet", index=False)
    s1 = pd.read_parquet(WORK / "train_source1.parquet", columns=["entity_id", "country"])
    s1["fold"] = folds(s1.entity_id)
    s1.to_parquet(WORK / "s1_folds.parquet", index=False)


def build_world(name: str, fold_ids: list, n_s1: int | None, seed: int = 0):
    s1 = pd.read_parquet(WORK / "s1_folds.parquet")
    s1 = s1[s1.fold.isin(fold_ids)]
    if n_s1:
        s1 = s1.groupby("country").sample(frac=n_s1 / len(s1), random_state=seed)
    pairs = pd.read_parquet(WORK / "gt_pairs.parquet")
    kids = pairs[pairs.s1.isin(s1.entity_id)]
    sx = pd.concat([pd.read_parquet(WORK / f"train_source{i}.parquet") for i in (2, 3)], ignore_index=True)
    in_world = sx.entity_id.isin(kids.sx)
    parts = [sx[in_world]]
    kid_country = s1.set_index("entity_id").country.reindex(kids.s1).to_numpy()
    for c, g in s1.groupby("country"):
        need = round(TEST_DENSITY[c] * len(g)) - int((kid_country == c).sum())
        pool = sx[~in_world & (sx.country == c)]
        parts.append(pool.sample(n=max(need, 0), random_state=seed))
    out = WORK / "worlds" / name
    out.mkdir(parents=True, exist_ok=True)
    full_s1 = pd.read_parquet(WORK / "train_source1.parquet")
    full_s1[full_s1.entity_id.isin(s1.entity_id)].to_parquet(out / "s1.parquet", index=False)
    pd.concat(parts, ignore_index=True).to_parquet(out / "sx.parquet", index=False)
    kids.to_parquet(out / "truth.parquet", index=False)
    print(f"{name}: {len(s1):,} S1, {sum(len(p) for p in parts):,} S2/S3, {len(kids):,} true links")


def load_world(name: str):
    d = WORK / "worlds" / name
    s1, sx, links = (pd.read_parquet(d / f) for f in ("s1.parquet", "sx.parquet", "truth.parquet"))
    truth = {s: set() for s in s1.entity_id}
    for a, b in zip(links.s1, links.sx):
        truth[a].add(b)
    return s1, sx, truth


def _check():
    s1, sx, truth = load_world("dev")
    assert set(truth) == set(s1.entity_id)
    kids = set().union(*truth.values())
    assert kids <= set(sx.entity_id), "every child of a world S1 is in the world"
    for c in ("US", "India"):
        dens = (sx.country == c).sum() / (s1.country == c).sum()
        assert abs(dens - TEST_DENSITY[c]) < 0.05, (c, dens)
    single = sum(1 for t in truth.values() if not t) / len(truth)
    assert 0.045 < single < 0.067, single            # ~5.6% singletons, as in train
    print("worlds self-check ok")


if __name__ == "__main__":
    build_base()
    for name, (f, n) in WORLDS.items():
        build_world(name, f, n)
    _check()
```

- [ ] **Step 2: Run:** `.venv/bin/python code/business_entity_resolution/src/worlds.py`. Expected: three `name: … S1` lines, then `worlds self-check ok`. Peak RAM ~4 GB, from loading 10.3M S2/S3 rows once per world. If MemAvailable is under 6 GB, build worlds one at a time.
- [ ] **Step 3: Commit:** `git add code/business_entity_resolution/src/worlds.py && git commit -m "Add deterministic folds and test-density mini-worlds"`

### Task 0.4: The ledgers

**Files:** Create `ledger.py`, `ledger/experiments.jsonl`, `ledger/pipelines.jsonl`

**Interfaces — Produces:** `log(kind: "exp"|"pipe", **row) -> dict`; `rows(kind) -> list[dict]`; CLI `python ledger.py show exp|pipe [--sort KEY] [--probe ID]`.

Row schemas: missing required fields raise an error, so no half-rows.

| Ledger | Required | Conventional extras |
|---|---|---|
| `exp` (one row per probe experiment) | `probe, subproblem, name, world, metrics{}, config{}` | `hypothesis, verdict(keep/drop/inconclusive), runtime_s, peak_gb, notes, artifacts` |
| `pipe` (one row per end-to-end run) | `version, world, metrics{}, config{}` | `lb, submission, runtime_s, peak_gb, cands_per_s1, notes` |

Standard metric keys: `f05, f05_US, f05_India, singleton_acc, pair_recall, s1_full_recall, cands_per_s1, auc, ap`.
`ts` and `git` (short sha plus `-dirty`) are added automatically.

- [ ] **Step 1: Write the module with a concurrency self-check**

```python
"""Append-only ledgers: `exp` = probe experiments, `pipe` = end-to-end pipeline runs.

    python ledger.py show exp --sort f05          python ledger.py show pipe
    python ledger.py selftest
    from ledger import log
    log("exp", probe="P3-block", subproblem="blocking", name="tfidf-char3-k20", world="dev",
        metrics={"pair_recall": 0.962, "cands_per_s1": 20}, config={"k": 20, "ngram": 3})
"""
import argparse, fcntl, json, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = {"exp": ROOT / "ledger" / "experiments.jsonl", "pipe": ROOT / "ledger" / "pipelines.jsonl"}
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
    FILES[kind].parent.mkdir(exist_ok=True)
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
```

- [ ] **Step 2: Run:** `.venv/bin/python ledger.py selftest`. Expected: `ledger self-test ok`.
- [ ] **Step 3: Create the empty ledgers and commit:** `mkdir -p ledger && touch ledger/experiments.jsonl ledger/pipelines.jsonl && git add ledger.py ledger && git commit -m "Add flock-guarded experiment and pipeline ledgers"`

### Task 0.5: The probe runner

**Files:** Create `probes/launch.sh`, `probes/run_probe.sh`, `probes/watch.sh`, `probes/_COMMON.md`, `probes/RESULT_TEMPLATE.md`

**Interfaces — Produces:** `probes/launch.sh <id> [MemoryMax]` → tmux window `amc:<id>`; files `probes/<id>/{session_id, stream.jsonl, DONE}`; `probes/watch.sh` exits printing the first probe that finished and hasn't been collected yet.

- [ ] **Step 1: `probes/run_probe.sh`.** This runs inside the tmux window; the stream/stdin/memory mechanics were verified on 25 Sep.

```bash
#!/usr/bin/env bash
# One headless Opus 5.5 probe: memory-capped, rendered live in the tmux pane, DONE marker with exit code.
set -uo pipefail
ID=$1; MEM=$2; DIR=probes/$ID; SID=$(cat "$DIR/session_id")
cat probes/_COMMON.md "$DIR/BRIEF.md" \
| systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 \
    claude -p --model claude-opus-5-5 --effort xhigh --dangerously-skip-permissions \
           --session-id "$SID" -n "$ID" --output-format stream-json --verbose \
| tee -a "$DIR/stream.jsonl" \
| jq -rj --unbuffered '
    if .type=="assistant" then (.message.content[] |
        if .type=="text" then .text+"\n"
        elif .type=="tool_use" then "  -> \(.name) \((.input.command // .input.file_path // "")|tostring|.[0:120])\n"
        else empty end)
    elif .type=="result" then "\n== \(.subtype)  cost $\(.total_cost_usd)\n" else empty end'
echo "exit=${PIPESTATUS[1]} $(date -Is)" > "$DIR/DONE"
```

- [ ] **Step 2: `probes/launch.sh`**

```bash
#!/usr/bin/env bash
# usage: probes/launch.sh <probe-id> [MemoryMax, default 3G]   — needs probes/<id>/BRIEF.md
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
ID=$1; MEM=${2:-3G}; DIR=probes/$ID
[[ -f $DIR/BRIEF.md ]] || { echo "missing $DIR/BRIEF.md" >&2; exit 1; }
if [[ -e $DIR/session_id && ! -e $DIR/DONE ]]; then echo "$ID is still running" >&2; exit 1; fi
need=$(( $(numfmt --from=iec "$MEM") + 1024**3 + 700*1024**2 ))     # cap + 1G headroom + CLI
avail=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) * 1024 ))
(( avail > need )) || { echo "RAM: $((avail>>20)) MiB available, $ID needs $((need>>20)) MiB" >&2; exit 1; }
rm -f "$DIR/DONE" "$DIR/.collected"; uuidgen > "$DIR/session_id"
tmux has-session -t amc 2>/dev/null || tmux new-session -d -s amc -n master
tmux new-window -d -t amc -n "$ID" -c "$PWD" "probes/run_probe.sh $ID $MEM; exec bash"
echo "launched $ID  session=$(cat "$DIR/session_id")  watch: tmux attach -t amc"
```

- [ ] **Step 3: `probes/watch.sh`.** The master runs this in the background; it exits and wakes the master.

```bash
#!/usr/bin/env bash
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
while :; do
  for d in probes/*/; do
    if [[ -e $d/DONE && ! -e $d/.collected ]]; then touch "$d/.collected"; echo "finished: $d $(cat "$d/DONE")"; exit 0; fi
  done
  sleep 20
done
```

- [ ] **Step 4: `probes/_COMMON.md`.** This preamble is prepended to every brief.

```markdown
You are a PROBE in a 72-hour Amazon ML Challenge team (business entity resolution, macro F0.5). A master
agent reads only your files. Work autonomously; never ask questions — decide, note the assumption, move on.

Read first: docs/PROBLEM.md (rules + data findings), data/notes/train_report.md, data/notes/test_report.md,
data/notes/research_techniques.md, data/notes/research_licences.md, and ledger/experiments.jsonl
(`.venv/bin/python ledger.py show exp`) to avoid repeating finished experiments.

Environment: repo root is the cwd. Use .venv/bin/python only. Data: data/work/*.parquet (full) and
worlds via `sys.path.insert(0, "code/business_entity_resolution/src"); from worlds import load_world`
-> load_world("dev"|"holdout"|"train") returns (s1_df, sx_df, truth{s1: set}). Metric: `from metric import score`.
Your process is memory-capped (OOM = killed): work on the dev world first; touch full data only in
chunks, one source at a time. GPU work must hold the lock: `flock /tmp/amc-gpu.lock <cmd>`. pandas 3 string regex is RE2:
`\w` is ASCII-only, so never strip with `[^\w]` — it deletes Indic and accented letters.

Hard rules: any model or library you adopt must be MIT/Apache-2.0/BSD/ISC (see research_licences.md;
unidecode and aksharamukha are banned) and models <= 8B params. No external data or APIs in pipeline
code; WebSearch/WebFetch is allowed for documentation and papers only. Never filter on {US, India}.
Never edit files under code/business_entity_resolution/src/ — write your code in probes/<your-id>/;
the master promotes it. Never commit; the master does. Never write raw data rows into RESULT.md.

Logging: every experiment gets one ledger row:
  from ledger import log   (sys.path.insert(0, "."))
  log("exp", probe="<your-id>", subproblem=..., name=..., world=..., metrics={...}, config={...},
      hypothesis=..., verdict="keep|drop|inconclusive", runtime_s=..., peak_gb=...)
Deliverable: probes/<your-id>/RESULT.md following probes/RESULT_TEMPLATE.md exactly. Write a draft
RESULT.md within your first hour and update it as you go, so a crash still leaves evidence.
Finish by making RESULT.md final, then stop.
```

- [ ] **Step 5: `probes/RESULT_TEMPLATE.md`**

```markdown
# <probe-id> — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
## Recommendation   (exactly what the master should adopt: module, function signature, config)
## Evidence         (table: ledger name | world | key metrics | verdict)
## What failed      (approaches tried and dropped, with the number that killed them)
## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
## Open questions   (for the master; each with your best guess)
```

- [ ] **Step 6: Smoke test.** Create `probes/P0-smoke/BRIEF.md` containing `Run .venv/bin/python ledger.py show exp, log one exp row with probe="P0-smoke", write RESULT.md with the 7 template headings, stop.` Then run `chmod +x probes/*.sh && probes/launch.sh P0-smoke 1G && probes/watch.sh`. Expected: `finished: probes/P0-smoke/ exit=0 …`, one ledger row, and a RESULT.md with 7 headings.
- [ ] **Step 7: Commit:** `git add probes ledger && git commit -m "Add tmux probe runner, watcher and probe contract"`. Add `probes/*/stream.jsonl`, `probes/*/session_id`, `probes/*/DONE` and `probes/*/.collected` to `.gitignore` first: they are logs and runtime state.

### Task 0.6: Submission #1: exact-key sanity baseline (day 1, ~07:00)

**Files:** Create `code/business_entity_resolution/src/s0_exact.py` (deleted once v1 exists)

Purpose: prove the whole path works end to end (format → validator → portal → `SCORED`), and get the first
reading of leaderboard score vs holdout score. Expected holdout F0.5 is low, around 0.2–0.3, because only 22% of
true pairs share an exact name.

- [ ] **Step 1: Write it**

```python
"""Submission #1 (sanity only): exact normalized name + first address segment, within country.

    python src/s0_exact.py holdout     # score on the holdout world, log a pipe row
    python src/s0_exact.py test        # write output/matching_results.tsv + output/candidate_pairs.tsv
"""
import sys
from pathlib import Path
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path[:0] = [str(SRC), str(SRC.parents[2])]          # src/ and the repo root (ledger.py)
from worlds import WORK, load_world
from metric import score
from ledger import log

PUNCT = r"[\s,.;:'\"()\[\]{}\-&/|#!?*+_@%$~`^<>=]+"    # explicit class: RE2's \w would eat Indic/accents


def norm(s: pd.Series) -> pd.Series:
    return s.str.lower().str.replace(PUNCT, " ", regex=True).str.strip()


def key(df: pd.DataFrame) -> pd.Series:
    return df.country + "|" + norm(df.business_name) + "|" + norm(df.business_address.str.split(",").str[0])


def match(s1: pd.DataFrame, sx: pd.DataFrame) -> dict:
    k1 = pd.DataFrame({"s1": s1.entity_id, "k": key(s1)})
    k1 = k1[norm(s1.business_name) != ""]                 # never match on an empty name
    kx = pd.DataFrame({"sx": sx.entity_id, "k": key(sx)})
    m = kx.merge(k1, on="k").drop_duplicates("sx")       # each S2/S3 goes to at most one S1
    out = {s: set() for s in s1.entity_id}
    for a, b in zip(m.s1, m.sx):
        out[a].add(b)
    return out


def write(sets: dict, path: Path, col: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s, ids in sets.items():
            f.write(f"{s}\t{','.join(sorted(ids))}\n")


if __name__ == "__main__":
    if sys.argv[1] == "holdout":
        s1, sx, truth = load_world("holdout")
        m = score(match(s1, sx), truth, dict(zip(s1.entity_id, s1.country)))
        print(m)
        log("pipe", version="s0-exact", world="holdout", metrics=m, config={"key": "country|name|addr_seg0"})
    else:
        s1 = pd.read_parquet(WORK / "test_source1.parquet")
        sx = pd.concat([pd.read_parquet(WORK / f"test_source{i}.parquet") for i in (2, 3)], ignore_index=True)
        sets = match(s1, sx)
        assert len(sets) == len(s1) == 1_732_544
        write(sets, Path("output/matching_results.tsv"), "matched_entity_ids")
        write(sets, Path("output/candidate_pairs.tsv"), "candidate_entity_ids")
```

- [ ] **Step 1b: Run it on holdout:** `.venv/bin/python code/business_entity_resolution/src/s0_exact.py holdout`. Expected: a dict with `f05` around 0.2–0.3 and a new `pipe` ledger row.
- [ ] **Step 2:** Run on test and write `output/matching_results.tsv` plus `output/candidate_pairs.tsv` (the same sets).
- [ ] **Step 3:** From `data/student_resource/`, run `python3 utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir dataset/test`. Expected: `PASS`.
- [ ] **Step 4 (user):** Upload on the play page. Before that, zip `code/` as the code file (open question 2). Paste the LB score back; the master logs `log("pipe", version="s0-exact", world="test", lb=<score>, submission="#1 25-Sep HH:MM", …)`.

### Task 0.7: Ask the open questions (user, 5 minutes)

- [ ] The master drafts the 7 questions from `docs/PROBLEM.md` → "Open questions" as one Google Form message; you submit it. Question 4 (fitting statistics on test) and question 2 (a zip per upload) change the plan, so ask them first.

---

## Step 1 + Step 3 — Sub-problems and the probe roster

Why these sub-problems: the data profile puts the F0.5 at stake here. 14% of true pairs share no name token
(websites, DBA names, cross-script); 18% of India pairs cross scripts; 26% of S2/S3 records are distractors;
France is 15% of test with no labels. The pipeline is **normalize → block → featurize → score → decide**, and
each stage has its own failure mode and metric, so each gets a probe. Every probe is Opus 5.5 at effort xhigh.

| Probe | Sub-problem | The question it must answer | Primary metric | Cap | Wave | Box |
|---|---|---|---|---|---|---|
| **P1-taxonomy** | Data analysis for modelling | Which difficulty classes hold the missed F0.5, and what normalization recovers them? | exact-match lift on true pairs; F0.5-at-stake per class | 2.5G | A | 4h |
| **P2-script** | Multilingual / cross-script | How do we make a Devanagari/Telugu name comparable to its Latin S1? | top-k retrieval recall of the true S1 for cross-script pairs | 3G | A | 4h |
| **P3-block** | Searching the relevant rows | What candidate generator reaches ≥ 97% pair recall at ≤ 40 candidates/S1 on the **full** corpus? | pair recall vs cands/S1; RAM; time | 4G | A | 4.5h |
| **P7-scale** | Engineering at 11.7M records | Can the full test run fit in 15 GB locally, and how should it be chunked? | peak RAM and wall time per stage | 5G | A (late) | 3h |
| **P4-match** | Pair scoring | Which features and model separate matches from look-alikes best? | pair AP; dev F0.5 with a fixed decision rule | 4G | B | 4.5h |
| **P6-france** | Unseen country | How do we stay precise on France with zero labels? | leave-one-country-out F0.5 drop | 3G | B | 4h |
| **P5-decide** | Pair scores → answer sets | Which decoding maximizes macro F0.5? | holdout F0.5 | 2.5G | C | 3h |
| **P4b-rerank** | Hard-pair reranker (GPU) | Does a fine-tuned multilingual cross-encoder on the uncertain band pay for itself? | holdout F0.5 gain per GPU-hour | 4G | C | 3h |

RAM schedule (after Task 0.0 frees ~11 GB): Wave A runs P1 + P2 + P3 together (9.5G of caps plus 2.1G of CLI
overhead). P7 launches the moment the first of them finishes. Wave B runs P4 + P6; Wave C runs P5 + P4b.
`launch.sh` refuses any launch that would break the budget, so the schedule can't OOM the laptop.

### P1-taxonomy — BRIEF.md

```markdown
# P1-taxonomy: data analysis for modelling
Goal: find where the F0.5 is, and build normalize.py v1 that recovers the easy part.
1. Difficulty taxonomy. On 100k true pairs from the train world, classify each pair (first match wins):
   exact-after-normalization · legal-suffix/abbreviation-only · typo (edit distance <= 2 on core name) ·
   word reorder · website-as-name · '|'-joined name/site · cross-script · no shared name token but address
   agrees · genuinely different surface form (DBA). Give shares per country and per source (S2 vs S3).
   Estimate F0.5-at-stake per class (share × how often a naive matcher misses it) and rank the classes.
2. Look-alikes (hard negatives): how many S1 share a normalized name within a country (chains)? For true pairs,
   how often does a wrong S1 with the same name exist? What address evidence separates them?
3. Address structure per country: segment order, city/state position, state abbreviations, PIN/ZIP presence,
   landmark phrases (Near/Opp/Behind — beware "Opp, AL" is a town), NULL placeholders. Define a city_key
   extraction rule and measure city_key agreement on true vs random pairs.
4. Mine substitution dictionaries automatically from aligned true pairs (token pairs that co-occur: pvt↔private,
   st↔street, co↔company …) and a legal-suffix inventory per country. French legal forms, street types and
   articles come from domain knowledge (see research_techniques.md §6); they can't be validated — label them.
Deliverable: probes/P1-taxonomy/normalize.py with name_norm, name_core (suffix-stripped), acronym, is_website,
website_core, addr_norm, city_key, state, postcode, house_number — vectorized over pandas Series, and a
self-check. Report the exact-match and token-overlap lift on true vs random pairs, before vs after.
```

### P2-script — BRIEF.md

```markdown
# P2-script: cross-script names and addresses (India)
Goal: make Indic-script S2/S3 names and addresses comparable to their Latin S1 counterparts.
Facts: S1 is always Latin; ~23.5% of India S2 names and ~13% of S3 names are Indic script; 18% of India true
pairs cross scripts (~550k labelled cross-script pairs in train — use them).
1. Inventory which scripts occur (Devanagari, Telugu, Tamil, Bengali, Gujarati, Kannada, …) and their shares.
2. Compare, on cross-script true pairs vs same-country hard negatives:
   a) indic-transliteration (MIT) rule romanization + schwa deletion + phonetic folding (aa→a, w→v, sh→s, z→j …)
   b) ai4bharat IndicXlit (MIT, ~11M params) neural romanization — measure throughput on ~1.2M strings (CPU and GPU under flock)
   c) a mapping learned from the train cross-script pairs (char n-gram alignment or a tiny seq2seq) — train only
   d) multilingual embeddings (multilingual-e5-small MIT, LaBSE Apache-2.0) cosine
   Metrics: similarity AUC, and recall@{1,5,20} of the true S1 when retrieving among that country's dev-world S1s.
3. Indic-script addresses: can transliterated city names be matched to Latin city_keys? Learn a city
   dictionary from train pairs if needed.
Deliverable: probes/P2-script/translit.py: to_latin(s: pd.Series) -> pd.Series, plus speed and RAM at full scale
and a recommendation (which method or combination, and where: blocking key, feature, or both).
```

### P3-block — BRIEF.md

```markdown
# P3-block: candidate generation ("search the relevant rows")
Goal: the recall ceiling. Target >= 0.97 pair recall at <= 40 candidates per S1, measured on holdout S1 against
the FULL train S2/S3 corpus (10.3M), not only the mini-world (the mini-world has fewer confusers).
Always block within country. Blockers to compare (research_techniques.md §1):
 B1 exact keys (name_core, city_key+first name token)       B2 IDF-weighted token inverted index, top-k
 B3 TF-IDF char 3/4-gram cosine top-k with sparse_dot_topn (Sparkly-style; name, and name+city variants)
 B4 MinHash-LSH on name tokens (optional)                    B5 embedding ANN (faiss) for the cross-script residual
 B6 unions / rank fusion; both directions: S1→SX top-k and SX→S1 top-k (each SX has at most one parent, so
    the reverse direction is natural)
Use probes/P1-taxonomy/normalize.py and probes/P2-script/translit.py when they appear (poll); until then,
lowercase + strip punctuation. Fit vectorizers on train only.
Report per country and per P1 difficulty class: pair recall, S1 full-recall, cands/S1, recall@k curves
(k = 5, 10, 20, 50, 100), build/query time, peak RAM, and extrapolated full-test runtime.
Deliverable: probes/P3-block/block.py: candidates(s1_df, sx_df, cfg) -> DataFrame[s1, sx, block_score,
block_rank_s1, block_rank_sx, via] and the recommended cfg.
```

### P7-scale — BRIEF.md

```markdown
# P7-scale: can the full run fit, and how to chunk it
Goal: an execution plan for the full test (1.73M S1, 9.97M S2/S3) on a 15 GB laptop.
Measure: parquet load RAM (pandas pyarrow strings vs polars) per source; TF-IDF char-3gram fit/transform RAM
on 5M strings; sparse_dot_topn throughput per chunk; features for ~50M candidate pairs (k≈30) must stream in
chunks — design it; LightGBM predict throughput; total wall time. If local won't fit, cost an AWS run (EC2 or
SageMaker with 64 GB, $200 credits; see docs/SAGEMAKER.md) and say which stages to move.
Deliverable: probes/P7-scale/SCALE.md (RAM and time budget per stage, chunk sizes) and a reusable
chunked-execution helper in probes/P7-scale/chunks.py.
```

### P4-match — BRIEF.md (Wave B)

```markdown
# P4-match: pair features and matcher
Inputs: adopted normalize/translit/block modules (see probes/MERGE-1.md for the exact versions).
Training pairs: candidates from the adopted blocker on the train world, labelled from truth. Validate on dev,
report on holdout.
Features: name (rapidfuzz ratio, partial_ratio, token_set, token_sort, Jaro-Winkler, token Jaccard, TF-IDF
cosine, name_core equality, acronym match, website_core vs name, transliterated variants), address (token
Jaccard, city_key, state, postcode, house number, street similarity, NULL/empty flags), context (block ranks
both directions, score gap to this S1's best candidate, #candidates, the S1's rank among this SX's candidate
S1s, name frequency within country = chain-ness), source S2/S3. Do NOT feed country as a raw category
(France is unseen) — P6 decides country handling.
Model ladder (log each): M0 hand-weighted rule score · M1 logistic regression · M2 LightGBM (expected
workhorse; research: Foursquare 1st place) · M3 Splink Fellegi-Sunter EM (label-free baseline).
Fixed decision rule for comparability: each SX goes only to its best S1, then a global threshold swept for F0.5.
Report pair AUC/AP, dev F0.5, feature importance, predict throughput (pairs/s).
Deliverable: probes/P4-match/{features.py, model.py}: featurize(pairs, s1_df, sx_df) -> DataFrame;
fit(X, y) -> model; predict(model, X) -> np.ndarray. Save out-of-fold probabilities for the dev and holdout
worlds to data/work/P4/ for P5.
```

### P6-france — BRIEF.md (Wave B)

```markdown
# P6-france: precision on an unseen country with zero labels
1. france.py: French normalization from domain knowledge — legal forms SARL/SAS/SASU/SA/EURL/SCI/SNC/EI/EIRL,
   Ets/Établissements, street types (av, bd, r, pl, imp, all, chem, rte, fg, qu), St/Ste → Saint/Sainte,
   articles l'/d'/de/la/le/du/des, accent folding (anyascii). Unit-test on handwritten examples; do not fit
   anything on test rows (open question 4).
2. Leave-one-country-out: using P4's pipeline, train on US only → score the India dev world, and the reverse;
   compare with in-country training. Find which features transfer; propose a country-agnostic feature set and
   a threshold policy for an unseen country.
3. Compare a label-free Splink Fellegi-Sunter model fitted on India (as if unlabelled) against the US-trained
   GBDT on India: which is safer for an unseen country?
4. Unsupervised France diagnostics on test outputs: score distribution, predicted matches per S1, predicted
   singleton rate vs US/India. Treat the train singleton rate (5.6%) as a prior, not a target.
Deliverable: probes/P6-france/france.py plus a France policy (features, thresholds, fallback) with the LOCO numbers.
```

### P5-decide — BRIEF.md (Wave C)

```markdown
# P5-decide: from pair probabilities to answer sets
Inputs: data/work/P4/ out-of-fold probabilities (dev, holdout) + truth.
Compare (research_techniques.md §4): D0 global threshold · D1 exclusivity first (each SX only to its argmax S1;
many-to-one, so argmax is exact — NOT Hungarian) · D2 expected-F0.5-optimal subset per S1 (Ye et al. 2012;
exact DP of Jansche / Dembczynski) on isotonic-calibrated probabilities · D3 singleton gate (empty if max p < t0)
· D4 per-country thresholds with an unseen-country default · D5 top-m cap per S1 · D6 sibling expansion
(add SX whose near-duplicate S2/S3 sibling was matched) — test, don't assume.
Tune on dev, report on holdout, per country and singleton accuracy.
Deliverable: probes/P5-decide/decide.py: decide(pairs: DataFrame[s1, sx, p], cfg) -> dict[s1, set], plus the
winning cfg.
```

### P4b-rerank — BRIEF.md (Wave C, GPU)

```markdown
# P4b-rerank: hard-pair reranker
Fine-tune a multilingual cross-encoder (xlm-roberta-base MIT, or muril-base-cased Apache-2.0, or
multilingual-e5-small MIT) on train pairs in P4's uncertain band (0.1 < p < 0.9) using the "name | address"
text of both records. GPU under `flock /tmp/amc-gpu.lock`; RTX 3050 6 GB → fp16, max_len 64, batch sized to fit.
Measure holdout F0.5 when blending reranker and GBDT scores on the band, and inference cost for the full test
band. Adopt only if the gain is >= 0.003 F0.5 and the full-test inference fits the time budget.
Deliverable: probes/P4b-rerank/rerank.py with train/predict and the blend rule.
```

---

## Step 2 — The model ladder and the experiment ledger

Every model is tried in this order, and each rung is one or more `exp` rows. "Promote" means the master copies it
into `src/` at a merge.

| Stage | Rung | What | Licence | Promote if |
|---|---|---|---|---|
| Block | B1 exact keys · B2 token index · **B3 TF-IDF char top-k** · B4 MinHash · B5 e5/LaBSE ANN · B6 union | the recall ceiling | sklearn BSD, sparse-dot-topn Apache, faiss MIT | best recall at ≤ 40 cands/S1 on the full corpus |
| Match | M0 rules · M1 logistic · **M2 LightGBM** · M3 Splink EM · M4 cross-encoder rerank · M5 Qwen2.5-1.5B / Qwen3-4B judge on a tiny residual | pair probability | lightgbm MIT, splink MIT, xlm-r MIT, Qwen Apache (only the sizes checked) | holdout F0.5 gain ≥ 0.003 with a fixed decision rule |
| Decide | D0 threshold · D1 argmax exclusivity · **D2 expected-F0.5 DP** · D3 singleton gate · D4 per-country · D5 top-m · D6 sibling expansion | answer sets | own code | holdout F0.5 gain ≥ 0.002 |
| France | country-agnostic features · LOCO-tuned threshold · Splink EM fallback | unseen country | — | smallest LOCO drop |

Reading the ledgers: `ledger.py show exp --sort f05`, `ledger.py show exp --probe P3-block`, `ledger.py show pipe`.
Each merge memo cites ledger rows by `probe/name`, so every claim traces back to its evidence.

---

## Step 4 — Merge protocol (master, after each wave)

`probes/MERGE-<n>.md`:

```markdown
# MERGE-<n> — <date time>
| Probe | Recommendation | Evidence (ledger rows) | Decision | Promoted to |
|---|---|---|---|---|
## Conflicts and how they were settled
## Current best pipeline config (the JSON that Phase 2 will run)
## Next wave: probes + one-line brief deltas
## Questions sent back to probes (session resumed)
```

Decision rules:
1. Adopt only what a ledger row shows.
2. Conflicting recommendations → holdout number wins; on a tie within 0.002, the simpler option wins.
3. Anything that breaks the licence, no-external or RAM rule is rejected, whatever its number.
4. A promoted module gets a self-check before it enters `src/`.
5. After every merge you get a 5-line checkpoint: best number, what changed, what's next, any decision needing you.

---

## Step 5 — Phase 2: base pipeline (25 Sep ~17:00 → submission #2 by 23:30)

### Task 2.1: Assemble `src/` from promoted modules and run end to end

**Files:** Create in `code/business_entity_resolution/src/`: `normalize.py`, `translit.py`, `block.py`, `features.py`,
`model.py`, `decide.py`, `run.py`, `config.json`, `test_pipeline.py`; `code/business_entity_resolution/README.md`

**Interfaces:**
- `run(s1: DataFrame, sx: DataFrame, cfg: dict) -> tuple[dict, dict]` returns `(matches, candidates)`, each `{s1_id: set(ids)}`, with one key per input S1 row and matches ⊆ candidates. Stages: normalize → block (per country, chunked per P7) → featurize (chunked) → predict → decide.
- CLI `run.py --split test|holdout|dev --config config.json --out output/` writes both TSVs (reusing `write()` from `s0_exact.py`). On a world it scores and appends one `pipe` ledger row. Before writing it asserts one row per S1, matches ⊆ candidates, only `S2-`/`S3-` ids, and no duplicates.

- [ ] **Step 1: Write `test_pipeline.py` first**

```python
"""End-to-end guard: every S1 of any country label gets exactly one row; matches ⊆ candidates."""
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run import run
from worlds import load_world
from metric import score

s1, sx, truth = load_world("dev")
fake = s1.sample(50, random_state=0).assign(country="Xland")        # an unseen country label, like France
fake["entity_id"] = "S1-X" + fake.entity_id.str[3:]
cfg = json.loads((Path(__file__).parent / "config.json").read_text())
matches, cands = run(pd.concat([s1, fake], ignore_index=True), sx, cfg)

assert set(matches) == set(s1.entity_id) | set(fake.entity_id), "every S1 row, every country, exactly once"
assert set(cands) == set(matches)
assert all(m <= cands[s] for s, m in matches.items()), "matches must be a subset of candidates"
assert all(i.startswith(("S2-", "S3-")) for m in cands.values() for i in m)
print(f"pipeline test ok  dev f05={score(matches, truth)['f05']:.4f}")
```

- [ ] **Step 2:** Run it: `.venv/bin/python code/business_entity_resolution/src/test_pipeline.py`. Expected: FAIL with `ModuleNotFoundError: No module named 'run'`.
- [ ] **Step 3:** Copy in the promoted modules and write `run.py` and `config.json` from MERGE-3's config.
- [ ] **Step 4:** Rerun the test. Expected: `pipeline test ok`.
- [ ] **Step 5:** `run.py --split holdout`: record `pipe` row `v1`.
- [ ] **Step 6:** `run.py --split test`, run the official validator, then **submission #2** (user uploads). Log `lb`.
- [ ] **Step 7:** Commit with the holdout F0.5 in the message: `git commit -m "v1 pipeline: holdout F0.5 0.xxxx, LB 0.xxxx"`.

---

## Step 5 (continued) — Phase 3: improvement loop (26 Sep 00:00 → 27 Sep 14:00)

Each idea gets one **improver probe** (Opus 5.5) in its own git worktree, so ideas never collide in `src/`:

```bash
git worktree add ../amc-26-wt/<idea> -b idea/<idea> && ln -s "$PWD/data" ../amc-26-wt/<idea>/data
```

The improver implements the idea, runs `run.py --split holdout` and logs `pipe` row `v<n>+<idea>`. It keeps the
change only if holdout F0.5 rises by ≥ 0.002. The master merges winners one at a time and re-runs holdout after
each merge, to catch ideas that fight each other. At most two full-test runs at once, never alongside another heavy
probe (RAM).

Seed backlog in `ledger/IDEAS.md`, ranked by expected gain ÷ cost. Probe results will re-rank it.
1. Union of forward and reverse top-k candidates (recall)
2. Reverse-rank and competition features: is this S1 the SX's best? (precision)
3. Expected-F0.5 DP decoding instead of a threshold (if P5 didn't already promote it)
4. Name-frequency / chain features, so common names demand address agreement
5. A second-stage LightGBM with cluster-level features (Foursquare 2-stage cascade)
6. Cross-encoder rerank on the uncertain band (if P4b > 0)
7. Splink Fellegi-Sunter score as a feature
8. Per-country thresholds plus the France policy from P6
9. Transliteration-aware TF-IDF blocking for India
10. Sibling agreement: near-duplicate S2/S3 records pointing at the same S1
11. More training pairs and hard negatives (the train world is 200k S1; raise it if RAM allows)
12. France pseudo-labels, only if the Google Form allows fitting on test

**Submission policy:**

| Day | Slots | Use |
|---|---|---|
| 25 Sep | 5 | #1 s0-exact sanity (~07:00) · #2 v1 (≤ 23:30) · 3 spare for bug fixes only |
| 26 Sep | 5 | only pipelines ≥ 0.003 above the last submitted holdout; space them out so each reading informs the next |
| 27 Sep | 5 | ≤ 3 before 18:00 · the final candidate by 22:00 with the final zip · 1 spare |

The leaderboard is a sanity check, not the objective: the private split decides. If the leaderboard and the holdout
disagree in direction twice, stop and investigate before submitting again.

---

## Phase 4 — Freeze and package (27 Sep 14:00 → 22:00)

- [ ] **14:00 freeze:** no new ideas. Choose the final pipeline by holdout F0.5, not leaderboard.
- [ ] **Reproduce from a clean clone:** `git clone . /tmp/amc-final && cd /tmp/amc-final/code/business_entity_resolution`, then follow its README exactly. Both TSVs must come out byte-identical to the submitted ones.
- [ ] **Documentation:** fill `Documentation_template.md` from the ledgers and MERGE memos: EDA (P1), blocking (P3 numbers), features + model (P4), decision (P5), France (P6), error analysis, and every model's licence and parameter count.
- [ ] **Build the zip:** `<team_name>_submission.zip` = `output/` + `code/business_entity_resolution/` + `Documentation_template.md`. Run the official validator and the pipeline's own asserts once more.
- [ ] **Final upload by 22:00 IST.** Keep 1 slot and 2 hours of buffer.

---

## Timeline at a glance (IST)

| When | Master | Probes (tmux `amc`) |
|---|---|---|
| 25 Sep 04:00–07:00 | Phase 0 foundation; submission #1 | smoke probe |
| 07:00–11:30 | watch, answer probe questions | **Wave A**: P1, P2, P3 → P7 |
| 11:30 | MERGE-1 + checkpoint to you | — |
| 12:00–16:30 | promote normalize/translit/block | **Wave B**: P4, P6 |
| 16:30 | MERGE-2 + checkpoint | — |
| 17:00–20:00 | start assembling `src/` | **Wave C**: P5, P4b |
| 20:00–23:30 | MERGE-3 → v1 → submission #2 | — |
| 26 Sep | merge winners; ≤ 5 submissions | improver probes (worktrees), 2–3 at a time |
| 27 Sep → 14:00 | last merges; ≤ 3 submissions | improvers |
| 27 Sep 14:00–22:00 | freeze, reproduce, document, zip, final upload | — |

## Risks

| Risk | Mitigation |
|---|---|
| RAM: browsers + probes OOM the laptop | Task 0.0; systemd MemoryMax per probe; `launch.sh` RAM gate; dev world first |
| A probe goes off-track or stalls | 1-hour draft RESULT.md; time boxes; master resumes the session or kills its tmux window |
| Holdout doesn't predict leaderboard | test-density worlds; full-corpus blocking recall; LB vs holdout logged side by side |
| France collapses | P6 LOCO policy; unsupervised France diagnostics; conservative unseen-country threshold |
| Licence or external-data slip | `_COMMON.md` rules; master checks every promoted dependency against `research_licences.md` |
| Probe usage cost | ~10–15 Opus 5.5 sessions at xhigh for 3–4 h each is a lot of usage; downgrade effort to `high` for mechanical probes (P7, improvers) if limits bite |
| Losing the day's 5 slots | the submission table above; the s0 sanity run happens on day 1 |
