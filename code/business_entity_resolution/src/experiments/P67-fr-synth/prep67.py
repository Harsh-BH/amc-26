"""P67 prep: the SageMaker channel data/work/p67/sm/ch/ (then s3 amc26-p67/ch/).
  frames   P55's frames as CE inputs (P3 cleaned texts, ta = S1, tb = SX; s1 = sx = frame row):
           frame_US_India (India pairs, LOCO), frame_India_US (US pairs), frame_France (P65's France frame order);
           fb_<same> = band rows only (0.02 < g < 0.98, the Qwen arm)
  train    train_run<T> = syn_<T> (gen67) + a replay slice of real labelled band pairs (P13 train file) of the
           countries the starting CE was trained on: India <- US, US <- India, France <- India + US
    .venv/bin/python probes/P67-fr-synth/prep67.py frames | train
"""
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P4b-rerank"), str(ROOT / "code/business_entity_resolution/src")]
W, SM = ROOT / "data/work", ROOT / "data/work/p67/sm"
VER = __import__("os").environ.get("P67_VER", "v2")
CH = SM / {"v1": "ch", "v2": "ch2"}.get(VER, f"ch_{VER}")   # S3 amc26-p67/<same name>/
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
REPLAY = {"India": {"US": 60_000}, "US": {"India": 60_000}, "France": {"India": 30_000, "US": 30_000}}  # v1: 25 % replay
if VER == "v2":
    REPLAY = {"India": {"US": 120_000}, "US": {"India": 120_000}, "France": {"India": 60_000, "US": 60_000}}


def _write(name, ta, tb, g):
    n = len(g)
    d = pd.DataFrame({"s1": np.arange(n, dtype=np.int64), "sx": np.arange(n, dtype=np.int64), "ta": ta, "tb": tb,
                      "p": g.astype(np.float32)})
    d.to_parquet(CH / f"frame_{name}.parquet", index=False)
    d[(g > 0.02) & (g < 0.98)].to_parquet(CH / f"fb_{name}.parquet", index=False)
    print(f"frame_{name}: {n:,} rows, band {int(((g > .02) & (g < .98)).sum()):,}; peak {peak():.2f} GB", flush=True)


def frames():
    import rerank as rr
    CH.mkdir(parents=True, exist_ok=True)
    for dr, tgt in (("US_India", "India"), ("India_US", "US")):
        ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet", columns=["q1", "qx"])
        g = np.load(W / f"p42/loco_{dr}.npz")["g"]
        assert len(ids) == len(g)
        ta, tb = rr.texts("train", tgt, ids.q1.to_numpy(), ids.qx.to_numpy())
        _write(dr, ta, tb, g)
        del ta, tb, ids
    f = pd.read_parquet(W / "p65/sm/frame_France.parquet", columns=["row", "s1", "sx", "g"])
    assert (f.row.to_numpy() == np.arange(len(f))).all()
    q1, qx = rr.rows("test", "France", f.s1, f.sx)
    ta, tb = rr.texts("test", "France", q1, qx)
    _write("France", ta, tb, f.g.to_numpy())


def train():
    CH.mkdir(parents=True, exist_ok=True)
    for t, rep in REPLAY.items():
        f = SM / f"syn_{t}_{VER}.parquet"
        if not f.exists():
            continue
        s = pd.read_parquet(f, columns=["ta", "tb", "y", "p"])
        parts = [s] + [pd.read_parquet(SM / f"train_{c}.parquet", columns=["ta", "tb", "y", "p"]).sample(n, random_state=67)
                       for c, n in rep.items()]
        d = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=67).reset_index(drop=True)
        d["y"] = d.y.astype(np.int64)
        d.to_parquet(CH / f"train_run{t}.parquet", index=False)
        print(f"train_run{t}: {len(d):,} pairs (synthetic {len(s):,}), true {d.y.mean():.3f}", flush=True)
    for c in ("India", "US"):
        shutil.copy(SM / f"val_{c}.parquet", CH / f"val_{c}.parquet")


if __name__ == "__main__":
    t0 = time.time()
    {"frames": frames, "train": train}[sys.argv[1]]()
    print(f"{time.time() - t0:.0f}s peak {peak():.2f} GB")
