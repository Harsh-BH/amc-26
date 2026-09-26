"""P45 neural matcher members (SageMaker CPU job, PyTorch DLC): an MLP on the same 65 pair features as m2_big, trained on
the same folds (hp folds, never 0/1), early stopping on the 25% S1-hash sample of fold 1, then p on #6's pruned holdout
and test keys (P22's score_split, the same pred/{holdout,test}_<c>.parquet layout as the tree members).

  inputs  : QuantileTransformer(normal) fit on 1M train rows; NaN -> 0 plus a missing flag per column that has NaNs
  mlp     : [d] -> 3 x 256 (ReLU, dropout 0.1) -> 1
  mlp2    : + an 8-dim learned embedding per integer context column (EMB, clipped to 0..63, NaN -> 63), 2 x 512, dropout 0.2
p45lib.py = solution_p45.py (load / score_split / S3 / log), shipped next to this file.
"""
import json, os, time
from pathlib import Path

import numpy as np

from p45lib import DROP, S3, files, load, log, score_split, T0

EMB = ["block_rank_s1", "block_rank_sx", "s1_n", "sx_n", "place_n_1", "place_n_x", "core_ntok_1", "core_ntok_x"]
DEFAULTS = {"name": "mlp", "arch": "mlp", "folds": [2, 3, 4, 5], "val_frac": 0.25, "frac": 1.0, "epochs": 8, "batch": 4096,
            "lr": 1e-3, "seed": 0, "patience": 2, "deadline_s": 6000, "s3": ""}


def build(arch: str, d: int, n_emb: int):
    import torch.nn as nn
    import torch

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            w, k, dr = (512, 2, 0.2) if arch == "mlp2" else (256, 3, 0.1)
            self.emb = nn.ModuleList([nn.Embedding(64, 8) for _ in range(n_emb)])
            layers, i = [], d + 8 * n_emb
            for _ in range(k):
                layers += [nn.Linear(i, w), nn.ReLU(), nn.Dropout(dr)]
                i = w
            self.body = nn.Sequential(*layers, nn.Linear(i, 1))

        def forward(self, x, e=None):
            if self.emb:
                x = torch.cat([x] + [m(e[:, j]) for j, m in enumerate(self.emb)], 1)
            return self.body(x).squeeze(1)
    return Net()


class Prep:
    """Quantile-normal inputs + missing flags (+ integer codes for the embedded columns)."""

    def __init__(self, X: np.ndarray, cols: list, arch: str, seed: int):
        from sklearn.preprocessing import QuantileTransformer
        rng = np.random.default_rng(seed)
        s = X[rng.choice(len(X), min(len(X), 1_000_000), replace=False)].astype(np.float32)
        self.qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=10**9, random_state=seed).fit(s)
        self.nan = np.where(np.isnan(s).any(0))[0]
        self.ei = [cols.index(c) for c in EMB if c in cols] if arch == "mlp2" else []
        self.d = X.shape[1] + len(self.nan)

    def __call__(self, X: np.ndarray):
        X = X.astype(np.float32)
        Z = self.qt.transform(X)
        miss = np.isnan(X[:, self.nan]).astype(np.float32)
        Z = np.nan_to_num(np.clip(Z, -5, 5), nan=0.0).astype(np.float32)
        E = np.clip(np.nan_to_num(X[:, self.ei], nan=63), 0, 63).astype(np.int64) if self.ei else None
        return np.hstack([Z, miss]), E


def main(train_dir, test_dir, model_dir, out_dir, params):
    import torch
    hp = DEFAULTS | {k: v for k, v in params.items() if k in DEFAULTS} | {k: v for k, v in (params.get("params") or {}).items() if k in DEFAULTS}
    assert not {0, 1} & set(hp["folds"]), "never train on folds 0 (holdout) / 1 (early stopping)"
    torch.manual_seed(hp["seed"])
    torch.set_num_threads(os.cpu_count())
    s3, model_dir = S3(hp["s3"]), Path(model_dir)
    fs = files(train_dir)
    cols = [c for c in __import__("pyarrow.parquet").parquet.read_schema(fs[0]).names if c not in DROP]
    Xtr, ytr = load(fs, cols, hp["folds"], frac=hp["frac"], dtype=np.float16)
    Xva, yva = load(fs, cols, (1,), frac=hp["val_frac"])
    prep = Prep(Xtr, cols, hp["arch"], hp["seed"])
    log(f"loaded {len(ytr):,} train / {len(yva):,} val pairs, {len(cols)} feats -> d {prep.d}, emb {len(prep.ei)}")
    CH = 1_000_000
    Ztr = np.empty((len(Xtr), prep.d), np.float16)
    Etr = np.empty((len(Xtr), len(prep.ei)), np.int8) if prep.ei else None
    for a in range(0, len(Xtr), CH):
        z, e = prep(Xtr[a:a + CH])
        Ztr[a:a + CH] = z
        if prep.ei:
            Etr[a:a + CH] = e
    del Xtr
    Zva, Eva = prep(Xva)
    net = build(hp["arch"], prep.d, len(prep.ei))
    steps = hp["epochs"] * (len(ytr) // hp["batch"])
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], total_steps=steps, pct_start=0.05)
    lossf = torch.nn.BCEWithLogitsLoss()
    yt = torch.from_numpy(ytr.astype(np.float32))

    def logits(Z, E, bs=65536):
        net.eval()
        out = []
        with torch.no_grad():
            for a in range(0, len(Z), bs):
                out.append(net(torch.from_numpy(Z[a:a + bs].astype(np.float32)),
                               None if E is None else torch.from_numpy(E[a:a + bs].astype(np.int64))).numpy())
        net.train()
        return np.concatenate(out)

    def vloss():
        z = logits(Zva, Eva)
        return float(np.mean(np.logaddexp(0, z) - yva * z))

    best, bad, half = (np.inf, None, -1), 0, len(ytr) // hp["batch"] // 2
    rng, step, info = np.random.default_rng(hp["seed"]), 0, {"name": hp["name"], "arch": hp["arch"], "train_pairs": len(ytr)}
    for ep in range(hp["epochs"]):
        perm = rng.permutation(len(ytr))
        for bi in range(len(ytr) // hp["batch"]):
            idx = np.sort(perm[bi * hp["batch"]:(bi + 1) * hp["batch"]])
            xb = torch.from_numpy(Ztr[idx].astype(np.float32))
            eb = None if Etr is None else torch.from_numpy(Etr[idx].astype(np.int64))
            loss = lossf(net(xb, eb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            step += 1
            if (bi + 1) % half == 0:                    # eval twice per epoch
                v = vloss()
                if v < best[0]:
                    best, bad = (v, {k: t.clone() for k, t in net.state_dict().items()}, step), 0
                else:
                    bad += 1
                log(f"epoch {ep} step {step} val_loss={v:.6f} best {best[0]:.6f} @ {best[2]}")
                s3.put_json(info | {"state": "training", "epoch": ep, "val_loss": v, "best": best[0]}, "status.json")
                if bad >= 2 * hp["patience"] or time.time() - T0 > hp["deadline_s"]:
                    break
        else:
            continue
        break
    net.load_state_dict(best[1])
    info |= {"best_val_loss": best[0], "best_step": best[2], "steps": step, "train_s": round(time.time() - T0)}
    log(f"trained {info}; val_loss={best[0]:.6f}")
    torch.save({"state": best[1], "hp": hp, "cols": cols}, model_dir / f"{hp['name']}.pt")
    s3.put(model_dir / f"{hp['name']}.pt", f"{hp['name']}.pt")
    del Ztr, Etr

    def predict(X):
        z, e = prep(X)
        return 1 / (1 + np.exp(-logits(z, e)))
    pred = model_dir / "pred"
    info["scored"] = score_split(predict, fs, Path(test_dir) / "keys", "holdout", cols, pred)
    info["scored"] |= score_split(predict, files(test_dir, "test"), Path(test_dir) / "keys", "test", cols, pred)
    for f in sorted(pred.glob("*.parquet")):
        s3.put(f, f"pred/{f.name}")
    res = info | {"state": "done", "elapsed_s": round(time.time() - T0)}
    (model_dir / "result.json").write_text(json.dumps(res))
    s3.put_json(res, "status.json")
    log(f"done {res}")
