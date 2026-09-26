"""P45 pretrained bi-encoder similarities on EVERY holdout pair (SageMaker GPU job; shipped alone as solution.py).
intfloat/multilingual-e5-small (MIT, 118M), frozen, no fine-tuning: "query: " + text, mean pooling, L2 norm, fp16.
Channel 'train' = data/work/p45/ce/holdout_<c>.parquet (s1, sx, ta, tb = P3's "name | addr"). Each unique S1 / SX
text is encoded once per field (name, addr, full "name | addr"); per pair cosines -> holdout_e5_<c>.parquet
(s1, sx, e_name, e_addr, e_full; e_addr NaN when either address is empty or NULL) in model_dir and hp['s3'].
"""
import json, os, time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
DEFAULTS = {"model": "intfloat/multilingual-e5-small", "bs": 1024, "maxlen": 64, "s3": "", "countries": ["India", "US"]}
NULLS = {"", "null", "<null>", "none", "nan"}


def log(msg):
    print(f"[p45e {time.time() - T0:6.0f}s] {msg}", flush=True)


def split(t: pd.Series):
    """P3's join is name + " | " + addr; a name may itself hold " | " (websites), an address practically never."""
    s = t.str.rsplit(" | ", n=1, expand=True)
    return s[0].fillna(""), s[1].fillna("") if s.shape[1] > 1 else pd.Series("", index=t.index)


def main(train_dir, test_dir, model_dir, out_dir, params):
    import torch
    from transformers import AutoModel, AutoTokenizer
    hp = DEFAULTS | {k: v for k, v in params.items() if k in DEFAULTS}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(hp["model"])
    net = AutoModel.from_pretrained(hp["model"]).to(dev).eval()
    if dev == "cuda":
        net = net.half()
    log(f"{hp['model']} on {dev}")

    def enc(texts: np.ndarray) -> np.ndarray:
        o = np.argsort([len(x) for x in texts], kind="stable")          # length-sorted batches
        out = np.empty((len(texts), net.config.hidden_size), np.float16)
        with torch.no_grad():
            for a in range(0, len(o), hp["bs"]):
                j = o[a:a + hp["bs"]]
                b = tok(["query: " + texts[i] for i in j], padding=True, truncation=True, max_length=hp["maxlen"], return_tensors="pt").to(dev)
                h = net(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                e = (h * m).sum(1) / m.sum(1).clamp(min=1)
                out[j] = torch.nn.functional.normalize(e.float(), dim=1).cpu().numpy().astype(np.float16)
        return out

    import boto3
    for c in hp["countries"]:
        d = pd.read_parquet(Path(train_dir) / f"holdout_{c}.parquet", columns=["s1", "sx", "ta", "tb"])
        res = {"s1": d.s1.to_numpy(), "sx": d.sx.to_numpy()}
        u1 = d.drop_duplicates("s1")[["s1", "ta"]]
        ux = d.drop_duplicates("sx")[["sx", "tb"]]
        i1 = pd.Index(u1.s1).get_indexer(d.s1)
        ix = pd.Index(ux.sx).get_indexer(d.sx)
        n1, a1 = split(u1.ta)
        nx, ax = split(ux.tb)
        log(f"{c}: {len(d):,} pairs, {len(u1):,} S1, {len(ux):,} SX")
        for f, t1, tx in (("name", n1, nx), ("addr", a1, ax), ("full", u1.ta, ux.tb)):
            E1, EX = enc(t1.to_numpy(object)), enc(tx.to_numpy(object))
            cos = np.einsum("ij,ij->i", E1[i1].astype(np.float32), EX[ix].astype(np.float32))
            if f == "addr":
                bad = a1.str.strip().str.lower().isin(NULLS).to_numpy()[i1] | ax.str.strip().str.lower().isin(NULLS).to_numpy()[ix]
                cos[bad] = np.nan
            res[f"e_{f}"] = cos.astype(np.float32)
            log(f"{c} {f}: encoded {len(t1):,} + {len(tx):,}")
            del E1, EX
        out = Path(model_dir) / f"holdout_e5_{c}.parquet"
        pd.DataFrame(res).to_parquet(out, index=False)
        if hp["s3"]:
            b, k = hp["s3"][5:].rstrip("/").split("/", 1)
            boto3.client("s3").upload_file(str(out), b, f"{k}/{out.name}")
        log(f"{c} done")
    log("done")
