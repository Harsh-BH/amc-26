"""P65 SageMaker job: ByT5 corrector. Fine-tune google/byt5-small (Apache-2.0, ~300M) as a byte-level seq2seq
SX "name | addr" -> its S1 "name | addr" on train_<c>.parquet (prep65.py), then greedy-decode the sx_txt of each
params['decode'] frame file. Shipped alone as solution.py; sm_train calls main().

Order (a crash keeps what is done; every output also goes to S3 the moment it exists):
  weights (S3 cache, else HF -> cached to S3) -> train (at `smoke` steps: decode 256 rows of the first decode file ->
  dec_<file>_smoke.parquet) -> save bf16 weights -> decode each file in chunks (band rows first, then hi, then lo; length-sorted batches) -> dec_<file>_<k>.parquet (row, out). Stops cleanly at deadline_s.
Byte ids are ByT5's own (utf-8 byte + 3, pad 0, eos 1), built with numpy instead of the tokenizer.
"""
import json, os, threading, time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
DEFAULTS = {"model": "google/byt5-small", "weights_s3": "", "tag": "p65", "train": ["US"], "decode": [],
            "n_train": 0, "lin": 192, "lout": 160, "bs": 64, "lr": 5e-4, "warm": 300, "epochs": 1, "cap_s": 1800,
            "smoke": 200, "dec_bs": 512, "chunk": 40000, "deadline_s": 6600, "s3": ""}


def log(ev, **kw):
    print(json.dumps({"t": round(time.time() - T0), "ev": ev, **kw}), flush=True)


class S3:
    def __init__(self, uri):
        self.on = bool(uri)
        if self.on:
            import boto3
            self.bucket, self.prefix = uri[5:].rstrip("/").split("/", 1)
            self.c = boto3.client("s3")

    def put(self, path, key):
        if not self.on:
            return
        for i in range(3):
            try:
                return self.c.upload_file(str(path), self.bucket, f"{self.prefix}/{key}")
            except Exception as e:
                log("s3_fail", key=key, err=repr(e))
                time.sleep(5 * (i + 1))


def fetch_weights(name, cache_uri, local: Path):
    import boto3
    local.mkdir(parents=True, exist_ok=True)
    if cache_uri:
        b, pre = cache_uri[5:].rstrip("/").split("/", 1)
        c = boto3.client("s3")
        keys = [o["Key"] for pg in c.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=pre + "/")
                for o in pg.get("Contents", [])]
        if any(k.endswith("/config.json") for k in keys) and any(k.endswith((".bin", ".safetensors")) for k in keys):
            for k in keys:
                f = local / k[len(pre) + 1:]
                f.parent.mkdir(parents=True, exist_ok=True)
                c.download_file(b, k, str(f))
            log("weights_s3", n=len(keys))
            return local, None
    from huggingface_hub import snapshot_download
    for i in range(3):
        try:
            snapshot_download(name, local_dir=str(local), allow_patterns=["*.json", "*.safetensors", "pytorch_model.bin"])
            break
        except Exception as e:
            log("hf_fail", attempt=i, err=repr(e))
            if i == 2:
                raise
            time.sleep(20)
    log("weights_hf", files=sorted(p.name for p in local.iterdir()))
    if not cache_uri:
        return local, None

    def up():                                   # config.json last: readers take it as "cache complete"
        s3 = S3(cache_uri)
        fs = [f for f in sorted(local.rglob("*")) if f.is_file() and ".cache" not in f.parts]
        for f in sorted(fs, key=lambda f: f.name == "config.json"):
            s3.put(f, f.relative_to(local).as_posix())
        log("weights_cached", uri=cache_uri)

    t = threading.Thread(target=up)
    t.start()
    return local, t


def enc(strs, L):
    """ByT5 ids: utf-8 bytes + 3, cut to L - 1, then eos (1); right-padded with 0. -> int64 [n, max len]"""
    bs = [s.encode("utf-8")[:L - 1] for s in strs]
    m = max(len(b) for b in bs) + 1
    out = np.zeros((len(bs), m), np.int64)
    for i, b in enumerate(bs):
        out[i, :len(b)] = np.frombuffer(b, np.uint8).astype(np.int64) + 3
        out[i, len(b)] = 1
    return out


def dec(ids):
    """Inverse of enc for generated rows (drops pad / eos / unk and anything past eos; bad utf-8 bytes dropped)."""
    res = []
    for r in ids:
        r = np.asarray(r)
        e = np.where(r == 1)[0]
        r = r[:e[0]] if len(e) else r
        r = r[r >= 3] - 3
        res.append(bytes(r.astype(np.uint8).tolist()).decode("utf-8", errors="ignore"))
    return res


def build(p, wdir):
    import torch
    from transformers import T5Config, T5ForConditionalGeneration
    if p["model"] == "tiny":                    # local code-path test only
        cfg = T5Config(vocab_size=384, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_decoder_layers=1, num_heads=2,
                       decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
        return T5ForConditionalGeneration(cfg)
    return T5ForConditionalGeneration.from_pretrained(str(wdir), torch_dtype=torch.float32)


def generate(model, strs, p, dev):
    import torch
    x = torch.from_numpy(enc(strs, p["lin"])).to(dev)
    with torch.no_grad():
        o = model.generate(input_ids=x, attention_mask=(x != 0).long(), max_new_tokens=p["lout"], num_beams=1,
                           do_sample=False)
    return dec(o[:, 1:].cpu().numpy())         # o[:, 0] is the decoder start (pad)


def decode_file(model, f: Path, out: Path, s3, p, dev, tag, rows=None):
    d = pd.read_parquet(f, columns=["row", "g", "sx_txt"])
    if rows is not None:
        d = d.iloc[:rows]
    g = d.g.to_numpy()
    pri = np.where((g > 0.02) & (g < 0.98), 2, np.where(g >= 0.98, 1, 0))   # band rows first, then hi, then lo
    d = d.iloc[np.argsort(-pri, kind="stable")].reset_index(drop=True)
    for k, s in enumerate(range(0, len(d), p["chunk"])):
        if time.time() - T0 > p["deadline_s"]:
            log("deadline", file=f.stem, chunk=k)
            return False
        c = d.iloc[s:s + p["chunk"]]
        o = np.argsort(c.sx_txt.str.len().to_numpy(), kind="stable")
        txt = c.sx_txt.to_numpy()[o]
        res, t1 = [], time.time()
        for b in range(0, len(txt), p["dec_bs"]):
            res += generate(model, list(txt[b:b + p["dec_bs"]]), p, dev)
        r = pd.DataFrame({"row": c.row.to_numpy()[o], "out": res})
        name = f"dec_{f.stem}_{'smoke' if rows else k}.parquet"
        r.to_parquet(out / name, index=False)
        s3.put(out / name, f"dec/{name}")
        log("decoded", file=name, n=len(r), s=round(time.time() - t1), rate=round(len(r) / (time.time() - t1), 1))
    return True


def main(train_dir, test_dir, model_dir, out_dir, params):
    import torch
    os.environ.setdefault("HF_HOME", str(Path(params.get("checkpoint_dir", "/tmp")) / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    p = {**DEFAULTS, **{k: v for k, v in params.items() if k in DEFAULTS}}
    train_dir, model_dir = Path(train_dir), Path(model_dir)
    out = model_dir / "dec"
    out.mkdir(parents=True, exist_ok=True)
    s3 = S3(p["s3"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("start", p=p, torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "-")
    up = None
    wdir = None
    if p["model"] != "tiny":
        wdir, up = fetch_weights(p["model"], p["weights_s3"], Path(params.get("checkpoint_dir", "/tmp")) / "weights")
    model = build(p, wdir).to(dev)
    log("model", params_m=round(sum(x.numel() for x in model.parameters()) / 1e6, 1))

    d = pd.concat([pd.read_parquet(train_dir / f"train_{c}.parquet") for c in p["train"]] or
                  [pd.DataFrame({"src": [], "tgt": []})], ignore_index=True)          # train [] = decode only
    if p["n_train"]:
        d = d.iloc[:p["n_train"]]
    d = d.sample(frac=1.0, random_state=0).reset_index(drop=True)
    src, tgt = d.src.to_numpy(), d.tgt.to_numpy()
    n_steps = p["epochs"] * len(d) // p["bs"]
    if not n_steps:
        first = None
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / p["warm"]) * max(0.05, 1 - s / max(n_steps, 1)))
    model.train()
    first = train_dir / f"{p['decode'][0]}.parquet" if p["decode"] and n_steps else None
    t1, ema = time.time(), None
    for step in range(n_steps):
        if time.time() - t1 > p["cap_s"]:
            log("train_cap", step=step)
            break
        i = np.arange(step * p["bs"], (step + 1) * p["bs"]) % len(d)
        x = torch.from_numpy(enc(src[i], p["lin"])).to(dev)
        y = torch.from_numpy(enc(tgt[i], p["lout"])).to(dev)
        y[y == 0] = -100
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            loss = model(input_ids=x, attention_mask=(x != 0).long(), labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 200 == 0:
            log("train", step=step, of=n_steps, loss=round(ema, 4), lr=sched.get_last_lr()[0],
                sps=round((step + 1) / (time.time() - t1), 2))
        if step == p["smoke"] and first is not None:
            model.eval()
            decode_file(model, first, out, s3, {**p, "chunk": 256}, dev, p["tag"], rows=256)
            model.train()
    log("trained", s=round(time.time() - t1), loss=ema)
    model.eval()
    if dev == "cuda":
        model = model.to(torch.bfloat16)
    if p["model"] != "tiny" and n_steps:
        sd = model_dir / "model"
        model.save_pretrained(str(sd))
        for f in sd.iterdir():
            s3.put(f, f"model/{f.name}")
        log("saved")
    for name in p["decode"]:
        if not decode_file(model, train_dir / f"{name}.parquet", out, s3, p, dev, p["tag"]):
            break
    if up is not None:
        up.join()
    log("done")


if __name__ == "__main__":                     # local code-path test: tiny random T5 on CPU
    import sys, tempfile
    td = Path(tempfile.mkdtemp())
    pd.DataFrame({"src": ["acme corp | 1 main st, x", "bar é | रमेश road"] * 40,
                  "tgt": ["Acme Corp | 1 Main St, X", "Bar E | Ramesh Road"] * 40}).to_parquet(td / "train_US.parquet")
    pd.DataFrame({"row": np.arange(6, dtype=np.int32), "g": np.array([.1, .999, .5, .9999, .2, .3], np.float32),
                  "sx_txt": ["a | b", "ééé | c", "x" * 300, "", "रमेश | d", "q | r"]}).to_parquet(td / "frame_US_India.parquet")
    main(td, td, td, td, {"model": "tiny", "train": ["US"], "decode": ["frame_US_India"], "bs": 8, "smoke": 3,
                          "chunk": 4, "dec_bs": 3, "lout": 12})
    got = pd.concat([pd.read_parquet(f) for f in sorted((td / "dec").glob("dec_frame_US_India_[0-9]*.parquet"))])
    assert sorted(got.row) == list(range(6)), got
    assert (td / "dec/dec_frame_US_India_smoke.parquet").exists()
    assert dec(enc(["héllo | रमेश"], 64)) == ["héllo | रमेश"]
    print("selfcheck ok", got.to_dict("records")[:2])
