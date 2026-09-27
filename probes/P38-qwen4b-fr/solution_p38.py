"""P38 SageMaker job: Qwen3-Reranker-4B (Apache-2.0, 4.02B params) + LoRA as a pair cross-encoder on m2_big's
uncertain band. Shipped alone as solution.py next to sm_train.py; sm_train calls main(train_dir, test_dir, model_dir,
out_dir, params). P13's solution_p13.py, cut down to one model per job, plus multi-GPU DDP.

One job = one model. Fine-tune on train_<c> for c in params['train'] (LOCO: one country, pool: both), capped in
seconds, then score params['score'] files. With params['ckpt'] (a save() file) and train [], score only.
Order -- every r file goes to model_dir/r AND S3 the moment it exists, so a crash keeps what is done:
  weights (S3 cache, else HF -> cached to S3) -> one process per GPU -> selfcheck (batched == one-at-a-time logits)
  -> smoke (score + write `smoke` pairs of every score file: the write / S3 path runs in minute 5, not hour 3; also
  the zero-shot AUC) -> train (at 50%: score + write the 'mid' files; LoRA saved every `save_s`) -> save LoRA + head
  -> score files -> val AUC.
Score = the causal LM's last-token logit(yes) - logit(no) (Qwen3-Reranker's own score) as a trainable 2560-row head
+ bias on the base model's last hidden state. Base weights frozen bf16; LoRA on every attention / MLP projection.
DDP: `bs` is the GLOBAL batch (each rank takes every world-th pair of it), so any GPU count trains the same recipe;
scoring shards length-sorted pairs round-robin and all_gathers. Every collective is reached by all ranks at the same
step (step count + stop flag come from rank 0), and a training OOM is fatal under DDP (a skipped batch on one rank
would hang the others).
Channel 'train' (P13's sm_data): train_<c>, val_<c> (ta, tb, y, p), holdout_<c> (+ s1, sx), test_<c> (s1, sx, ta,
tb, p). Output r files: (s1, sx, r = sigmoid(logit)); log.jsonl = one JSON line per event (rank 0).
"""
import datetime, gc, json, math, os, threading, time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
DEFAULTS = {
    "model": "Qwen/Qwen3-Reranker-4B",
    "weights_s3": "",
    "tag": "qwen",
    "train": ["US"],
    "score": ["holdout_India", "holdout_US"],
    "mid": ["holdout_India", "holdout_US"],
    "val": ["val_India", "val_US"],
    "ckpt": "",
    "gpus": 0,
    "maxlen": 160,
    "bs": 128,
    "micro": 32,
    "lr": 1e-4,
    "rank": 16,
    "alpha": 32,
    "epochs": 1,
    "cap_s": 9000,
    "save_s": 1200,
    "gc": False,
    "score_bs": 256,
    "score_speed": 0,
    "deadline_s": 12000,
    "smoke": 512,
    "s3": "",
}
assert len(DEFAULTS) == 24, sorted(DEFAULTS)          # P13's crash: a '#' comment swallowed two keys
PRE = "<|im_start|>user\n"                            # no system turn: 39 tokens of every pair, P38 fine-tunes anyway
SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INST = "Are the Query and the Document records of the same business?"
LORA = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class S3:
    """Best-effort uploads under s3://bucket/prefix/; '' disables. A failed upload is logged, never fatal (the same
    files also land in model_dir -> model.tar.gz)."""

    def __init__(self, uri: str):
        self.on = bool(uri)
        if self.on:
            import boto3
            self.bucket, self.prefix = uri[5:].rstrip("/").split("/", 1)
            self.c = boto3.client("s3")

    def put(self, path, key: str):
        if not self.on:
            return
        for i in range(3):
            try:
                return self.c.upload_file(str(path), self.bucket, f"{self.prefix}/{key}")
            except Exception as e:                        # network / throttling: retry, then give up quietly
                print(f"[s3] {key} attempt {i + 1} failed: {e!r}", flush=True)
                time.sleep(5 * (i + 1))


def fetch_weights(name, cache_uri, local: Path):
    """Model files -> local dir: from the S3 cache if it holds config.json, else from HF (then copied to the cache in
    a background thread, best effort). Returns (local dir, the upload thread or None)."""
    import boto3
    local.mkdir(parents=True, exist_ok=True)
    if cache_uri:
        b, pre = cache_uri[5:].rstrip("/").split("/", 1)
        c = boto3.client("s3")
        keys = [o["Key"] for pg in c.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=pre + "/")
                for o in pg.get("Contents", [])]
        if any(k.endswith("/config.json") for k in keys):
            for k in keys:
                f = local / k[len(pre) + 1:]
                f.parent.mkdir(parents=True, exist_ok=True)
                c.download_file(b, k, str(f))
            print(f"[p38] weights from S3 cache: {len(keys)} files, {time.time() - T0:.0f}s", flush=True)
            return local, None
    from huggingface_hub import snapshot_download
    for i in range(3):
        try:
            snapshot_download(name, local_dir=str(local))
            break
        except Exception as e:
            print(f"[p38] HF download attempt {i + 1} failed: {e!r}", flush=True)
            if i == 2:
                raise
            time.sleep(20)
    print(f"[p38] weights from HF, {time.time() - T0:.0f}s", flush=True)
    if not cache_uri:
        return local, None

    def up():                                             # config.json last: readers take it as "cache complete"
        s3 = S3(cache_uri)
        fs = [f for f in sorted(local.rglob("*")) if f.is_file() and ".cache" not in f.parts]
        for f in sorted(fs, key=lambda f: f.name == "config.json"):
            s3.put(f, f.relative_to(local).as_posix())
        print(f"[p38] weights cached to {cache_uri}", flush=True)

    t = threading.Thread(target=up)
    t.start()
    return local, t


class Job:
    def __init__(self, train_dir, model_dir, params, rank=0, world=1):
        import torch
        extra = set(params) - set(DEFAULTS) - {"checkpoint_dir", "_weights_dir"}
        assert not extra, f"unknown params {sorted(extra)}"   # a typo in --hp must not silently use a default
        self.torch, self.P = torch, {**DEFAULTS, **params}
        self.rank, self.world, self.main = rank, world, rank == 0
        self.dev = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        self.cuda = self.dev != "cpu"
        torch.backends.cuda.matmul.allow_tf32 = True
        self.src, self.out = Path(train_dir), Path(model_dir)
        (self.out / "r").mkdir(parents=True, exist_ok=True)
        self.ck = Path(params.get("checkpoint_dir", "/tmp")) / "p38"
        self.ck.mkdir(parents=True, exist_ok=True)
        self.s3 = S3(self.P["s3"] if self.main else "")
        self.deadline = T0 + float(self.P["deadline_s"])
        self.D, self.T = {}, {}
        self.speed = 150.0                                 # score pairs/s (all ranks) until measured (smoke)

    # ------------------------------------------------------------------ bookkeeping
    def log(self, **kw):
        if not self.main:
            return
        kw = {"t": round(time.time() - T0), **kw}
        if self.cuda:
            kw["gpu_gb"] = round(self.torch.cuda.max_memory_allocated(self.dev) / 1e9, 2)
        print("[p38] " + json.dumps(kw, default=str), flush=True)
        with open(self.out / "log.jsonl", "a") as f:
            f.write(json.dumps(kw, default=str) + "\n")

    def flush(self):
        if self.main:
            self.s3.put(self.out / "log.jsonl", "log.jsonl")

    def left(self):
        return self.deadline - time.time()

    def data(self, name):
        if name not in self.D:
            self.D[name] = pd.read_parquet(self.src / f"{name}.parquet")
        return self.D[name]

    # ------------------------------------------------------------------ collectives (no-ops on one process)
    def barrier(self):
        if self.world > 1:
            self.torch.distributed.barrier()

    def bcast(self, x: float) -> float:
        """rank 0's value of x on every rank."""
        if self.world == 1:
            return x
        t = self.torch.tensor([float(x)], dtype=self.torch.float64, device=self.dev)
        self.torch.distributed.broadcast(t, 0)
        return float(t.item())

    def reduce(self, x: float, op: str) -> float:
        if self.world == 1:
            return x
        dist = self.torch.distributed
        t = self.torch.tensor([float(x)], dtype=self.torch.float64, device=self.dev)
        dist.all_reduce(t, op={"min": dist.ReduceOp.MIN, "max": dist.ReduceOp.MAX}[op])
        return float(t.item())

    # ------------------------------------------------------------------ model
    def load(self):
        torch, P = self.torch, self.P
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = P.get("_weights_dir") or P["model"]
        tok = AutoTokenizer.from_pretrained(name)
        lm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if self.cuda else torch.float32)
        yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
        W = lm.get_output_embeddings().weight
        w = (W[yes] - W[no]).detach().float().clone()
        body = lm.model
        del lm, W
        gc.collect()
        body.requires_grad_(False)
        body.config.use_cache = False
        if P["gc"]:
            body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        r, scale = int(P["rank"]), float(P["alpha"]) / int(P["rank"])
        torch.manual_seed(0)                               # identical LoRA init on every rank

        class LoRA(torch.nn.Module):
            def __init__(s, base):
                super().__init__()
                s.base = base
                s.a = torch.nn.Parameter(torch.empty(r, base.in_features))
                torch.nn.init.kaiming_uniform_(s.a, a=math.sqrt(5))
                s.b = torch.nn.Parameter(torch.zeros(base.out_features, r))

            def forward(s, x):
                return s.base(x) + (x @ s.a.to(x.dtype).T) @ s.b.to(x.dtype).T * scale

        n_lora = 0
        for mod in list(body.modules()):
            for k in LORA:
                if isinstance(getattr(mod, k, None), torch.nn.Linear):
                    setattr(mod, k, LoRA(getattr(mod, k)))
                    n_lora += 1

        class Scorer(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.m, s.w, s.bias = body, torch.nn.Parameter(w), torch.nn.Parameter(torch.zeros(()))

            def forward(s, x, msk):
                h = s.m(input_ids=x, attention_mask=msk, use_cache=False).last_hidden_state
                h = h[torch.arange(len(h), device=h.device), msk.sum(1) - 1]
                with torch.autocast(h.device.type, enabled=False):   # the head in fp32, not autocast's bf16
                    return h.float() @ s.w + s.bias

        sc = Scorer().to(self.dev)                        # LoRA + head stay fp32 (autocast runs them in bf16)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.log(ev="loaded", name=P["model"], lora_modules=n_lora, yes_no=[yes, no], world=self.world,
                 params_m=round(sum(p.numel() for p in sc.parameters()) / 1e6, 1),
                 trainable_m=round(sum(p.numel() for p in sc.parameters() if p.requires_grad) / 1e6, 2))
        assert n_lora > 0 and None not in (yes, no)
        return tok, sc, pad

    def tokens(self, tok, name, rows=None):
        """Texts of data file <name> -> (flat int32 ids, int64 offsets): PRE + instruct/query/document + SUF, the body
        truncated so the suffix (where the score is read) always survives. Cached per file (rows: a prefix slice)."""
        key = (name, rows)
        if key in self.T:
            return self.T[key]
        t0, d = time.time(), self.data(name)
        if rows:
            d = d.iloc[:rows]
        ta, tb = d.ta.tolist(), d.tb.tolist()
        pre, suf = tok.encode(PRE, add_special_tokens=False), tok.encode(SUF, add_special_tokens=False)
        room = int(self.P["maxlen"]) - len(pre) - len(suf)
        ids, off = [], [np.zeros(1, np.int64)]
        for i in range(0, len(ta), 50_000):
            body = [f"<Instruct>: {INST}\n<Query>: {a}\n<Document>: {b}" for a, b in zip(ta[i:i + 50_000], tb[i:i + 50_000])]
            e = [pre + x + suf for x in tok(body, truncation=True, max_length=room, add_special_tokens=False,
                                            return_attention_mask=False)["input_ids"]]
            ids.append(np.fromiter((t for x in e for t in x), np.int32))
            off.append(np.cumsum([len(x) for x in e]) + off[-1][-1])
        self.T[key] = (np.concatenate(ids), np.concatenate(off))
        ln = np.diff(self.T[key][1])
        self.log(ev="tokens", file=name, pairs=len(ln), mean_len=round(float(ln.mean()), 1), max_len=int(ln.max()),
                 s=round(time.time() - t0, 1))
        return self.T[key]

    def batch(self, ids, off, j, pad):
        torch = self.torch
        ln = (off[j + 1] - off[j]).astype(np.int64)
        L = int(ln.max())
        x = np.full((len(j), L), pad, np.int64)
        msk = np.arange(L)[None, :] < ln[:, None]
        x[msk] = ids[np.concatenate([np.arange(off[k], off[k + 1]) for k in j])]
        return torch.from_numpy(x).to(self.dev), torch.from_numpy(msk.astype(np.int64)).to(self.dev)

    def ac(self):
        return self.torch.autocast("cuda" if self.cuda else "cpu", dtype=self.torch.bfloat16, enabled=self.cuda)

    def selfcheck(self, tok, sc, pad, name):
        """Batched right-padded logits of 16 pairs == one-pair-at-a-time unpadded forwards (catches a wrong last-token
        gather / mask). A failure raises: every score after it would be garbage. Local to each rank."""
        torch = self.torch
        ids, off = self.tokens(tok, name, rows=16)
        j = np.arange(len(off) - 1)
        sc.eval()
        with torch.no_grad(), self.ac():
            b = sc(*self.batch(ids, off, j, pad)).float().cpu().numpy()
            one = np.array([float(sc(*self.batch(ids, off, np.array([k]), pad))) for k in j])
        dz = float(np.abs(b - one).max())
        self.log(ev="selfcheck", max_dlogit=round(dz, 4), logit_mean=round(float(one.mean()), 3),
                 logit_std=round(float(one.std()), 3))
        self.flush()
        assert dz < 1.0, f"selfcheck failed: batched vs single logits differ by {dz}"

    # ------------------------------------------------------------------ scoring
    def score(self, tok, sc, pad, name, rows=None):
        """sigmoid(logit) for every pair of data file <name> (or its first `rows`). Length-sorted pairs, dealt
        round-robin to the ranks (balanced lengths), all_gathered; halves the batch on OOM (inference: rank-local,
        no collective inside). Returns (r on every rank, pairs/s)."""
        torch = self.torch
        ids, off = self.tokens(tok, name, rows)
        n, bs, t0 = len(off) - 1, int(self.P["score_bs"]), time.time()
        mine = np.argsort(np.diff(off), kind="stable")[self.rank::self.world]
        rl = np.empty(len(mine), np.float32)
        sc.eval()
        i = 0
        with torch.no_grad(), self.ac():
            while i < len(mine):
                j = mine[i:i + bs]
                try:
                    rl[i:i + len(j)] = torch.sigmoid(sc(*self.batch(ids, off, j, pad))).float().cpu().numpy()
                    i += len(j)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                    print(f"[p38] rank {self.rank} score_oom bs {bs}", flush=True)
        r = np.full(n, np.nan, np.float32)
        if self.world == 1:
            r[mine] = rl
        else:
            m = -(-n // self.world)
            buf = torch.full((m,), float("nan"), dtype=torch.float32, device=self.dev)
            buf[:len(rl)] = torch.from_numpy(rl).to(self.dev)
            got = [torch.empty_like(buf) for _ in range(self.world)]
            torch.distributed.all_gather(got, buf)
            o = np.argsort(np.diff(off), kind="stable")
            for k, g in enumerate(got):
                idx = o[k::self.world]
                r[idx] = g.cpu().numpy()[:len(idx)]
        assert not np.isnan(r).any(), f"{name}: {int(np.isnan(r).sum())} pairs unscored"
        sp = n / max(time.time() - t0, 1e-6)
        self.log(ev="scored", file=name, pairs=n, pairs_s=round(sp), s=round(time.time() - t0))
        return r, sp

    def score_write(self, tok, sc, pad, name, tag, rows=None):
        """Score <name> (all ranks), then on rank 0: AUCs where it has labels (holdout / val: p, r, logistic stack) and
        r_<tag>_<name> for files with s1/sx. Returns the measured pairs/s (every rank)."""
        r, sp = self.score(tok, sc, pad, name, rows)
        if self.main:
            d = self.data(name).iloc[:len(r)]
            if "y" in d and d.y.nunique() == 2:
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import roc_auc_score
                y, p = d.y.to_numpy(), d.p.to_numpy()
                X = np.c_[_logit(p), _logit(r)]
                st = LogisticRegression(C=1.0).fit(X, y).decision_function(X)
                self.log(ev="auc", tag=tag, file=name, pairs=len(r), auc_p=round(roc_auc_score(y, p), 4),
                         auc_r=round(roc_auc_score(y, r), 4), auc_stack=round(roc_auc_score(y, st), 4))
            if "s1" in d:
                f = self.out / "r" / f"r_{tag}_{name}.parquet"
                pd.DataFrame({"s1": d.s1.to_numpy(), "sx": d.sx.to_numpy(), "r": r}).to_parquet(f, index=False)
                self.s3.put(f, f"r/{f.name}")
                self.log(ev="wrote", file=f.name, rows=len(r))
            self.flush()
        self.barrier()
        return sp

    # ------------------------------------------------------------------ training
    def fit_micro(self, sc, micro, L):
        """Largest per-rank micro-batch (<= micro) whose max-length forward/backward fits the GPU, on the bare model
        (no DDP hooks, no collective), then the min over ranks."""
        torch = self.torch
        while self.cuda:
            try:
                x = torch.full((micro, L), 5, dtype=torch.long, device=self.dev)
                with self.ac():
                    loss = sc(x, torch.ones_like(x)).float().mean()
                loss.backward()
                break
            except torch.cuda.OutOfMemoryError:
                loss = x = None
                micro //= 2
                print(f"[p38] rank {self.rank} micro_oom next {micro}", flush=True)
                assert micro >= 1, "even one max-length pair does not fit"
            finally:
                sc.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
        micro = int(self.reduce(micro, "min"))
        self.log(ev="micro_ok", micro=micro, L=L)
        return micro

    def train(self, tok, sc, pad, cap_s, mid):
        """BCE fine-tune, P['epochs'] passes over the train files at most, capped at cap_s seconds of training: AdamW,
        warmup + linear decay to the (capped) last step, grad clip 1, length-bucketed global batches (P13's recipe),
        each rank taking every world-th pair. Rank 0 re-plans the step count from the speed of steps 10-40 and
        broadcasts it. mid(): once at 50% of the steps. LoRA saved (rank 0) every save_s seconds."""
        torch, P = self.torch, self.P
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        cs = list(P["train"])
        st = [self.tokens(tok, f"train_{c}") for c in cs]
        ids = np.concatenate([s[0] for s in st])
        off = np.concatenate([st[0][1]] + [s[1][1:] + sum(len(q[0]) for q in st[:k]) for k, s in enumerate(st) if k])
        y = np.concatenate([self.data(f"train_{c}").y.to_numpy(np.float32) for c in cs])
        ln = np.diff(off)
        bs = int(P["bs"])
        assert bs % self.world == 0, (bs, self.world)
        per = bs // self.world
        micro = self.fit_micro(sc, min(int(P["micro"]), per), int(ln.max()))
        acc = -(-per // micro)
        model = sc
        if self.world > 1:
            model = torch.nn.parallel.DistributedDataParallel(sc, device_ids=[self.rank])
        params = [p for p in sc.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=float(P["lr"]), weight_decay=0.0)
        epochs = int(P["epochs"])
        n_ep = epochs * (len(y) // bs)
        total, warm = [n_ep], max(1, min(100, n_ep // 10))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / total[0]))
        batches, win = [], bs * 100
        for _ in range(epochs):               # length buckets: sort within 100-batch windows, shuffle the batches
            perm = rng.permutation(len(y))
            perm = np.concatenate([c[np.argsort(ln[c], kind="stable")] for c in np.split(perm, range(win, len(perm), win))])
            b = [perm[i:i + bs] for i in range(0, len(perm) - bs + 1, bs)]
            batches += [b[k] for k in rng.permutation(len(b))]
        w0, w1 = min(10, n_ep // 4), min(40, n_ep // 2)
        lossf = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.log(ev="train_start", countries=cs, pairs=len(y), true=round(float(y.mean()), 3), bs=bs, per_rank=per,
                 micro=micro, acc=acc, epochs=epochs, steps=n_ep, cap_s=round(cap_s), world=self.world)
        self.flush()
        model.train()
        step, t_train, ta, did_mid, run, t_save = 0, 0.0, 0.0, mid is None, 0.0, time.time()
        for j in batches:
            t1 = time.time()
            jr = j[self.rank::self.world]
            tot = 0.0
            for a in range(acc):
                jj = jr[a * micro:(a + 1) * micro]
                with self.ac():
                    logit = model(*self.batch(ids, off, jj, pad))
                # sum / per-rank pairs: DDP averages over ranks -> the mean over the global batch
                loss = lossf(logit.float(), torch.from_numpy(y[jj]).to(self.dev)) / len(jr)
                loss.backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            step += 1
            run = 0.98 * run + 0.02 * tot if step > 1 else tot
            if self.cuda:
                torch.cuda.synchronize(self.dev)
            t_train += time.time() - t1
            if step == w0:
                ta = t_train
            if step == w1 and w1 > w0:
                sp = (w1 - w0) * bs / (t_train - ta)
                total[0] = int(self.bcast(max(step + 1, min(n_ep, step + int((cap_s - t_train) * sp / bs)))))
                self.log(ev="train_speed", pairs_s=round(sp, 1), steps=total[0], of=n_ep, pairs=total[0] * bs,
                         eta_s=round((total[0] - step) * bs / sp))
                self.flush()
            if step % 100 == 0:
                self.log(ev="step", step=step, of=total[0], loss=round(run, 4), train_s=round(t_train))
                self.flush()
                stop = self.reduce(float(time.time() > self.deadline), "max")
                if self.bcast(float(time.time() - t_save > float(P["save_s"]))):
                    self.save(sc, partial=True)
                    t_save = time.time()
                if stop:
                    self.log(ev="deadline_stop", step=step)
                    break
            if not did_mid and step >= total[0] // 2 and step >= w1:
                did_mid = True
                mid()
                model.train()
            if step >= total[0]:
                break
        self.log(ev="train_done", steps=step, pairs=step * bs, train_s=round(t_train), loss=round(run, 4))
        del opt, sch, model
        gc.collect()
        if self.cuda:
            torch.cuda.empty_cache()
        return step * bs

    def save(self, sc, partial=False):
        """Trainable weights (LoRA + head), rank 0 -> checkpoint dir + S3; every rank waits."""
        if self.main:
            f = self.ck / f"lora_{self.P['tag']}{'_partial' if partial else ''}.pt"
            self.torch.save({k: v.detach().float().cpu() for k, v in sc.named_parameters() if v.requires_grad}, f)
            self.s3.put(f, f"ckpt/{f.name}")
            self.log(ev="saved", file=f.name, mb=round(f.stat().st_size / 1e6))
        self.barrier()

    def restore(self, sc, src):
        """Trainable weights written by save() into a freshly loaded + LoRA-wrapped base. src = s3://bucket/key or a
        local path."""
        f = Path(src)
        if src.startswith("s3://"):
            import boto3
            f = self.ck / f"r{self.rank}_{Path(src).name}"
            b, k = src[5:].split("/", 1)
            boto3.client("s3").download_file(b, k, str(f))
        sd = self.torch.load(f, map_location="cpu")
        want = {k for k, p in sc.named_parameters() if p.requires_grad}
        assert set(sd) == want, sorted(set(sd) ^ want)[:5]
        _, unexp = sc.load_state_dict(sd, strict=False)
        assert not unexp, unexp[:5]
        self.log(ev="restored", file=f.name, tensors=len(sd))

    # ------------------------------------------------------------------ plan
    def run(self):
        P, tag = self.P, self.P["tag"]
        tok, sc, pad = self.load()
        self.selfcheck(tok, sc, pad, P["score"][0])
        if P["ckpt"]:
            self.restore(sc, P["ckpt"])
        sp = [self.score_write(tok, sc, pad, nm, f"{tag}_smoke", rows=int(P["smoke"])) for nm in P["score"]]
        # planning speed: P['score_speed'] (pairs/s measured on a full file by an earlier job) or 0.6 x the smoke's
        self.speed = self.bcast(float(P["score_speed"]) or 0.6 * float(np.median(sp)))
        n_score = sum(len(self.data(x)) for x in P["score"]) + sum(min(20_000, len(self.data(x))) for x in P["val"])
        n_mid = sum(len(self.data(x)) for x in P["mid"])
        if P["train"]:
            cap = self.bcast(min(float(P["cap_s"]), self.left() - (n_score + n_mid) / self.speed * 1.15 - 900))
            self.log(ev="plan", cap_s=round(cap), left_s=round(self.left()), score_pairs=n_score, mid_pairs=n_mid,
                     score_speed=round(self.speed))
            assert cap > 300, "no time left to train"

            def mid():
                for nm in P["mid"]:
                    self.score_write(tok, sc, pad, nm, f"{tag}_mid")

            self.train(tok, sc, pad, cap, mid if P["mid"] else None)
            self.save(sc)
        for nm in P["score"]:
            need = len(self.data(nm)) / self.speed
            if self.bcast(float(need > self.left())):
                self.log(ev="skip", file=nm, reason="deadline", need_s=round(need))
                continue
            self.speed = self.bcast(self.score_write(tok, sc, pad, nm, tag))
        for nm in P["val"]:                                 # fold-3 validation band, P4b's monitor (AUC only)
            if self.bcast(float(self.left() > 300)):
                self.score_write(tok, sc, pad, nm, tag, rows=20_000)
        self.log(ev="all_done", s=round(time.time() - T0))
        self.flush()


def _logit(x):
    x = np.clip(np.asarray(x, np.float64), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def _worker(rank, world, train_dir, model_dir, params):
    import torch
    if world > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend, init_method="tcp://127.0.0.1:29538", rank=rank,
                                             world_size=world, timeout=datetime.timedelta(minutes=45))
    job = Job(train_dir, model_dir, params, rank, world)
    job.log(ev="start", params=job.P, world=world, files=sorted(p.name for p in Path(train_dir).glob("*.parquet")))
    job.flush()
    try:
        job.run()
    finally:
        job.flush()
        if world > 1:
            torch.distributed.destroy_process_group()


def main(train_dir, test_dir, model_dir, out_dir, params):
    import importlib.metadata as md
    import torch
    os.environ.setdefault("HF_HOME", str(Path(params.get("checkpoint_dir", "/tmp")) / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")        # xet stalled twice on the laptop; plain HTTPS on AWS
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # forked tokenizers x ranks: keep it single-threaded
    vers = {k: md.version(k) for k in ("transformers", "tokenizers", "pandas", "pyarrow", "numpy", "scikit-learn")}
    n_gpu = torch.cuda.device_count()
    world = int(params.get("gpus") or n_gpu or 1)
    print(f"[p38] torch {torch.__version__} cuda {torch.version.cuda} gpus {n_gpu} "
          f"{torch.cuda.get_device_name(0) if n_gpu else '-'} world {world} {vers}", flush=True)
    p = dict(params)
    up = None
    if "/" in p.get("model", DEFAULTS["model"]) and not Path(p.get("model", DEFAULTS["model"])).exists():
        wd, up = fetch_weights(p.get("model", DEFAULTS["model"]), p.get("weights_s3", ""),
                               Path(p.get("checkpoint_dir", "/tmp")) / "weights")
        p["_weights_dir"] = str(wd)
    try:
        if world == 1:
            _worker(0, 1, train_dir, model_dir, p)
        else:
            torch.multiprocessing.spawn(_worker, args=(world, train_dir, model_dir, p), nprocs=world, join=True)
    finally:
        if up is not None:
            up.join()
