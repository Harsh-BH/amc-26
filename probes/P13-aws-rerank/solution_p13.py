"""P13 SageMaker job (ONE ml.g5.xlarge run): big multilingual rerankers on m2_big's uncertain band 0.02<p<0.98.
Shipped alone as solution.py next to sm_train.py; sm_train calls main(train_dir, test_dir, model_dir, out_dir, params).

Phases, in order of value. Each writes its r files to model_dir/r AND to S3 (params['s3']) the moment it finishes,
so a timeout keeps everything done so far. A phase starts only if its measured-speed estimate fits the deadline;
training phases are capped in seconds (the step count is fixed after timing the first ~120 steps).
  bge_zs         bge-reranker-v2-m3 zero-shot -> holdout band (India, US) + France test band
  bge_locoUS     fine-tune on US train only    -> holdout band at 50% of training and at the end
  bge_locoIndia  fine-tune on India train only -> same
  bge_pool       fine-tune on India + US       -> holdout band, test bands India / US / France; weights -> S3
  qwen_zs        Qwen3-Reranker-0.6B zero-shot (its own yes/no template) -> holdout band
  qwen_pool      fine-tune on India + US -> holdout band; test bands only if its holdout stack AUC beats bge_pool's
  qwen_zs_fr     Qwen3 zero-shot France test band (time permitting)
  bge_ckpt       scoring only: base bge + the fine-tuned weights at params['ckpt'] -> holdout (as bge_poolck) and the
                 test bands India / US / France (as bge_pool)
Channel 'train': train_<c>, val_<c>, holdout_<c> (c = India, US) and test_<c> (India, US, France) parquet, columns
ta, tb (P3's "name | addr" texts), p (GBDT), y (not in test), s1/sx (holdout, test).
Output r files: (s1, sx, r = sigmoid(logit)); log.jsonl = one JSON line per event (speeds, AUCs, GPU memory).
"""
import gc, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
DEFAULTS = {
    "bge": "BAAI/bge-reranker-v2-m3", "qwen": "Qwen/Qwen3-Reranker-0.6B",
    "phases": ["bge_zs", "bge_locoUS", "bge_locoIndia", "bge_pool", "qwen_zs", "qwen_pool", "qwen_zs_fr"],
    "maxlen": 96, "qmaxlen": 160, "bs": 128, "micro": 64, "qmicro": 32, "lr": 2e-5, "qlr": 1e-5,
    "cap_s": {"bge_locoUS": 1800, "bge_locoIndia": 1200, "bge_pool": 4800, "qwen_pool": 2400},
    "epochs": {"bge_pool": 2},           # max passes (P11: 2 epochs beat 1 for e5-small); others 1; the cap decides
    "score_bs": 512, "deadline_s": 22200, "min_train_s": 300, "s3": "", "qwen_test": "auto", "ckpt": "",
    "load_dtype": "float32",             # bfloat16: laptop checks only
    # pairs/s used for planning until a phase has measured the real speed
    "speed": {"bge_train": 300, "bge_score": 1000, "qwen_train": 90, "qwen_score": 350},
}
QPRE = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
        "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
QSUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QINST = "Given two business records (name | address), judge whether the Document is the same business as the Query"


class S3:
    """Best-effort uploads under s3://bucket/prefix/; '' disables. A failed upload is logged, never fatal (the
    same files also land in model_dir -> model.tar.gz)."""

    def __init__(self, uri: str):
        self.on = bool(uri)
        if self.on:
            import boto3
            self.bucket, self.prefix = uri[5:].rstrip("/").split("/", 1)
            self.c = boto3.client("s3")

    def _try(self, fn, key):
        for i in range(3):
            try:
                return fn()
            except Exception as e:                      # network / throttling: retry, then give up quietly
                print(f"[s3] {key} attempt {i + 1} failed: {e!r}", flush=True)
                time.sleep(5 * (i + 1))

    def put(self, path, key: str):
        if self.on:
            self._try(lambda: self.c.upload_file(str(path), self.bucket, f"{self.prefix}/{key}"), key)

    def put_json(self, obj, key: str):
        if self.on:
            body = json.dumps(obj, default=str).encode()
            self._try(lambda: self.c.put_object(Bucket=self.bucket, Key=f"{self.prefix}/{key}", Body=body), key)


class Job:
    def __init__(self, train_dir, model_dir, params):
        import torch
        self.torch, self.P = torch, {**DEFAULTS, **params}
        self.P["cap_s"] = {**DEFAULTS["cap_s"], **params.get("cap_s", {})}
        self.P["speed"] = {**DEFAULTS["speed"], **params.get("speed", {})}
        self.P["epochs"] = {**DEFAULTS["epochs"], **params.get("epochs", {})}
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cuda.matmul.allow_tf32 = True
        self.src, self.out = Path(train_dir), Path(model_dir)
        (self.out / "r").mkdir(parents=True, exist_ok=True)
        self.ck = Path(params.get("checkpoint_dir", "/tmp")) / "p13"
        self.ck.mkdir(parents=True, exist_ok=True)
        self.s3 = S3(self.P["s3"])
        self.deadline = T0 + float(self.P["deadline_s"])
        self.D, self.tok, self.auc = {}, {}, {}
        self.speed = dict(self.P["speed"])

    # ------------------------------------------------------------------ bookkeeping
    def log(self, **kw):
        kw = {"t": round(time.time() - T0), **kw}
        if self.dev == "cuda":
            kw["gpu_gb"] = round(self.torch.cuda.max_memory_allocated() / 1e9, 2)
        print("[p13] " + json.dumps(kw, default=str), flush=True)
        with open(self.out / "log.jsonl", "a") as f:
            f.write(json.dumps(kw, default=str) + "\n")

    def flush_log(self):
        self.s3.put(self.out / "log.jsonl", "log.jsonl")

    def left(self):
        return self.deadline - time.time()

    def data(self, name):
        if name not in self.D:
            self.D[name] = pd.read_parquet(self.src / f"{name}.parquet")
        return self.D[name]

    def write(self, tag, d, r):
        f = self.out / "r" / f"r_{tag}.parquet"
        pd.DataFrame({"s1": d.s1.to_numpy(), "sx": d.sx.to_numpy(), "r": r.astype(np.float32)}).to_parquet(f, index=False)
        self.s3.put(f, f"r/{f.name}")
        self.log(ev="wrote", file=f.name, rows=len(r))

    # ------------------------------------------------------------------ models + tokens
    def load(self, fam):
        """fam 'bge': HF sequence classifier (1 logit). fam 'qwen': the causal LM's last-token logit(yes) - logit(no)
        (Qwen3-Reranker's own score) as a 1-row head, so both fine-tune the same way. fp32 master weights, word
        embeddings frozen (bf16 autocast at run time)."""
        torch = self.torch
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        name, dt = self.P[fam], getattr(torch, self.P["load_dtype"])
        for i in range(3):                                  # the HF download is the one network step: retry
            try:
                tok = AutoTokenizer.from_pretrained(name)
                if fam == "bge":
                    body = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, dtype=dt)
                    w = None
                else:
                    lm = AutoModelForCausalLM.from_pretrained(name, dtype=dt)
                    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
                    W = lm.get_output_embeddings().weight
                    w = (W[yes] - W[no]).detach().float().clone()[None]
                    body = lm.model
                    del lm
                break
            except Exception as e:
                self.log(ev="load_retry", fam=fam, err=repr(e)[:300])
                if i == 2:
                    raise
                time.sleep(20)

        class Scorer(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.m = body
                s.w = None if w is None else torch.nn.Parameter(w)

            def forward(s, x, msk):
                if s.w is None:
                    return s.m(input_ids=x, attention_mask=msk).logits.float().squeeze(-1)
                h = s.m(input_ids=x, attention_mask=msk).last_hidden_state
                return (h[torch.arange(len(h), device=h.device), msk.sum(1) - 1].float() @ s.w.T).squeeze(-1)

        sc = Scorer().to(self.dev)
        sc.m.get_input_embeddings().weight.requires_grad_(False)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.log(ev="loaded", fam=fam, name=name, params_m=round(sum(p.numel() for p in sc.parameters()) / 1e6, 1),
                 trainable_m=round(sum(p.numel() for p in sc.parameters() if p.requires_grad) / 1e6, 1))
        return tok, sc, pad

    def tokens(self, fam, tok, name):
        """Pair texts of data file <name> -> (flat int32 ids, int64 offsets), cached per (fam, file). bge: XLM-R
        pair encoding, longest_first to maxlen. qwen: template prefix + instruct/query/document + suffix, the body
        truncated so the suffix (where the yes/no logit is read) always survives."""
        key = (fam, name)
        if key in self.tok:
            return self.tok[key]
        t0, d = time.time(), self.data(name)
        ta, tb = d.ta.tolist(), d.tb.tolist()
        ids, off = [], [np.zeros(1, np.int64)]
        if fam == "qwen":
            pre, suf = tok.encode(QPRE, add_special_tokens=False), tok.encode(QSUF, add_special_tokens=False)
            room = int(self.P["qmaxlen"]) - len(pre) - len(suf)
        for i in range(0, len(ta), 50_000):
            if fam == "bge":
                e = tok(ta[i:i + 50_000], tb[i:i + 50_000], truncation="longest_first", max_length=int(self.P["maxlen"]),
                        return_attention_mask=False)["input_ids"]
            else:
                body = [f"<Instruct>: {QINST}\n<Query>: {a}\n<Document>: {b}" for a, b in zip(ta[i:i + 50_000], tb[i:i + 50_000])]
                e = [pre + x + suf for x in tok(body, truncation=True, max_length=room, add_special_tokens=False,
                                                return_attention_mask=False)["input_ids"]]
            ids.append(np.fromiter((t for x in e for t in x), np.int32))
            off.append(np.cumsum([len(x) for x in e]) + off[-1][-1])
        self.tok[key] = (np.concatenate(ids), np.concatenate(off))
        ln = np.diff(self.tok[key][1])
        self.log(ev="tokens", fam=fam, file=name, pairs=len(ln), mean_len=round(float(ln.mean()), 1),
                 max_len=int(ln.max()), s=round(time.time() - t0, 1))
        return self.tok[key]

    def batch(self, ids, off, j, pad):
        torch = self.torch
        ln = (off[j + 1] - off[j]).astype(np.int64)
        L = int(ln.max())
        x = np.full((len(j), L), pad, np.int64)
        msk = np.arange(L)[None, :] < ln[:, None]
        x[msk] = ids[np.concatenate([np.arange(off[k], off[k + 1]) for k in j])]
        return torch.from_numpy(x).to(self.dev), torch.from_numpy(msk.astype(np.int64)).to(self.dev)

    def ac(self):
        # bf16 on the GPU; plain fp32 on CPU (the rehearsal), where bf16 autocast is ~10x slower
        return self.torch.autocast(self.dev, dtype=self.torch.bfloat16, enabled=self.dev == "cuda")

    def selfcheck(self, fam, tok, sc, pad):
        """Real-model guard (the laptop could not fetch the weights to test them): batched right-padded logits of 32
        holdout pairs == one-pair-at-a-time unpadded forwards (catches a wrong last-token gather / mask), plus
        Qwen's yes/no ids vs the model card's LogitScore config (9693 / 2152). False -> the family is skipped."""
        torch = self.torch
        ids, off = self.tokens(fam, tok, "holdout_India")
        j = np.arange(min(32, len(off) - 1))
        sc.eval()
        with torch.no_grad(), self.ac():
            b = sc(*self.batch(ids, off, j, pad)).float().cpu().numpy()
            one = np.array([float(sc(*self.batch(ids, off, np.array([k]), pad))) for k in j])
        dz = float(np.abs(b - one).max())
        info = {"max_dlogit": round(dz, 4), "logit_mean": round(float(one.mean()), 3), "logit_std": round(float(one.std()), 3)}
        if fam == "qwen":
            info["yes_no_ids"] = [tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")]
        ok = dz < 0.5                                        # bf16 kernel noise is ~1e-3..0.1; a gather bug is whole units
        self.log(ev="selfcheck", fam=fam, ok=ok, **info)
        self.flush_log()
        return ok

    # ------------------------------------------------------------------ scoring
    def score(self, fam, tok, sc, pad, name):
        """sigmoid(logit) for every pair of data file <name>, length-sorted batches; halves the batch on OOM."""
        torch = self.torch
        ids, off = self.tokens(fam, tok, name)
        n, bs, t0 = len(off) - 1, int(self.P["score_bs"]), time.time()
        o = np.argsort(np.diff(off), kind="stable")
        r = np.empty(n, np.float32)
        sc.eval()
        i = 0
        with torch.no_grad(), self.ac():
            while i < n:
                j = o[i:i + bs]
                try:
                    r[j] = torch.sigmoid(sc(*self.batch(ids, off, j, pad))).float().cpu().numpy()
                    i += len(j)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                    self.log(ev="score_oom", bs=bs)
        sp = n / max(time.time() - t0, 1e-6)
        self.speed[f"{fam}_score"] = sp
        self.log(ev="scored", fam=fam, file=name, pairs=n, pairs_s=round(sp), s=round(time.time() - t0))
        return r

    def holdout(self, fam, tok, sc, pad, tag):
        """Score + write the holdout band (both countries); log per-country AUC of p, r and their logistic stack."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        parts, res = [], {}
        for c in ("India", "US"):
            d = self.data(f"holdout_{c}")
            r = self.score(fam, tok, sc, pad, f"holdout_{c}")
            parts.append((d, r))
            y, p = d.y.to_numpy(), d.p.to_numpy()
            X = np.c_[_logit(p), _logit(r)]
            st = LogisticRegression(C=1.0).fit(X, y).decision_function(X)
            res[c] = {"auc_p": roc_auc_score(y, p), "auc_r": roc_auc_score(y, r), "auc_stack": roc_auc_score(y, st)}
        d = pd.concat([x[0][["s1", "sx"]] for x in parts], ignore_index=True)
        self.write(f"{tag}_holdout", d, np.concatenate([x[1] for x in parts]))
        self.auc[tag] = res
        self.log(ev="holdout_auc", tag=tag, **{c: {k: round(v, 4) for k, v in m.items()} for c, m in res.items()})
        self.flush_log()

    def test(self, fam, tok, sc, pad, tag, countries):
        for c in countries:
            n = len(self.data(f"test_{c}"))
            if n / self.speed[f"{fam}_score"] * 1.1 > self.left():
                self.log(ev="skip", what=f"{tag}_test_{c}", reason="deadline", need_s=round(n / self.speed[f"{fam}_score"]))
                continue
            self.write(f"{tag}_test_{c}", self.data(f"test_{c}"), self.score(fam, tok, sc, pad, f"test_{c}"))
            self.flush_log()

    # ------------------------------------------------------------------ training
    def fit_micro(self, sc, opt, micro, L, pad):
        """Largest micro-batch (<= micro) whose max-length forward/backward + optimizer step fits the GPU. The probe
        step runs with lr 0 and its Adam state is dropped, so the weights are unchanged."""
        torch = self.torch
        while True:
            try:
                x = torch.full((micro, L), 5, dtype=torch.long, device=self.dev)
                msk = torch.ones_like(x)
                with self.ac():
                    loss = sc(x, msk).float().mean()
                loss.backward()
                lrs = [g["lr"] for g in opt.param_groups]
                for g in opt.param_groups:
                    g["lr"] = 0.0
                opt.step()
                for g, lr in zip(opt.param_groups, lrs):
                    g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                opt.state.clear()
                self.log(ev="micro_ok", micro=micro, L=L)
                return micro
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                opt.state.clear()
                torch.cuda.empty_cache()
                micro //= 2
                self.log(ev="micro_oom", next=micro)
                assert micro >= 1, "even one max-length pair does not fit"

    def train(self, fam, tok, sc, pad, tag, countries, cap_s, mid=None):
        """BCE fine-tune, P['epochs'][tag] passes (default 1) over the train files of <countries> at most, capped at
        cap_s seconds of training:
        AdamW, warmup + linear decay to the (capped) last step, grad clip 1, length-bucketed batches (P4b/P11's
        recipe). mid(): called once at 50% of the steps (LOCO transfer at half training)."""
        torch = self.torch
        from sklearn.metrics import roc_auc_score
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        st = [self.tokens(fam, tok, f"train_{c}") for c in countries]
        ids = np.concatenate([s[0] for s in st])
        off = np.concatenate([st[0][1]] + [s[1][1:] + sum(len(q[0]) for q in st[:k]) for k, s in enumerate(st) if k])
        y = np.concatenate([self.data(f"train_{c}").y.to_numpy(np.float32) for c in countries])
        ln = np.diff(off)
        bs = int(self.P["bs"])
        lr = float(self.P["qlr" if fam == "qwen" else "lr"])
        params = [p for p in sc.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        micro = int(self.P["qmicro" if fam == "qwen" else "micro"])
        if self.dev == "cuda":
            micro = self.fit_micro(sc, opt, micro, int(ln.max()), pad)
        acc = max(1, bs // micro)
        mb = bs // acc
        epochs = int(self.P["epochs"].get(tag, 1))
        n_ep = epochs * (len(y) // bs)
        total, warm = [n_ep], max(1, min(500, n_ep // 10))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / total[0]))
        batches, w = [], bs * 100
        for _ in range(epochs):              # length buckets: sort within 100-batch windows, shuffle the batches
            perm = rng.permutation(len(y))
            perm = np.concatenate([c[np.argsort(ln[c], kind="stable")] for c in np.split(perm, range(w, len(perm), w))])
            b = [perm[i:i + bs] for i in range(0, len(perm) - bs + 1, bs)]
            batches += [b[k] for k in rng.permutation(len(b))]
        w0, w1 = min(20, n_ep // 4), min(120, n_ep // 2)
        lossf = torch.nn.BCEWithLogitsLoss()
        self.log(ev="train_start", tag=tag, countries=countries, pairs=len(y), true=round(float(y.mean()), 3), bs=bs,
                 micro=mb, acc=acc, lr=lr, epochs=epochs, steps=n_ep, cap_s=cap_s)
        sc.train()
        step, t_train, ta, did_mid, run = 0, 0.0, None, mid is None, 0.0
        for j in batches:
            t1 = time.time()
            for a in range(acc):
                jj = j[a * mb:(a + 1) * mb]
                with self.ac():
                    logit = sc(*self.batch(ids, off, jj, pad))
                loss = lossf(logit.float(), torch.from_numpy(y[jj]).to(self.dev)) / acc
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            step += 1
            run = 0.98 * run + 0.02 * loss.item() * acc if step > 1 else loss.item() * acc
            if self.dev == "cuda":
                torch.cuda.synchronize()
            t_train += time.time() - t1
            if step == w0:
                ta = t_train
            if step == w1 and w1 > w0:
                sp = (w1 - w0) * bs / (t_train - ta)
                self.speed[f"{fam}_train"] = sp
                total[0] = max(step + 1, min(n_ep, step + int((cap_s - t_train) * sp / bs)))
                self.log(ev="train_speed", tag=tag, pairs_s=round(sp), steps=total[0], of=n_ep,
                         pairs=total[0] * bs, eta_s=round((total[0] - step) * bs / sp))
            if step % 500 == 0:
                self.log(ev="step", tag=tag, step=step, of=total[0], loss=round(run, 4), train_s=round(t_train))
            if not did_mid and step >= total[0] // 2 and step >= w1:
                did_mid = True
                mid()
                sc.train()
            if step >= total[0] or time.time() > self.deadline:
                break
        self.log(ev="train_done", tag=tag, steps=step, pairs=step * bs, train_s=round(t_train), loss=round(run, 4))
        del opt, sch
        gc.collect()
        if self.dev == "cuda":
            torch.cuda.empty_cache()
        v = {}
        for c in ("India", "US"):                        # fold-3 validation band, P4b's monitor
            d = self.data(f"val_{c}")
            r = self.score(fam, tok, sc, pad, f"val_{c}")
            yy, p = d.y.to_numpy(), d.p.to_numpy()
            v[c] = {"auc_gbdt": round(roc_auc_score(yy, p), 4), "auc_ce": round(roc_auc_score(yy, r), 4),
                    "auc_sum": round(roc_auc_score(yy, _logit(p) + _logit(r)), 4)}
        self.log(ev="val_auc", tag=tag, **v)
        return step * bs

    def save(self, sc, tag):
        f = self.ck / f"ce_{tag}.pt"
        frozen = {k for k, p in sc.named_parameters() if not p.requires_grad}     # the base model's word embeddings
        self.torch.save({k: v.half() for k, v in sc.state_dict().items() if k not in frozen}, f)
        self.s3.put(f, f"ckpt/{f.name}")
        self.log(ev="saved", file=f.name, mb=round(f.stat().st_size / 1e6))
        f.unlink()

    def restore(self, sc, src):
        """Weights written by save() (every weight but the frozen word embeddings, fp16) into a freshly loaded base
        model. src = s3://bucket/key (downloaded once) or a local path."""
        f = Path(src)
        if src.startswith("s3://"):
            import boto3
            f = self.ck / Path(src).name
            b, k = src[5:].split("/", 1)
            boto3.client("s3").download_file(b, k, str(f))
        sd = self.torch.load(f, map_location="cpu")
        miss, unexp = sc.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
        frozen = {k for k, p in sc.named_parameters() if not p.requires_grad}
        assert not unexp and set(miss) == frozen, (miss[:5], unexp[:5])
        self.log(ev="restored", file=f.name, tensors=len(sd), missing=len(miss), mb=round(f.stat().st_size / 1e6))

    # ------------------------------------------------------------------ plan
    def n(self, *names):
        return sum(len(self.data(x)) for x in names)

    def est(self, fam, train_pairs=0, cap=0, score_pairs=0):
        tr = min(cap, train_pairs / self.speed[f"{fam}_train"]) if train_pairs else 0
        return tr + score_pairs / self.speed[f"{fam}_score"] + 60

    def fits(self, what, need):
        ok = need * 1.1 < self.left()
        self.log(ev="plan", what=what, need_s=round(need), left_s=round(self.left()), run=ok)
        return ok

    def run(self):
        H = ("holdout_India", "holdout_US")
        tv = self.n("val_India", "val_US")
        todo = list(self.P["phases"])
        models, checked = {}, {}

        def model(fam, fresh=False):
            if fresh or fam not in models:
                models.pop(fam, None)
                gc.collect()
                if self.dev == "cuda":
                    self.torch.cuda.empty_cache()
                models[fam] = self.load(fam)
                if fam not in checked:
                    checked[fam] = self.selfcheck(fam, *models[fam])
            if fam == "qwen" and not checked[fam]:
                self.log(ev="skip", what=fam, reason="qwen selfcheck failed")
                return None                                   # bge runs regardless: its HF head is the standard path
            return models[fam]

        for ph in todo:
            tok = sc = pad = None                             # drop the previous phase's model before loading one
            t0 = time.time()
            fam = "qwen" if ph.startswith("qwen") else "bge"
            if ph in ("bge_zs", "qwen_zs"):
                sp = self.n(*H) + (self.n("test_France") if ph == "bge_zs" else 0)
                if not self.fits(ph, self.est(fam, score_pairs=sp)):
                    continue
                if (M := model(fam, fresh=True)) is None:
                    continue
                tok, sc, pad = M
                self.holdout(fam, tok, sc, pad, ph)
                if ph == "bge_zs":
                    self.test(fam, tok, sc, pad, ph, ["France"])
            elif ph == "qwen_zs_fr":
                if not self.fits(ph, self.est(fam, score_pairs=self.n("test_France"))):
                    continue
                if (M := model(fam, fresh=True)) is None:
                    continue
                tok, sc, pad = M
                self.test(fam, tok, sc, pad, "qwen_zs", ["France"])
            elif ph == "bge_ckpt":      # scoring only: the pooled model an earlier job saved (params['ckpt'])
                if not self.fits(ph, self.est(fam, score_pairs=self.n(*H, "test_India", "test_US", "test_France"))):
                    continue
                if (M := model(fam, fresh=True)) is None:
                    continue
                tok, sc, pad = M
                self.restore(sc, self.P["ckpt"])
                self.holdout(fam, tok, sc, pad, "bge_poolck")    # the restored model; must match r_bge_pool_holdout
                self.test(fam, tok, sc, pad, "bge_pool", ["India", "US", "France"])
                models.pop(fam, None)
            elif ph.startswith("bge_loco"):
                cty = [ph.removeprefix("bge_loco")]
                cap = float(self.P["cap_s"][ph])
                if not self.fits(ph, self.est(fam, self.n(f"train_{cty[0]}"), cap, 2 * self.n(*H) + tv)):
                    continue
                if (M := model(fam, fresh=True)) is None:
                    continue
                tok, sc, pad = M
                self.train(fam, tok, sc, pad, ph, cty, cap, mid=lambda: self.holdout(fam, tok, sc, pad, f"{ph}_mid"))
                self.holdout(fam, tok, sc, pad, ph)
            elif ph in ("bge_pool", "qwen_pool"):
                cty = ["India", "US"]
                cap = float(self.P["cap_s"][ph])
                tests = self.n("test_India", "test_US", "test_France") if ph == "bge_pool" else 0
                rest = (self.n(*H) + tv + tests) / self.speed[f"{fam}_score"] + 120
                cap = min(cap, self.left() / 1.1 - rest)       # squeeze training, never the scoring
                if cap < float(self.P["min_train_s"]):
                    self.log(ev="skip", what=ph, reason="deadline", cap_s=round(cap))
                    continue
                ep = int(self.P["epochs"].get(ph, 1))
                if not self.fits(ph, self.est(fam, ep * self.n("train_India", "train_US"), cap, self.n(*H) + tv + tests)):
                    continue
                if (M := model(fam, fresh=True)) is None:
                    continue
                tok, sc, pad = M
                self.train(fam, tok, sc, pad, ph, cty, cap)
                self.holdout(fam, tok, sc, pad, ph)
                self.save(sc, ph)
                go = {"always": True, "never": False}.get(self.P["qwen_test"])
                if ph == "qwen_pool" and go is None:
                    q, b = self.auc.get("qwen_pool"), self.auc.get("bge_pool")
                    mean = lambda a: np.mean([a[c]["auc_stack"] for c in a])
                    go = bool(q and (not b or mean(q) > mean(b) + 0.001))
                    self.log(ev="qwen_test_decision", go=go, qwen=q and round(mean(q), 4), bge=b and round(mean(b), 4))
                if ph == "bge_pool" or go:
                    self.test(fam, tok, sc, pad, ph, ["India", "US", "France"])
                models.pop(fam, None)                         # a fine-tuned model is never reused as zero-shot
            else:
                self.log(ev="unknown_phase", phase=ph)
                continue
            self.log(ev="phase_done", phase=ph, s=round(time.time() - t0), speed={k: round(v) for k, v in self.speed.items()})
            self.flush_log()
        self.log(ev="all_done", s=round(time.time() - T0))
        self.flush_log()


def _logit(x):
    x = np.clip(np.asarray(x, np.float64), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def main(train_dir, test_dir, model_dir, out_dir, params):
    import importlib.metadata as md
    import torch
    os.environ.setdefault("HF_HOME", str(Path(params.get("checkpoint_dir", "/tmp")) / "hf"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    vers = {k: md.version(k) for k in ("transformers", "tokenizers", "pandas", "pyarrow", "numpy", "scikit-learn",
                                         "huggingface-hub", "boto3")}
    print(f"[p13] torch {torch.__version__} cuda {torch.version.cuda} gpu "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'} {vers}", flush=True)
    job = Job(train_dir, model_dir, params)
    job.log(ev="start", params=job.P, versions=vers, files=sorted(p.name for p in Path(train_dir).glob("*.parquet")))
    job.flush_log()
    try:
        job.run()
    finally:
        job.flush_log()
