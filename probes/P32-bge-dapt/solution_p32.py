"""P32 job: domain-adaptive pre-training (MLM) of bge-reranker-v2-m3's encoder on UNLABELLED target-country text, then
P13's exact cross-encoder fine-tune, then scoring. Shipped as solution.py next to sm_train.py AND solution_p13.py (P13's
Job: loading, tokens, train, holdout, test, save/restore are reused unchanged). Runs locally too (the portable kit):
params['s3'] = '' disables every AWS call.

One chain per job (params):
  dapt      'India' | 'US' | 'France' | ''   -> MLM on dapt_<dapt>.parquet (column t = "name | address", unique strings)
  ft        ['US'] | ['India'] | ['India', 'US'] -> BCE fine-tune on train_<c>.parquet (P13's recipe; epochs[tag])
  tag       output tag: r_<tag>_holdout.parquet (holdout band, both countries) + r_<tag>_test_<c> for c in tests
  tests     e.g. ['France'] -> test_<c>.parquet scored with the fine-tuned model
  save_ft   true -> fine-tuned weights to ckpt/ce_<tag>.pt (P13 save format, loadable by P13's bge_ckpt / restore)
MLM: 15% of non-special tokens (80% <mask> / 10% random / 10% kept), max_len dapt_maxlen, word embeddings frozen (as in
P13's fine-tune) and tied to a fresh LM head (dense + gelu + LayerNorm + bias) that is trained alone for head_warm steps
before the encoder joins; AdamW, warm-up + linear decay, bf16 autocast, length-bucketed batches, OOM-halving micro-batch.
"""
import gc, json, time
from pathlib import Path

import numpy as np

import solution_p13 as B

DEF32 = {"dapt": "", "ft": ["India", "US"], "tag": "dapt", "tests": [], "save_ft": False, "save_dapt": True,
         "dapt_epochs": 2, "dapt_cap_s": 6000, "dapt_bs": 256, "dapt_micro": 128, "dapt_lr": 5e-5, "dapt_head_lr": 5e-4,
         "dapt_maxlen": 64, "mask_p": 0.15, "head_warm": 300, "dapt_warm": 0.06, "dapt_reserve_s": 3600, "ft_cap_s": 6000}


def mlm(job, tok, sc, name):
    """Continue masked-LM pre-training of sc.m.roberta on data file <name> (column t). Mutates sc in place."""
    torch, P = job.torch, job.P
    F = torch.nn.functional
    t0 = time.time()
    txt = job.data(name).t.tolist()
    ids, off = [], [np.zeros(1, np.int64)]
    for i in range(0, len(txt), 50_000):
        e = tok(txt[i:i + 50_000], truncation=True, max_length=int(P["dapt_maxlen"]), return_attention_mask=False)["input_ids"]
        ids.append(np.fromiter((t for x in e for t in x), np.int32))
        off.append(np.cumsum([len(x) for x in e]) + off[-1][-1])
    ids, off = np.concatenate(ids), np.concatenate(off)
    ln = np.diff(off)
    n = len(ln)
    job.log(ev="dapt_tokens", file=name, texts=n, mean_len=round(float(ln.mean()), 1), max_len=int(ln.max()),
            trunc=round(float((ln >= int(P["dapt_maxlen"])).mean()), 4), s=round(time.time() - t0, 1))

    enc, emb = sc.m.base_model, sc.m.get_input_embeddings()
    H, V = emb.weight.shape[1], emb.weight.shape[0]
    special = torch.tensor(sorted(set(tok.all_special_ids)), device=job.dev)
    mask_id, pad = tok.mask_token_id, tok.pad_token_id

    class Head(torch.nn.Module):          # XLM-R's LM head shape; decoder = the (frozen) word embeddings, tied
        def __init__(s):
            super().__init__()
            s.dense, s.ln = torch.nn.Linear(H, H), torch.nn.LayerNorm(H, eps=1e-5)
            s.bias = torch.nn.Parameter(torch.zeros(V))

        def forward(s, h):
            return F.linear(s.ln(F.gelu(s.dense(h))), emb.weight, s.bias)

    head = Head().to(job.dev)
    g = torch.Generator(device=job.dev)

    def masked(x, msk, seed=None):
        """(input with masking applied, bool mask of predicted positions, their labels); seed -> a fixed mask."""
        gen = g.manual_seed(seed) if seed is not None else None
        u = torch.rand(x.shape, device=job.dev, generator=gen)
        v = torch.rand(x.shape, device=job.dev, generator=gen)
        m = (u < float(P["mask_p"])) & (msk > 0) & ~torch.isin(x, special)
        y = x[m]
        x = x.clone()
        x[m & (v < 0.8)] = mask_id
        rnd = m & (v >= 0.8) & (v < 0.9)
        x[rnd] = torch.randint(5, V - 1, (int(rnd.sum()),), device=job.dev, generator=gen)
        return x, m, y

    def loss_acc(j, grad_enc, seed=None):
        x, msk = job.batch(ids, off, j, pad)
        xm, m, y = masked(x, msk, seed)
        with job.ac():
            if grad_enc:
                h = enc(input_ids=xm, attention_mask=msk).last_hidden_state[m]
            else:
                with torch.no_grad():
                    h = enc(input_ids=xm, attention_mask=msk).last_hidden_state[m]
            z = head(h)
        z = z.float()
        return F.cross_entropy(z, y), (z.argmax(-1) == y).float().mean(), int(m.sum())

    rng = np.random.default_rng(1)
    ev_j = rng.choice(n, min(4096, n), replace=False)

    def evaluate(stage):
        """Masked-token loss / accuracy on a fixed 4,096-text sample with a fixed mask (same positions every call)."""
        sc.eval(); head.eval()
        L, A, N = 0.0, 0.0, 0
        with torch.no_grad():
            for k, i in enumerate(range(0, len(ev_j), 64)):          # fixed chunks -> the same masks every call
                l, a, nm = loss_acc(ev_j[i:i + 64], False, seed=1000 + k)
                L, A, N = L + float(l) * nm, A + float(a) * nm, N + nm
        job.log(ev="dapt_eval", stage=stage, loss=round(L / N, 4), acc=round(A / N, 4), tokens=N)
        sc.train(); head.train()
        return L / N

    bs, micro = int(P["dapt_bs"]), int(P["dapt_micro"])
    epochs, hw = int(P["dapt_epochs"]), int(P["head_warm"])
    enc_p = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": float(P["dapt_lr"])},
                             {"params": list(head.parameters()), "lr": float(P["dapt_head_lr"])}], weight_decay=0.01)
    total = [hw + epochs * (n // bs)]
    warm = max(1, int(float(P["dapt_warm"]) * (total[0] - hw)))
    lam_enc = lambda s: 0.0 if s < hw else min(1.0, (s - hw + 1) / warm) * max(0.0, 1 - (s - hw) / max(1, total[0] - hw))
    lam_head = lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / total[0])
    sch = torch.optim.lr_scheduler.LambdaLR(opt, [lam_enc, lam_head])
    batches, w = [], bs * 100
    for _ in range(epochs + 1):             # +1: the head-only warm-up steps are drawn from an extra pass
        perm = rng.permutation(n)
        perm = np.concatenate([c[np.argsort(ln[c], kind="stable")] for c in np.split(perm, range(w, n, w))])
        b = [perm[i:i + bs] for i in range(0, n - bs + 1, bs)]
        batches += [b[k] for k in rng.permutation(len(b))]
    batches = batches[-hw:] + batches[:epochs * (n // bs)] if hw else batches[:epochs * (n // bs)]
    cap = float(P["dapt_cap_s"])
    job.log(ev="dapt_start", texts=n, bs=bs, micro=micro, epochs=epochs, steps=total[0], head_warm=hw,
            lr=P["dapt_lr"], head_lr=P["dapt_head_lr"], mask_p=P["mask_p"], cap_s=cap)
    sc.train(); head.train()
    step, t_enc, run, racc, ta = 0, 0.0, 0.0, 0.0, None
    s0, s1 = hw + 20, hw + 220
    for j in batches:
        t1 = time.time()
        grad_enc = step >= hw
        while True:
            try:
                parts = [j[i:i + micro] for i in range(0, len(j), micro)]
                tot = 0.0
                accs = 0.0
                for q in parts:
                    l, a, nm = loss_acc(q, grad_enc)
                    if nm == 0:                 # nothing masked in this micro-batch (tiny micro only)
                        continue
                    (l * len(q) / len(j)).backward()
                    tot += float(l) * len(q) / len(j)
                    accs += float(a) * len(q) / len(j)
                break
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                micro = max(1, micro // 2)
                job.log(ev="dapt_oom", micro=micro)
        torch.nn.utils.clip_grad_norm_(enc_p + list(head.parameters()), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        step += 1
        run = 0.98 * run + 0.02 * tot if step > 1 else tot
        racc = 0.98 * racc + 0.02 * accs if step > 1 else accs
        if job.dev == "cuda":
            torch.cuda.synchronize()
        if grad_enc:
            t_enc += time.time() - t1
        if step == hw:
            evaluate("head_warm")
        if step == s0:
            ta = t_enc
        if step == s1:
            sp = (s1 - s0) * bs / (t_enc - ta)
            total[0] = max(step + 1, min(total[0], step + int((cap - t_enc) * sp / bs)))
            job.log(ev="dapt_speed", texts_s=round(sp), steps=total[0], of=hw + epochs * (n // bs), micro=micro,
                    eta_s=round((total[0] - step) * bs / sp))
        if step % 500 == 0:
            job.log(ev="dapt_step", step=step, of=total[0], loss=round(run, 4), acc=round(racc, 4), enc_s=round(t_enc))
            job.flush_log()
        if step >= total[0] or time.time() > job.deadline - float(P["dapt_reserve_s"]):
            break
    job.log(ev="dapt_done", steps=step, texts=max(0, step - hw) * bs, epochs_done=round(max(0, step - hw) * bs / n, 3),
            enc_s=round(t_enc), loss=round(run, 4), acc=round(racc, 4))
    evaluate("end")
    del opt, sch, head
    gc.collect()
    if job.dev == "cuda":
        torch.cuda.empty_cache()


def save(job, sc, tag):
    """P13's save (fp16, frozen word embeddings dropped) -> S3 ckpt/; locally (no S3) -> model_dir/ce_<tag>.pt."""
    if job.s3.on:
        return job.save(sc, tag)
    frozen = {k for k, p in sc.named_parameters() if not p.requires_grad}
    job.torch.save({k: v.half() for k, v in sc.state_dict().items() if k not in frozen}, job.out / f"ce_{tag}.pt")
    job.log(ev="saved", file=f"ce_{tag}.pt")


def chain(job):
    """One chain with job.P: fresh bge -> [MLM] -> [fine-tune] -> holdout band -> [tests]."""
    P = job.P
    tok, sc, pad = job.load("bge")
    job.selfcheck("bge", tok, sc, pad)
    if P["dapt"]:
        mlm(job, tok, sc, f"dapt_{P['dapt']}")
        if P["save_dapt"]:
            save(job, sc, f"dapt{P['dapt']}")
        job.flush_log()
    tag = P["tag"]
    if P["ft"]:
        job.train("bge", tok, sc, pad, tag, list(P["ft"]), float(P["ft_cap_s"]))
    job.holdout("bge", tok, sc, pad, tag)
    if P["save_ft"]:
        save(job, sc, tag)
    job.test("bge", tok, sc, pad, tag, list(P["tests"]))
    job.log(ev="chain_done", tag=tag, s=round(time.time() - B.T0))
    job.flush_log()


def run(job):
    """params['chains'] = [ {dapt, ft, tag, tests, ...}, ... ] run in order (each on a fresh model), else one chain."""
    base = dict(job.P)
    for ch in base.get("chains") or [{}]:
        job.P = {**base, **ch}
        chain(job)
        gc.collect()
        if job.dev == "cuda":
            job.torch.cuda.empty_cache()
    job.log(ev="all_done", s=round(time.time() - B.T0))
    job.flush_log()


def gpu_defaults():
    """Big cards (H100 80 GB, L40S 48 GB): the same per-example recipe (bs 128 fine-tune, bs 256 MLM), just without
    gradient accumulation and with bigger scoring batches. Smaller cards: P13's A10G settings. Explicit params win."""
    import torch
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    return ({"micro": 128, "dapt_micro": 256, "score_bs": 2048} if gb >= 40 else {}), round(gb, 1)


def main(train_dir, test_dir, model_dir, out_dir, params):
    import importlib.metadata as md
    import os
    import torch
    os.environ.setdefault("HF_HOME", str(Path(params.get("checkpoint_dir", "/tmp")) / "hf"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    vers = {k: md.version(k) for k in ("torch", "transformers", "tokenizers", "pandas", "pyarrow", "numpy", "scikit-learn")}
    print(f"[p32] cuda {torch.version.cuda} gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'} "
          f"{vers}", flush=True)
    known = set(DEF32) | set(B.DEFAULTS) | {"checkpoint_dir", "chains"}
    for p in [params] + list(params.get("chains", [])):
        assert known >= set(p), f"unknown params {set(p) - known}"          # a typo must not silently fall back
    auto, gb = gpu_defaults()
    params = {**DEF32, **auto, **params}
    q = {**B.DEFAULTS, **params}
    print(f"[p32] gpu memory {gb} GB -> micro {q['micro']}, dapt_micro {q['dapt_micro']}, score_bs {q['score_bs']}", flush=True)
    job = B.Job(train_dir, model_dir, params)
    job.log(ev="start", params=job.P, versions=vers, files=sorted(p.name for p in Path(train_dir).glob("*.parquet")))
    job.flush_log()
    try:
        run(job)
    finally:
        job.flush_log()


if __name__ == "__main__":        # local / portable kit: python solution_p32.py DATA_DIR OUT_DIR '<json params>'
    import sys
    Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
    main(sys.argv[1], None, sys.argv[2], sys.argv[2], {"checkpoint_dir": sys.argv[2], **json.loads(sys.argv[3])})
