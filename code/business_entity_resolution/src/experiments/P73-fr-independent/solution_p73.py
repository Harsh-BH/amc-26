"""P73 GPU job: P13's Job (p13sol.py = solution_p13.py, unchanged) run once per entry of params['runs'].
Each run: a FRESH public checkpoint run['model'] (no India/US-tuned weights) -> BCE fine-tune on train_ind<k>
(P67 synthetic pairs of country k + agreement pseudo-labels of k's band; no labelled pair) -> score run['score']
-> <s3>/<k>_<m>/r/r_ind_<m>_<file>.parquet (s1, sx, r); run['save'] also uploads the weights.
  runs India  score fb_US_India (India pairs, LOCO)   runs US  score fb_India_US   runs France  score fb_France
"""
from pathlib import Path


def main(train_dir, test_dir, model_dir, out_dir, params):
    import gc
    import p13sol
    import torch
    for run in params["runs"]:
        k, m = run["k"], run["m"]
        md = Path(model_dir) / f"{k}_{m}"
        md.mkdir(parents=True, exist_ok=True)
        tag = f"ind_{m}_{k}"
        job = p13sol.Job(train_dir, md, dict(params, phases=[], s3=f"{params['s3']}/{k}_{m}", bge=run["model"],
                                             lr=run["lr"], epochs={tag: run["epochs"]}))
        job.log(ev="p73_run", run=run)
        seed = int(run.get("seed", 0))                       # seed arm: offsets P13's fixed seeds (head init + batches)
        if seed:
            torch.manual_seed(seed)
            _ms, _rng = torch.manual_seed, p13sol.np.random.default_rng
            torch.manual_seed = lambda s, _ms=_ms: _ms(int(s) + seed)
            p13sol.np.random.default_rng = lambda s=None, _r=_rng: _r(int(s) + seed) if isinstance(s, (int, p13sol.np.integer)) else _r(s)
        tok, sc, pad = job.load("bge")
        job.train("bge", tok, sc, pad, tag, [f"ind{k}"], float(run.get("cap_s", 2400)))
        if seed:
            torch.manual_seed, p13sol.np.random.default_rng = _ms, _rng
        for f in run["score"]:
            job.write(f"ind_{m}_{f}", job.data(f), job.score("bge", tok, sc, pad, f))
            job.flush_log()
        if run.get("save"):
            job.save(sc, tag)
        job.flush_log()
        del tok, sc, job
        gc.collect()
        torch.cuda.empty_cache()
