"""P67 GPU job, bge arm: P13's Job (p13sol.py = solution_p13.py, unchanged) run once per entry of params['runs'].
Each run: fresh bge-reranker-v2-m3 + the fine-tuned weights at run['ckpt'] -> score run['score'] files ('pre', the
transfer CE as shipped) -> continue training 1 epoch on train_run<k> (synthetic pairs of the target country + replay
of real source pairs) -> score the same files again ('syn') -> save the weights.
  runs India   ckpt ce_bge_locoUS.pt (US labels only),  score frame_US_India  (LOCO: India pairs)
  runs US      ckpt ce_bge_locoIndia.pt,                 score frame_India_US  (LOCO: US pairs)
  runs France  ckpt ce_bge_pool.pt (the shipped CE),     score frame_France
-> <s3>/<k>/r/r_{pre,syn}_<file>.parquet (s1, sx, r) + <s3>/<k>/ckpt/ce_syn_<k>.pt + log.jsonl
"""
from pathlib import Path


def main(train_dir, test_dir, model_dir, out_dir, params):
    import gc
    import p13sol
    import torch
    for run in params["runs"]:
        k = run["k"]
        md = Path(model_dir) / k
        md.mkdir(parents=True, exist_ok=True)
        job = p13sol.Job(train_dir, md, dict(params, phases=[], s3=f"{params['s3']}/{k}"))
        job.log(ev="p67_run", run=run)
        tok, sc, pad = job.load("bge")
        job.restore(sc, run["ckpt"])
        if run.get("pre", True):
            for f in run["score"]:
                job.write(f"pre_{f}", job.data(f), job.score("bge", tok, sc, pad, f))
                job.flush_log()
        job.train("bge", tok, sc, pad, f"syn_{k}", [f"run{k}"], float(params.get("train_cap_s", 2400)))
        for f in run["score"]:
            job.write(f"syn_{f}", job.data(f), job.score("bge", tok, sc, pad, f))
            job.flush_log()
        job.save(sc, f"syn_{k}")
        job.flush_log()
        del tok, sc, job
        gc.collect()
        torch.cuda.empty_cache()
