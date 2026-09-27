"""P48 GPU job: P13's bge_ckpt scoring path (p13sol.py = solution_p13.py, unchanged), run once per channel sub-dir with
its own fine-tuned weights. Channel 'train' = data/work/p48/sm/: run_<k>/ = P45's 64-row stubs + one real test_<c>.
hp: runs {k: ckpt s3 uri} + P13's (deadline_s, s3). -> <s3>/<k>/r/r_bge_pool_test_<c>.parquet (s1, sx, r).
  run_India   LOCO US->India g < 0.02 argmax pairs, P30's ce_bge_locoUS.pt (source-only)
  run_US      LOCO India->US g < 0.02 argmax pairs, P30's ce_bge_locoIndia.pt
  run_France  France chain pairs without CE r, ce_bge_pool.pt (the shipped CE)
"""
from pathlib import Path


def main(train_dir, test_dir, model_dir, out_dir, params):
    import p13sol
    import torch
    for k, ckpt in params["runs"].items():
        md = Path(model_dir) / k
        md.mkdir(parents=True, exist_ok=True)
        print(f"[p48] run {k}: ckpt {ckpt}", flush=True)
        p13sol.main(Path(train_dir) / f"run_{k}", test_dir, md, out_dir,
                    dict(params, phases=["bge_ckpt"], ckpt=ckpt, s3=f"{params['s3']}/{k}"))
        torch.cuda.empty_cache()
