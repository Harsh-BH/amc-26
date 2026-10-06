"""P67 GPU job, Qwen arm: P38's job (p38sol.py = solution_p38.py, unchanged) run once per entry of params['stages'],
each a full P38 run with its own tag: weights from P38's S3 cache -> restore stage['ckpt'] (a P38 LoRA) -> optional
continued training on train_<c> for c in stage['train'] (capped at cap_s) -> score stage['score'] files.
  qpre_<k>  the LOCO LoRA as shipped (train [])                     -> r_qpre_<k>_fb_<frame>.parquet
  qsyn_<k>  the same LoRA + synthetic target pairs + source replay   -> r_qsyn_<k>_fb_<frame>.parquet, lora_qsyn_<k>.pt
"""
import time
from pathlib import Path


def main(train_dir, test_dir, model_dir, out_dir, params):
    import p38sol
    base = {k: v for k, v in params.items() if k != "stages"}
    for st in params["stages"]:
        p38sol.T0 = time.time()                        # each stage gets its own deadline_s
        md = Path(model_dir) / st["tag"]
        md.mkdir(parents=True, exist_ok=True)
        p38sol.main(train_dir, test_dir, md, out_dir, {**base, **st})
