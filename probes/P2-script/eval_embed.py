"""(d) multilingual embedding cosine, raw Indic query name vs Latin S1 name, dev world.

    flock /tmp/amc-gpu.lock .venv/bin/python probes/P2-script/eval_embed.py intfloat/multilingual-e5-small
"""
import sys, time, json, os
sys.path[:0] = ["probes/P2-script", "code/business_entity_resolution/src"]
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from eval_names import cross_script, hard_negatives, _scores, peak_gb

name = sys.argv[1]
roman = len(sys.argv) > 2 and sys.argv[2] == "dict"
q, s1 = cross_script()
neg = hard_negatives(q, s1)
m = SentenceTransformer(name, device="cuda", model_kwargs={"torch_dtype": torch.float16})
pre = "query: " if "e5" in name else ""
qs = q.business_name
if roman:
    from translit import to_latin
    qs = to_latin(qs)
t = time.time()
enc = lambda xs: m.encode([pre + x for x in xs], batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
Q, S = enc(qs.tolist()), enc(s1.business_name.tolist())
dt = time.time() - t
res = _scores(lambda a, b: (Q[a] * S[b]).sum(1), lambda lo, hi: Q[lo:hi] @ S.T, q, s1, neg)
res["s"] = round(dt, 1); res["strings_per_s"] = round((len(Q) + len(S)) / dt)
key = name.split("/")[-1] + ("+dict" if roman else "")
print(key, res, "peak", peak_gb())
out = "probes/P2-script/eval_names.json"
r = json.load(open(out)); r[key] = res; json.dump(r, open(out, "w"), indent=1)
