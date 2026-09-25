# P11-ce-big: does a better-fed or bigger cross-encoder beat e5-small on India + US? (local GPU)
Read first: probes/P4b-rerank/RESULT.md, probes/P4-match/RESULT.md ("Improvement round"), probes/P9-integrate/RESULT.md
(CE train entrypoint), probes/MERGE-2.md. You work in your own worktree; write code only under probes/P11-ce-big/
(P9 edits src/, P10 works under its own dir).

Current (#5, LB 0.980017): m2_big LightGBM (data/work/p4/m2_big.txt, trained folds 2-5, early-stopped fold 1) + P4b
cross-encoder = multilingual-e5-small (MIT, 118M), 1 epoch on 778k band pairs from train fold 3 scored by a fold-2
LightGBM, max_len 64 + length buckets, blended on 0.02<p<0.98 for India/US only. Full-corpus holdout F0.5 0.9885
(report half; 0.9834 without CE). P4b's "not tried": bigger models, >1 epoch, more folds.
GPU: local RTX 3050 **6 GB**, idle; serialise GPU use with `flock /tmp/amc-gpu.lock` like rerank.py does.

Question: which change moves the holdout by >= +0.0005 over 0.9885 on the SAME yardstick (P5 bench: blend refit on
the tune half, reported on the report half, with SE)? Run in this order, ledger each run before the next:
1. **More data, same model:** e5-small trained on the band of 3-4 folds (e.g. folds 3-5 scored by the fold-2 model
   data/work/p4/m2fast_fold2.txt, which never saw them; or folds 6-9 scored by m2_big if their features exist).
   Never train on folds 0/1. Try 1 vs 2 epochs.
2. **Bigger model:** intfloat/multilingual-e5-base (278M; confirm MIT on the model card before downloading) on the best
   data from step 1. Must fit 6 GB: bf16, max_len 64, length buckets, gradient accumulation.
   Also measure its test-band scoring time (~1.5M India+US pairs): must stay under ~1 h on this GPU.
3. If a band from a model as strong as m2_big is cheap to get, check whether matching the train band to m2_big's band helps.
Time-box ~3 h of GPU. Do not touch France (CE trained on one country hurts another: LOCO -0.0016; France keeps m2_big p).

Also answer, with numbers: does capacity help (e5-base vs e5-small on identical data)? If yes, estimate what a
560M model (multilingual-e5-large, MIT) needs: VRAM (a full fine-tune won't fit 6 GB) and hours on a 24 GB A10G.
Do NOT use AWS — the master decides that.

Deliverables:
1. Ledger rows per run; RESULT.md with the table and one recommendation.
2. If a winner: data/work/p11/ce.pt and data/work/p11/prob_test_{India,US}_blend.parquet (s1, sx, p: m2_big p blended
   with the new CE) so the master can write #6 with src/submit_probs.py (France = data/work/p4/prob_test_France_big.parquet).
Memory cap 3 GB incl. the CLI, swap allowed. Disk ~8 GB free: e5-base is ~1.1 GB; keep only the final checkpoint.
Licences MIT/Apache only; no hosted APIs.
