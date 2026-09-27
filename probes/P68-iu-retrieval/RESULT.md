# P68-iu-retrieval — RESULT (27 Sep ~14:25 IST; India done, US paused for P43's final package)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Recovering India/US retrieval misses does not pay: the missed records get retrieved, but the matcher can't confirm them.** On the India holdout (50% of the SX files), two extra searches for addressed SX add 1,265,831 new pairs. Only **244 are true**, and only **38** of those reach m2 p ≥ 0.71. The searches are P3's own view at ranks 2-4 (P3 keeps the top 2) and an address-only view at ranks 0-4.
- **India gate (add-only on free SX, t picked on half 0):** h1 **+0.00004 ± 0.00002 F0.5**, i.e. ≈ +0.00002 LB-eq on the sample, and ≈ +0.00004 at full volume. The adds are 90-100% precise, but there are only 20-59 of them.
- **Recommendation: no India file.** A US holdout run was started and killed at P43's request. The master decides whether to rerun it (≈ 30 min CPU, then ≈ 1 h for the test side). My best guess for the US is ≤ +0.0001 LB.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Adopt nothing for India.**
- **US: optional.** If a CPU window opens after P43, run the US holdout first. Only if `gate_final` gives h1 ≥ 2 SE, run the free-only test search and `feats_c` with `P68_KEEP` set from the gate's slices, then `build`:
```
P68_THREADS=15 P68_FILES=0,2,4,6,8,10 .venv/bin/python probes/P68-iu-retrieval/p68.py search train US
.venv/bin/python probes/P68-iu-retrieval/p68.py feats train US && .venv/bin/python probes/P68-iu-retrieval/p68.py gate US
P68_ALL_S1=1 P68_TAG=_all P68_ONLY_QX=data/work/p68/gate2_qx_US.parquet .venv/bin/python probes/P68-iu-retrieval/p68.py search train US
P68_TAG=_all .venv/bin/python probes/P68-iu-retrieval/p68.py feats train US && .venv/bin/python probes/P68-iu-retrieval/p68.py gate_final US
# only if it passes:
P68_FREE_BASE=output/<base> .venv/bin/python probes/P68-iu-retrieval/p68.py search test US
P68_KEEP=<slices> .venv/bin/python probes/P68-iu-retrieval/p68.py feats_c test US
.venv/bin/python probes/P68-iu-retrieval/p68.py build output/<base> output/sub68 <t>
```

## Evidence         (table: ledger name | world | key metrics | verdict)
Ledger rows are logged under the old probe name P67-iu-retrieval.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| search-train-India | holdout, SX files 0/2/4/6 (50%), hits on holdout (fold 0) S1 | 1,638,931 hits (na ranks 2-4 / addr ranks 0-4), 1.7 GB peak, ~16 min at 8 threads | — |
| feats-train-India | same | 1,265,831 new pairs not in the X files (R1 547,583 / R2 791,997). The pruner keeps (q ≥ 0.03) 1,357. m2 p ≥ 0.71: 42. **True: 244; true at p ≥ 0.71: 38** | — |
| gate-India | holdout, #8c base (P23 dump), free SX only, caps 5/6 | best new pair per free SX, p ≥ t. t 0.5: h0 +0.00006 ± 0.00003, h1 +0.00004 ± 0.00002, 59 adds at 0.90 precision. t 0.9: 24 adds at 1.00. Pruned arm equal. Slices: na ranks 2-4 carry most of it (54 of 58 adds), addr view 7-20 adds | drop |

**Why:** P23's blocking-miss tags said an address-only search finds 24% (India) / 60% (US) of addressed misses within the top 5. Retrieval does recover them, but the recovered pairs are exactly the records whose evidence is weak (renamed, heavily corrupted, chain members), so m2 scores them low. India's addresses rarely carry a house number (26% of S1), so a renamed India copy has nothing left to confirm it. The oracle loss P36 assigned to blocking misses (0.0022 LB-eq) assumes perfect acceptance and is not reachable with the current matcher.

## What failed      (approaches tried and dropped, with the number that killed them)
- India add-only retrieval tier: +0.00004 ± 0.00002 F0.5 (h1). Too small to ship.
- Not measured: re-scoring the new pairs with the bge CE (it might lift some of the 206 true pairs below 0.71). An AWS scoring job for about 1.3M pairs is out of budget for a ≤ 0.0001 ceiling.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `p68.py search SPLIT C`: P3's index with views {na: 5, addr: 5}, addressed SX only. Env options:
  - `P68_FILES`: SX file sample.
  - `P68_FREE_BASE`: test, only SX that the base does not assign.
  - `P68_ALL_S1` / `P68_ONLY_QX` / `P68_TAG`: gate pass 2.
  - Writes `data/work/p68/hits_<split><tag>_<c>.parquet`.
- `feats SPLIT C` / `feats_c SPLIT C` (chunked, for test): the exact 65 features for new pairs. Block score = P3's na cosine. S1 / SX context is computed over existing blocked pairs + new pairs, and via_r = 1. Then P10's q and m2_big's p. Writes `new_<split><tag>_<c>.parquet`.
- `gate C`: pass 1 (holdout-S1 hits only; optimistic: the best new pair is chosen among holdout S1 only). Writes `gate2_qx_<c>`.
- `gate_final C`: pass 2 (the best new pair over ALL S1).
- `build BASE OUT T [arm]`: test adds + candidate_pairs.tsv extended with the scored new pairs.
- `selftest`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- The India gate is pass 1 only, which is optimistic. It is already ≈ 0, so pass 2 would not change the verdict.
- The base is #8c's holdout decision (P23 dump), not the #54 combiner. Free-SX status outside the dump is approximated by m2 argmax ≥ 0.71.
- Test side: new pairs enter candidate_pairs.tsv (the model scores them), which costs candidates per S1.

## Open questions   (for the master; each with your best guess)
- Run the US? Only if CPU is idle after P43. Best guess ≤ +0.0001 LB, not worth crowding the final package.
