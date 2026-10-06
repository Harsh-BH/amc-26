# P67-fr-synth — RESULT (DRAFT 15:35; strict re-run pending)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **A bge CE tuned on synthetic pairs made from the target country's own S1 records passes the LOCO gate, as an extra column in P55's combiner.**
  - v1: India→US **+0.00191 ± 0.00013**, US→India **+0.00236 ± 0.00014** France-F0.5 at the shipped cell. v2: +0.00161 / +0.00206.
  - An untuned-CE control column gives only +0.00006 / +0.00081, so the gain comes from the synthetic tuning.
  - The tuned CE is *worse* on its own (band AUC 0.94–0.96 → 0.90–0.92). The combiner uses it as a second, decoy-strict opinion.
- On the final chain (P55 veto only, no adds), the useful lever is **P55's add arm with the new combiner, gated by France's Qwen r ≥ .5**: LOCO **+0.00308 / +0.00437** over the chain analogue. The old combiner's adds give +0.0024 / +0.0028.
- **Upload candidate `output/sub67q`** = sub66_final + 2,793 France pairs (India/US byte-identical). Expected LB ≈ +0.0001. Risk: 78% of these pairs are in sub54's loose France set, which sub66_final dropped.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Files** (all on `output/sub66_final`, candidates = sub66_final's hard link; every added pair is also in `sub66_final_small`'s candidates, checked):

| folder | France change | validator `--check-ids` (vs sub66_final_small candidates) | md5 |
|---|---|---|---|
| **`sub67q`** (recommended) | +2,793 pairs on 2,669 S1: extra same-source 2,518, fill-source 222, fill-empty 53 | p67-valq-0927-1525 (pending) | d7199116… |
| `sub67` | +3,302 pairs on 3,147 S1, no Qwen gate | p67-val-0927-1516 PASS | a610c9ac… |
| `sub67a` | sub67 + the p55n veto swap (−299 / +282 restored) | PASS | eb3b1bba… |
| `sub67v` | the veto swap only (579 S1) | PASS | d40ac456… |

- **Rebuild** (≈ 20 s each, ≤ 1.4 GB):
  1. `ev67.py frscore bge p67-bge-loco-0927-1405 p67-bge-fr-0927-1405 syn` → `data/work/p55/cf_p67bge.npy`.
  2. `build67.py sub66_final sub67 p67bge sub66_final_small 0.95 noswap`.
  3. `build67.py qgate sub66_final sub67 sub67q`.
- **Rule** (unseen country; a final-chain stage after P66, before the last dup):
  - c = P55's 3-seed pooled bag on P55's 20 features + `r_syn`. r_syn = logit r of the pooled bge CE after 1 epoch on synthetic France pairs + India/US replay; band rows only, NaN elsewhere.
  - Add free own argmax pairs with c > .95, best first:
    - caps 5 / 6;
    - skip an SX already listed for any France S1;
    - skip P64 legal-form conflicts (94 skipped);
    - the pair must be a candidate in both candidate files.
  - Then drop the adds whose France Qwen r (P38 band file) is < .5 (509 dropped); unscored adds are kept.
- **Weights:** France CE = `s3://…/amc26-p67/results/p67-bge-fr-0927-1405/France/ckpt/ce_syn_France.pt` (bge-reranker-v2-m3 format of P13's save()). Band r = `…/France/r/r_syn_fb_France.parquet`. The generator for a src port is `gen67.py gen France 70000 v1`.

## Evidence         (table: ledger name | world | key metrics | verdict)
France-F0.5 dF on P55's LOCO frames; "A→B" = fit on A's labels, test on B; ± = SE over target S1.
| ledger name | world | key metrics (India→US / US→India) | verdict |
|---|---|---|---|
| (pipeline check) | LOCO India | untuned 'pre' scores vs the shipped LOCO CE r: corr 0.99996, median abs diff 0 | pass |
| p67-loco-bge-syn (job p67-bge-loco-0927-1405, v1) | holdout-loco | ship cell syn−base **+0.00191 ± 0.00013 / +0.00236 ± 0.00014**; pre−base +0.00006 / +0.00081; syn−pre +0.00185 / +0.00155; veto06 syn−base +0.00125 / +0.00115; band AUC r_syn 0.910 / 0.901 (pre 0.939 / 0.956) | **keep** |
| p67-loco-bge-syn (job p67v2-bge-loco-0927-1432, v2) | holdout-loco | ship syn−base +0.00161 ± 0.00012 / +0.00206 ± 0.00013; syn−pre +0.00155 / +0.00125; AUC r_syn 0.922 / 0.918 | keep (v1 ≥ v2) |
| cell67_bge.json | holdout-loco | vs the final chain's analogue (p55n veto, rescue exempt): veto swap +0.00099 ± 0.00008 / +0.00095 ± 0.00007; adds (syn c > .95) on the old veto **+0.00303 / +0.00398** (prec .938 / .895); old-combiner adds +0.00236 / +0.00278; swap + adds +0.00403 / +0.00493 | adds keep |
| qgate67.json | holdout-loco | adds with source-only Qwen r < .5 are 64% / 49% precise (227 / 470 pairs); gated adds **+0.00308 ± 0.00008 / +0.00437 ± 0.00013** | keep → sub67q |
| check_sub67.json | holdout-loco | add precision by type: extra .938 / .912, fill-source .940 / .855, fill-empty .953 / .659 | — |
| audit_sub67*.json, check_sub67*.json | test-France | sub67 adds: Qwen r ≥ .5 83%, median 0.993 (extra 85% / 0.994, fill-source 54% / 0.79). sub67v removes: Qwen 87% ≥ .5 (disagrees), restores median 0.51. sub67q: 2,176 of 2,793 adds are in sub54's 40,888 France pairs that sub66_final lacks | sub67q |
| p67-loco-wf-wf | holdout-loco | wordfreq 4 columns: +0.000067 ± 0.000087 / +0.000923 ± 0.000157 | drop |

## What failed      (approaches tried and dropped, with the number that killed them)
- **wordfreq distinctive-token overlap:** India→US +0.00007, under the +0.0003 bar. (That ledger row's hypothesis text wrongly says "CE"; its name and config are right.)
- **The veto swap:** only +0.00003 LB expected. France's Qwen disagrees with 87% of its removes, so it is not recommended on its own.
- **The v1 recipe as a standalone CE:**
  - On India's frame, 3,114 true band pairs flip from r > .7 to r < .3. They are real true pairs with house-number typos.
  - 3,181 false pairs flip the other way: branch decoys with a generic word added.
  - It is only useful beside the transfer CE inside the combiner.
- **Qwen arm:** two LOCO jobs waited for g6e capacity (4xlarge / 8xlarge / 16xlarge) for more than an hour; none started.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `gen67.py gen C N v1|v2|v1h` → `data/work/p67/sm/syn_C_ver.parquet` (25–55 s, ≤ 1.4 GB). `sanity C` / `self`.
- `prep67.py train|frames` (env `P67_VER`) → the channel `data/work/p67/sm/ch*/` = S3 `amc26-p67/ch*/`.
- `launch67.py bge|qwen NAME INSTANCE RUNS [channel]`; `solution_p67.py` (bge) / `solution_p67q.py` (Qwen).
  - One bge run ≈ 16 min on g5: 240k pairs at 293 pairs/s, band scoring at 1,200–1,700 pairs/s.
- `ev67.py loco ARM JOBS…` (≈ 2–6 min, 0.9 GB) · `frscore ARM JOBS… FS` · `cell67.py` · `qgate67.py` · `check67.py BASE NEW` · `audit67.py BASE NEW` · `build67.py …` · `wf67.py feats`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Rules:**
  - Only the provided records are used, plus allowed small hand-written dictionaries (French/US/Indian legal suffixes, street types, ~10 generic business words, house-number labels).
  - The France département-slot map is learned from test SX (test-side stats).
  - Op rates are P40's India/US catalogue.
  - Models: bge-reranker-v2-m3 (Apache-2.0, 568M) and Qwen3-Reranker-4B (Apache-2.0, 4.02B), both already shipped. wordfreq (Apache-2.0) was installed in .venv and only tried.
- **LOCO honesty:** the LOCO generator used the target country's own op rates (learned from its labels). The strict re-run with the source country's rates (p67h-bge-loco-0927-1501) is pending. The decoy design was read off US band-false pairs, so US→India is the direction clean of it.
- **LB risk:**
  - 88% of the adds are extra same-source members.
  - 78% are in sub54's France pairs that sub66_final no longer has.
  - LOCO says the c > .95 extras are 91–94% precise. The LB never tested these pairs alone.
- **Scaling:** France gets 4–6× fewer adds per S1 than LOCO, because the France chain already lists many of them. Expected LB = LOCO dF × (France adds / LOCO adds per S1) × 0.1498 ≈ +0.00010 to +0.00012.

## Open questions   (for the master; each with your best guess)
- sub67q vs sub67? sub67q: the Qwen gate is LOCO-positive in both directions, and France's own CE is Qwen.
