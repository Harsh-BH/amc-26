# P79-p73-takeover — RESULT (final, 19:00)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- None of P73's six later arms beats its incumbent, ele4n (electra, 4 epochs, v3 synthetic + noise list). The rule was ≥ +0.0003 US-target and ≥ 0 India-target, both measured against the incumbent. The six arms: bge-base, bge-large, mxbai-base, mMiniLM, electra 6 epochs, and replay. The closest is p67+ele4n+ele6n at +0.00003 / −0.00002.
- **output/sub79q** = sub77 − 159 P73 vetoes + 167 P73 adds (Qwen-gated like 67q). Validator `--check-ids` PASS, md5 d409c956. In a paired LOCO test over sub77's 67q state it gains +0.00175 / +0.00201 France-F0.5; ungated sub79 gains +0.00172 / +0.00165.
- Expected LB ≈ **+0.00002–0.00004 over sub77** (0.988292). Upload it only if a slot is free. Do not ship the adds-only sub79a: the vetoes carry most of the value.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- Candidate upload: `output/sub79q` (candidates + code.zip = sub77's hardlinks). It rebuilds byte-for-byte with:
  `probes/P79-p73-takeover/final_build.py output/sub66_final_small output/sub79q --upto 73q`.
  That command reads `data/work/p79/final_changes.parquet`, a superset of P78's parquet: `--upto 41` still gives sub77 (9e4fb898), and `--upto 73x` gives sub79 (a28f09a4).
- Promote (master): `cp probes/P79-p73-takeover/final_build.py code/business_entity_resolution/src/final_build.py`.
  - The copy adds steps `73v` (remove), `73q` and `73x` (adds), plus the lead's `--skip`. Selftest ok.
  - From src/ it finds `data/work/p79/final_changes.parquet` through its parent directories.
  - If a zip ships the parquet next to final_build.py, ship p79's parquet, not p78's.
- Choosing between the three files (LOCO France-F0.5 of the new part over sub77's 67q state, India→US / US→India):

  | file | change on France vs sub77 | LOCO new part | md5 |
  |---|---|---|---|
  | **sub79q** | −159 vetoes, +167 adds (Qwen ≥ .5 or unscored) | **+0.00175 / +0.00201** | d409c956 |
  | sub79 | −159, +497 (ungated, P73's sub73e recipe) | +0.00172 / +0.00165 | a28f09a4 |
  | sub79a (lead) | +494 adds, no vetoes | add part only: +0.00063 / +0.00005 | 2354afe5 |

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| p73-loco-independent-ce_mml / _batch2 (probe P79) | holdout-loco | incumbent p67+ele4n **+0.002804 / +0.003059**, reproducing loco73_ep4 | — |
| p79-gate-vs-incumbent-mml | holdout-loco | p67+ele4n+mmln −0.00004 / +0.00024; p67+mmln −0.00052 / +0.00024 | drop |
| p79-gate-vs-incumbent-batch2 | holdout-loco | best 3-column vs incumbent: +ele6n +0.00003 / −0.00002, +bgbn −0.00003 / −0.00021, +mxbn −0.00013 / +0.00022, +bgln −0.00012 / −0.00019, +ele4r −0.00044 / +0.00001. p67+ele4r (replay) −0.00058 / **+0.00105**: other-country labels help India and hurt US | drop |
| p79-inc-over-67q-p67+ele4n | holdout-loco | new part +0.001715 / +0.001645; veto part +0.00109 / +0.00160; add part +0.00063 / +0.00005; Qwen-gated adds +0.00175 / +0.00201; gated and veto-protected (q ≥ .5 kept) +0.00145 / +0.00201 | keep (gated) |
| same row, new-add precision | holdout-loco | q ≥ .5: 0.855 / 0.967; unscored: 0.951 / 0.851; **q < .5: 0.662 / 0.506**; q < .5 and hn-equal: 0.786 / 0.49 (break-even ≈ F/1.25 ≈ 0.76) | — |
| same row, new vetoes | holdout-loco | 1,040 / 855 pairs, true share 0.303 / 0.153; q ≥ .5 vetoes 0.199 / 0.757 true (186 / 37 pairs) | — |
| p79-stage-sub79q | test | sub79q +167 / −159 (326 S1, 5 emptied, 4 filled), links/S1 3.2182 → 3.2183 (truth 3.46); sub79 +497 / −159 (653 S1); validator --check-ids PASS both (p79-val-0927-1752); 0 N54 re-adds (151 kept out); ids exist; all adds in candidates; unique SX | keep |

France audit (check_sub79*.json, hnfr79_sub79.json):
- Qwen on sub79's 497 adds: median 0.33, and 70% are below .5. sub79q's 167 adds: median 0.96.
- The 159 removes have Qwen median 0.97; 75% are ≥ .5, against 18% / 4% of LOCO's new vetoes. This composition shift is the main transfer risk.
- P52 hn_rel hn-equal share: adds 0.646 (q ≥ .5: 0.709, q < .5: 0.597); removes 0.522.

Budget (budget79.py): P73 + P79 spent **$14.63 since 17:26 of the $45 cap**; lifetime of all P73/P79 jobs $27.64.
- Stopped for capacity, cost or rule: p73c-mxl ($6.8/h) and p73c-bgl ($15.75/h), both Pending; p73d-mxl, Pending 17:34–18:35; p73e-mxl, Pending, stopped 18:42.
- p73c-mxb (69 min) and p73c-ele6 (61 min) finished past the 60-min rule while my session was down (network error 18:00, reboot 18:31). Together ≈ $1 over.

## What failed      (approaches tried and dropped, with the number that killed them)
Every backbone and epoch variant loses to ele4n on the paired LOCO gate (numbers above):
- bge-reranker-base: 4 epochs, MIT.
- bge-reranker-large: 2 epochs, MIT.
- mxbai-rerank-base: 4 epochs, Apache-2.0.
- mMiniLM-L12: 4 epochs, Apache-2.0.
- electra: 6 epochs.
- Replay (v3 synthetic + 60k other-country labelled pairs).

Larger models do not help. mxbai-large never got LOCO capacity; its France scores exist (p73d-mxlfr), but it cannot be gated.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `build79.py FS`: build67 swap @ .95 on sub66_final with `cf_p73<FS>.npy` → change rows (73v / 73q / 73x, N54 pairs kept out) → `data/work/p79/final_changes.parquet` → output/sub79q + output/sub79. About 1.5 min, < 1 GB.
- `final_build.py`: src's, plus steps 73v / 73q / 73x and `--skip`. Selftest covers the veto-frees-SX case.
- `inc79.py FS [JOB..]`: paired LOCO value of an arm's new part over the 67q state, plus Qwen × hn splits. About 3 min, ~1 GB.
- `loco79.py TAG ARMS JOB..`: ev73.loco with trimmed FSETS plus a paired incumbent gate; `loco79.py vs TAG ARMS` recomputes from the JSON. About 30 s per feature set, 0.6 GB.
- `check79.py BASE NEW` (staging checks, Qwen audit); `hnfr79.py BASE NEW` (France hn-equal by Qwen band); `budget79.py` ($ per job; flags jobs over 60 min and still Pending after 18:30).
- `launch79.py` (launch73 with a 65-min MaxRuntime; skips the rate-limited quota lookup); `val79.sh SUB..` (validator `--check-ids` on SageMaker ml.m5.xlarge).
- France scores: `data/work/p55/cf_p73p67+ele4n.npy` (ev73 frscore, 34 s, 0.63 GB).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Licences: electra-base (Apache-2.0, 109M), the incumbent; all tried backbones are MIT/Apache-2.0 and ≤ 560M. No external data. Pseudo-labels come from India/US models (P61 rule).
- Transfer:
  - France's new part is ~10× smaller per S1 than LOCO's (326–656 pairs over 259k S1, against 2–2.7k over 88–132k). The expected LB is a per-pair scaling of LOCO.
  - France's vetoes are mostly Qwen-endorsed, which is rare in LOCO. In LOCO, the veto-protect variant cost −0.0003 India→US and was neutral US→India, so I kept every veto.
- Disagreement on record: the team's row 79 argues Qwen is India/US-biased on France and that France's adds are hn-equal. In LOCO, hn-equality does not rescue Qwen-rejected adds (0.49 precise US→India), so I side with the Qwen gate, which is also sub77's own 67q rule.

## Open questions   (for the master; each with your best guess)
- sub79q vs sub78g / sub80 for the last slots? Guess: sub79q is worth ≈ +0.00003 at best, which is below LB noise. Upload it only after the higher-expected files.
