# P78-final-pkg2 — RESULT
## TL;DR
- **sub76 (LB 0.988292) and sub77 both rebuild BYTE FOR BYTE from src.** The chain is `run.py --recipe sub66_final --drop-sibling-cands` → `src/final_build.py`, which applies 6 France change sets (22,562 rows). md5: sub76 a7b2cfe0…, sub77 9e4fb898…, candidates c8850dee…. Per-step counts equal the lead's hand build, and it runs in 21 s at 0.5 GB.
- **`output/final/Yeti_submission.zip` = sub76** (32 files, 94.2 MB, md5 520204eb…). It contains `src/final_build.py` and `src/final_changes.parquet`. The unzipped copy rebuilds sub76 byte for byte from its own src. The old zip is in `output/final/archive/Yeti_submission_sub66_final.zip`. **SageMaker `--check-ids` on the unzipped zip: PASS** (p78-val-zip76, 48 s billable; TSV md5s match sub76).
- Recommendation: this zip is the final. If sub77 beats 0.988292, re-zip with the 3 commands in "Re-zip for sub77" (≈ 5 min + 3 min SageMaker).

## Recommendation
- Final build (documented in SETUP.md §7, code README, Documentation_template 4.11 / §6):
  ```
  run.py --split test --stage model --recipe sub66_final --drop-sibling-cands --out output/sub66_final_small
  final_build.py output/sub66_final_small output/sub76            # --upto 41 -> sub77
  package.py --team Yeti --out output/sub76 --dest output/final
  ```
- `final_build.build(base, out, changes, upto="71a") -> info` writes both TSVs and `final_build_info.json` (per-step counts, md5s).
  - The steps are `67q, 70a, t975ce, n54 (removal), 71a, 41`.
  - This is combine.py's logic with the relative-to-sub66_final add mode folded in: one owner per SX (a counter, so dup-step SX that sit on two lines stay owned after one removal), caps 5 / 6, and a pair on the sub66_final line or already added is skipped. Candidates get missing matches appended after each add step.
- `package.py` now also zips `final_changes.parquet` (imported path `final_build.CHANGES`: `src/` first, then `data/work/p78/`).
- ponytail: the change sets ship as rows, not code. Each came from a probe's GPU model or frame run by hand (P67 CE + Qwen gate, P70 combiner, P69 t .975 frame, N54 lens frame, P71 word list). `extract.py` re-derives them from the staged folders. SETUP §7 lists each probe's own build command. The upgrade path is to port each stage into france.py as a run.py recipe step.

### Re-zip for sub77 (if the lead says it won)
```
.venv/bin/python probes/P78-final-pkg2/rezip77.py <sub77 LB> --dry   # all 22 doc anchors present once (checked 17:20)
.venv/bin/python probes/P78-final-pkg2/rezip77.py <sub77 LB>         # ~6 min
git add README.md APPROACH.md Documentation_template.md SETUP.md SUBMISSIONS.md code/business_entity_resolution/README.md ledger/experiments.jsonl && git commit && git push
```
The script does the following in order. It refuses an LB ≤ 0.988292.
1. It edits [FINAL LB] everywhere to sub77, and adds a P41 row to Documentation_template §4.11 and the ablation table.
2. It archives the sub76 zip as `output/final/archive/Yeti_submission_sub76.zip` and runs `package.py --out output/sub77`.
3. It runs the SageMaker `--check-ids` job p78-val-zip77, and stops if that fails.
4. It copies the zip to S3 and Drive `final/` (the sub76 zip goes to `archive/` on both first), writes the zip md5 and PASS into SETUP / SUBMISSIONS, and adds a ledger row.

## Evidence
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| P78/final-build-sub76 | test | matching md5 a7b2cfe0 = output/sub76; cands c8850dee = sub75a/76/77; 67q +2793, 70a +424, t975ce +4904 / 1 rej, n54 −2120, 71a +4523 | keep |
| P78/final-build-sub77 | test | matching md5 9e4fb898 = output/sub77; 41 +5824 / 0 rej; cands unchanged | keep |
| P78/final-zip-sub76 | test | zip 32 files, 94.2 MB, md5 520204eb; unzipped TSVs = sub76; unzipped src rebuilds sub76 byte for byte; self-checks final_build / package / metric pass; SageMaker --check-ids PASS (p78-val-zip76) | keep |

## What failed
- Nothing was dropped. Regenerating each add set from probe code was not attempted in the time budget: P67 needs GPU CE and Qwen scores, and N54 needs p74_frame.npz (106 MB). They ship as change-set rows instead, and the rows are documented as such.

## Interfaces
- `code/business_entity_resolution/src/final_build.py BASE OUT [--upto STEP] [--changes P] | --selftest` (21 s, ≈0.5 GB; it streams the base and holds only the touched lines).
- `probes/P78-final-pkg2/extract.py` → `data/work/p78/final_changes.parquet` (66 s; read from the output/ staged folders, which all share sub66_final's S1 order).
- `build_info_76.json`, `build_info_77.json`, `extract_info.json`: counts and md5s.
- `rezip77.py LB [--dry]`: the sub77 swap (docs → zip → SageMaker check → backups → ledger), about 6 min.

## Risks
- Anyone reproducing needs `final_changes.parquet`. It is in the zip, in S3 / Drive `amc26-offload/final/work/p78/`, and in data/work/p78. It is not committed to git (data/ rule).
- sub76's and sub77's `candidate_pairs.tsv` were full copies, identical to sub75a's by md5 (cmp). They are now hard links (+240 MB of disk).
- `output/sub76/code.zip` is still sub66_final's src (no final_build.py). The final zip is the deliverable that matters.

## Open questions
- Does sub77 beat 0.988292? If so, re-zip (above). Best guess: +0.0001–0.0002, going by P41's LOCO.
