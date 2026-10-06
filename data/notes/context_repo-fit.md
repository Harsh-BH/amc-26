Confirmed real problem via all four sources (official PDF, guidelines PDF, DRAFT_REQUIREMENTS.md, and the actual dataset files on disk, which match the PDF spec exactly: tab-separated, `entity_id/business_name/business_address/country`, France test-only, ~2.2M S1 train / ~1.7M S1 test rows). Read-only — nothing edited.

## Audit table

| File | Verdict | Why | Adapt spec |
|---|---|---|---|
| `metrics.py` | **ADAPT** (full rewrite) | Computes numeric SMAPE; real metric is set-based F_0.5 (precision/recall over predicted vs. true match-ID sets) macro-averaged per S1 entity, singletons included (empty↔empty = 1.0) | Replace `smape()` with `f_beta(pred_sets: dict[s1_id, set[str]], truth_sets: dict[s1_id, set[str]], beta=0.5) -> float`, precision/recall per S1 id (0 matches predicted+true → 1.0), macro-mean. Keep the `demo()`/assert self-check pattern, just new assertions incl. the PDF's own worked example (P=2/3, R=1.0 → 0.714) |
| `baseline.py` | **ADAPT** | TF-IDF+LightGBM *regression* on one numeric target — wrong task shape entirely | Task is now pairwise match classification (or blocking+scoring), not regression. Reuse the TF-IDF word+char vectorizer (`featurize()`) as name/address similarity features feeding an `LGBMClassifier` over `(S1,S2/S3)` candidate pairs with a 0/1 match label derived from `train_ground_truth.tsv`; drop `log_space`/`np.clip` price logic entirely. No blocking stage exists yet — that's the bigger missing piece |
| `submit.py` | **OBSOLETE** (rewrite, don't patch) | Assumes single `submission.csv`, `approach.pdf` page-count gate, and a flat code-zip — none of that matches the real package (`<team>_submission.zip` → `output/{matching_results.tsv,candidate_pairs.tsv}` + `code/business_entity_resolution/{src,README.md,requirements.txt}` + `Documentation_template.md`, **no page limit**) | Rewrite to assemble that exact structure and shell out to the already-correct official `data/student_resource/utils/validate_submission.py` rather than reinventing row/NaN checks. Drop the PDF page-count check — it actively contradicts the problem PDF ("no page limit") |
| `sm_launch.py` | **KEEP**, light ADAPT | Problem-agnostic: packs `data/train`/`data/test` dirs, boto3 submit/wait/fetch, hyperparams via JSON not argv — none of that assumes CSV or regression | `METRICS` list only greps `smape=`/`val_loss=` from stdout (line 28) — add an `f0_5=` regex so a fine-tuned matcher's score shows up as a SageMaker job metric |
| `sm_train.py` | **KEEP** | Filesystem contract (`/opt/ml/...`, checkpoints, `fit`/`predict`/`main`) has zero format assumptions; `solution.py` owns the modelling | None needed |
| `sm_requirements.txt` | **ADAPT** (one line) | `timm` is vision-model tooling; the real dataset is pure text (no images at all) | Drop `timm`. Keep `lightgbm`, `transformers`, `sentence-transformers` — all still relevant for name/address embeddings and a cross-encoder matcher |
| `test_sm.py` | **KEEP** | 29 offline tests of the launcher/entrypoint plumbing using synthetic `train.csv`/`test.csv` fixtures — tests the framework, not the problem's data shape | None needed |
| `docs/SAGEMAKER.md` | **KEEP** | Day-1 ops, quota, spot/checkpoint mechanics — orthogonal to problem type | None needed |
| `docs/CLAIM_CREDITS.md` | **KEEP** | AWS free-credit claiming steps, unrelated to the ML task | None needed. (The Bedrock-playground credit-claim step is a one-off console action to earn $20, not part of the entity-resolution pipeline — doesn't touch the external-lookup rule) |
| `docs/EMBEDDINGS.md` | **ADAPT** — contains disqualification-risk content | See below | See fix list |
| `.gitignore` | **KEEP** | Already has `*.tsv`, `output/`, `data/*` (with `!data/.gitkeep`) — Amazon's data and both output files are already covered | None needed |
| `README.md` | **ADAPT** | Wrong problem shape throughout: "Amazon catalog dataset," "1-2 page approach doc" as sole deliverable, Hour-0 commands (`train.csv`/`--target`/`submission.csv`), Files table describing SMAPE/regression tooling, and the closing guess "another multimodal catalog task is the safe bet" — now falsified | Rewrite: Business Entity Resolution across 3 TSV sources, F_0.5 macro metric, real Hour-0 commands pointing at `dataset/train/train_source{1,2,3}.tsv` + `train_ground_truth.tsv`, real deliverable (the submission zip, no page limit), drop the "safe bet" guess and the 2025-multimodal framing. Dates/team-size/compute-plan sections hold and can stay |
| `CLAUDE.md` (project) | **ADAPT** | Two stale lines: the `submit.py` command example (`--doc approach.pdf --preds submission.csv`) and "`metrics.py` assumes SMAPE from 2025. Confirm against the 2026 problem PDF" — now confirmed, should state F_0.5 directly | Update the preflight command line once `submit.py` is rewritten; replace the SMAPE-uncertainty line with "metric is F_0.5 macro over match-ID sets (confirmed from PDF)"; change "holdout SMAPE" → "holdout F_0.5" in the commit-message rule |

### `docs/EMBEDDINGS.md` — license/disqualification check

The Bedrock section (lines 61–92: `amazon.titan-embed-text-v2:0`, `titan-embed-image-v1`, `cohere.embed-v4:0`, etc.) is an **external hosted inference API called during the resolution pipeline** — the rules ban "external databases, APIs, or services to look up business identities or resolve entities." Using Bedrock to embed business names/addresses for matching is the kind of external-service dependency that risks the disqualification clause, even though it isn't a registry lookup per se. Flag as offending; recommend deleting the Bedrock section entirely rather than defending an edge case.

The local open-weight recommendations are fine on both axes I can verify (all well under 8B params; license flagged where I'm not 100% certain — verify before relying on it):
- `BAAI/bge-m3` — MIT, ~568M params. Compliant.
- `Alibaba-NLP/gte-multilingual-base` — Apache-2.0, ~305M params. Compliant.
- `Qwen/Qwen3-Embedding-0.6B` — Apache-2.0, 0.6B params. Compliant.
- `sentence-transformers/all-MiniLM-L6-v2` — Apache-2.0, 22M params. Compliant.
- Entire **Image** section (DINOv2/v3, SigLIP2, CLIP, MetaCLIP) — the real dataset has **no image column at all** (`entity_id`, `business_name`, `business_address`, `country` only). This section is now dead weight, not a compliance risk, but it's actively misleading — delete it.

## Prioritized fix list

1. **`metrics.py`** — rewrite to F_0.5 macro over match-ID sets. Everything else (baseline, sm_train metric logging) depends on trusting this number.
2. **`docs/EMBEDDINGS.md`** — delete the Bedrock section (disqualification risk) and the Image section (no images exist); keep the local text-embedding table as blocking/feature material.
3. **`submit.py`** — rewrite for the real package layout; wire in `data/student_resource/utils/validate_submission.py` instead of re-deriving validation rules.
4. **`baseline.py`** — repurpose the TF-IDF vectorizer as name/address similarity features for a pairwise classifier; there's currently no blocking stage anywhere in the repo, and blocking is what caps recall (per the PDF's own tips) — this is the actual missing piece, not just a rename.
5. **`README.md` / `CLAUDE.md`** — text-only edits, low risk, do last.
6. `sm_requirements.txt` — drop `timm` (trivial, bundle with #4).

Scale note for whoever builds the blocking stage: train is ~2.2M S1 / ~5M S2 / ~5.3M S3 rows, test ~1.7M S1 / ~4.9M S2 / ~5.1M S3 — brute-force pairwise comparison is ~10^13 pairs, blocking keys (PIN/phonetic/TF-IDF-shortlist) are load-bearing, not optional.