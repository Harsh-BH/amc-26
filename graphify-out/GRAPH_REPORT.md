# Graph Report - .  (2026-09-24)

## Corpus Check
- Corpus is ~9,174 words - fits in a single context window. You may not need a graph.

## Summary
- 103 nodes · 189 edges · 12 communities (9 shown, 3 thin omitted)
- Extraction: 96% EXTRACTED · 3% INFERRED · 1% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.74)
- Token cost: 147,266 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Problem context, metric & docs|Problem context, metric & docs]]
- [[_COMMUNITY_Offline test suite|Offline test suite]]
- [[_COMMUNITY_Container entrypoint & hyperparameters|Container entrypoint & hyperparameters]]
- [[_COMMUNITY_AWS account setup & quota|AWS account setup & quota]]
- [[_COMMUNITY_TF-IDF baseline|TF-IDF baseline]]
- [[_COMMUNITY_Submission preflight|Submission preflight]]
- [[_COMMUNITY_Image resolution & local runs|Image resolution & local runs]]
- [[_COMMUNITY_Source packaging & container deps|Source packaging & container deps]]
- [[_COMMUNITY_Job monitoring|Job monitoring]]
- [[_COMMUNITY_Submission retry|Submission retry]]
- [[_COMMUNITY_Request validation|Request validation]]
- [[_COMMUNITY_Artifact fetch|Artifact fetch]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 19 edges
2. `test_sm.py offline framework test suite (29 tests)` - 15 edges
3. `README.md (AMC 2026 project overview)` - 15 edges
4. `EMBEDDINGS.md (embedding model selection guide)` - 10 edges
5. `build_request()` - 9 edges
6. `CLAUDE.md (project rules for agents)` - 9 edges
7. `SAGEMAKER.md (training framework guide)` - 9 edges
8. `load_params()` - 8 edges
9. `main()` - 8 edges
10. `run()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `TF-IDF + LightGBM baseline in log-space` --semantically_similar_to--> `Freeze backbone, cache embeddings, iterate on the head`  [INFERRED] [semantically similar]
  baseline.py → docs/EMBEDDINGS.md
- `image_uri()` --implements--> `boto3 submits jobs; sagemaker SDK only resolves the DLC image URI`  [EXTRACTED]
  sm_launch.py → CLAUDE.md
- `quota_advice()` --implements--> `SageMaker training quota starts at 0 on new accounts`  [EXTRACTED]
  sm_launch.py → docs/SAGEMAKER.md
- `build_request()` --implements--> `Spot training on by default, with checkpointing`  [EXTRACTED]
  sm_launch.py → docs/SAGEMAKER.md
- `run_local()` --implements--> `--local: free rehearsal in the real DLC container`  [EXTRACTED]
  sm_launch.py → docs/SAGEMAKER.md

## Hyperedges (group relationships)
- **Laptop-to-container SageMaker training pipeline** — amc_26_sm_launch_pack_source, amc_26_sm_launch_build_request, amc_26_sm_train_main, amc_26_sm_train_load_params [INFERRED 0.85]
- **SageMaker quota-zero problem documented and preflighted across code and docs** — amc_26_sm_launch_quota_advice, claim_credits_md, sagemaker_md [INFERRED 0.85]
- **Hour-0 workflow: baseline signal then submission preflight** — readme_md, amc_26_baseline_run, amc_26_submit_main [INFERRED 0.80]

## Communities (12 total, 3 thin omitted)

### Community 0 - "Problem context, metric & docs"
Cohesion: 0.12
Nodes (23): AMC 2023 problem: product length from tabular features, AMC 2024 problem: entity value extraction from product images, AMC 2025 problem: Smart Product Pricing (multimodal, SMAPE), demo(), Competition metric. AMC 2025 scored on SMAPE; confirm against the 2026 problem P, SMAPE as a percentage, 0..200. Lower is better., smape(), AWS Free Tier $200 credit program (signup + 5 activities) (+15 more)

### Community 1 - "Offline test suite"
Cohesion: 0.15
Nodes (10): _(), args(), _describe(), FakeLogs, FakeQuotas, Offline tests for sm_train.py / sm_launch.py. No AWS calls, no framework: `pytho, Run sm_train.py in a subprocess inside a fake /opt/ml, exactly as the container, Two pages of events, then CloudWatch's end-of-stream: empty page, same token. (+2 more)

### Community 2 - "Container entrypoint & hyperparameters"
Cohesion: 0.24
Nodes (11): build_request(), channel(), load_params(), _ls(), main(), parse_params(), SageMaker entry point. Problem-agnostic: it owns the SageMaker filesystem contra, --key value pairs; values JSON-decoded so '3' -> 3, 'true' -> True. Bare --flag (+3 more)

### Community 3 - "AWS account setup & quota"
Cohesion: 0.33
Nodes (9): create_role(), ensure_bucket(), main(), quota(), quota_advice(), Submit and manage SageMaker training jobs. boto3 builds and submits the request;, (value, code) for '<instance> for [spot ]training job usage'. New accounts are o, role_arn() (+1 more)

### Community 4 - "TF-IDF baseline"
Cohesion: 0.33
Nodes (8): demo(), featurize(), pick_target(), pick_text(), Text -> numeric-target baseline: TF-IDF (word + char) into LightGBM, scored with, Synthetic sanity check: the model must beat a median guess on a learnable signal, Object columns that look like prose, longest average first. Skips ids and urls., run()

### Community 5 - "Submission preflight"
Cohesion: 0.39
Nodes (7): check_doc(), check_preds(), collect(), main(), page_count(), Pack + pre-flight the Round 1 submission.  Rules (Unstop, Stage 2): a 1-2 page a, Pages via poppler; None if we cannot tell (then a human must check).

### Community 6 - "Image resolution & local runs"
Cohesion: 0.33
Nodes (6): boto3 submits jobs; sagemaker SDK only resolves the DLC image URI, image_uri(), is_cpu(), Run the job in the real SageMaker container via Docker, with the same /opt/ml la, run_local(), sess_ecr_token()

### Community 7 - "Source packaging & container deps"
Cohesion: 0.4
Nodes (6): pack_source(), Return (tar.gz bytes, member names). Top-level *.py, solution override, requirem, --local: free rehearsal in the real DLC container, SAGEMAKER.md (training framework guide), SageMaker training quota starts at 0 on new accounts, sm_requirements.txt (container pip requirements)

### Community 8 - "Job monitoring"
Cohesion: 0.67
Nodes (4): logs(), summary(), wait(), test_sm.py offline framework test suite (29 tests)

## Ambiguous Edges - Review These
- `CLAIM_CREDITS.md (AWS Free Tier credit guide)` → `Compute plan: AWS for data/orchestration, Kaggle/Colab for training loop`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to

## Knowledge Gaps
- **28 isolated node(s):** `Submit and manage SageMaker training jobs. boto3 builds and submits the request;`, `(value, code) for '<instance> for [spot ]training job usage'. New accounts are o`, `Return (tar.gz bytes, member names). Top-level *.py, solution override, requirem`, `Check against botocore's own schema - catches typos before anything is spent.`, `A role created seconds ago is not assumable yet (IAM is eventually consistent).` (+23 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `CLAIM_CREDITS.md (AWS Free Tier credit guide)` and `Compute plan: AWS for data/orchestration, Kaggle/Colab for training loop`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `README.md (AMC 2026 project overview)` connect `Problem context, metric & docs` to `Container entrypoint & hyperparameters`, `AWS account setup & quota`, `TF-IDF baseline`, `Submission preflight`, `Source packaging & container deps`, `Job monitoring`?**
  _High betweenness centrality (0.415) - this node is a cross-community bridge._
- **Why does `test_sm.py offline framework test suite (29 tests)` connect `Job monitoring` to `Problem context, metric & docs`, `Offline test suite`, `Container entrypoint & hyperparameters`, `AWS account setup & quota`, `Image resolution & local runs`, `Source packaging & container deps`, `Submission retry`, `Request validation`, `Artifact fetch`?**
  _High betweenness centrality (0.297) - this node is a cross-community bridge._
- **Why does `run_train()` connect `Offline test suite` to `Job monitoring`, `Container entrypoint & hyperparameters`?**
  _High betweenness centrality (0.217) - this node is a cross-community bridge._
- **What connects `Submit and manage SageMaker training jobs. boto3 builds and submits the request;`, `(value, code) for '<instance> for [spot ]training job usage'. New accounts are o`, `Return (tar.gz bytes, member names). Top-level *.py, solution override, requirem` to the rest of the system?**
  _28 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Problem context, metric & docs` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._