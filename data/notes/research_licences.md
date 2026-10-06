# Licence + Availability Audit — amc-26 Business Entity Resolution

Probe: **licence-audit** (feeds the master ledger for step 2/3 of the plan — model/library shortlisting under the "MIT/Apache-2.0, ≤8B params, no external APIs/registries" hard rules). All entries verified live via WebSearch/WebFetch against PyPI, GitHub LICENSE files, or HF model cards on 2026-09-25 — none taken from memory.

## Libraries

| Library | pip name | SPDX licence | Size | CPU/GPU feasibility | Verdict | Source |
|---|---|---|---|---|---|---|
| rapidfuzz | `rapidfuzz` | MIT | C++ ext, few MB | CPU-only, trivial on 16 cores | **OK** | [PyPI](https://pypi.org/project/RapidFuzz/) |
| jellyfish | `jellyfish` | MIT | pure/Rust ext, tiny | CPU-only, trivial | **OK** | [PyPI](https://pypi.org/project/jellyfish/) |
| anyascii | `anyascii` | ISC | tiny (static table) | CPU-only, trivial | **OK** | [GitHub](https://github.com/anyascii/anyascii) |
| Unidecode | `unidecode` | **GPL-2.0-or-later** | tiny | CPU-only, trivial | **BANNED** — copyleft, and user flagged it for a reason: it's not MIT/Apache and its terms could taint anything statically linked into a distributed pipeline. Use `anyascii` instead. | [PyPI](https://pypi.org/project/Unidecode/) |
| text-unidecode | `text-unidecode` | Dual: GPL/GPLv2+ **or** Artistic License | tiny | CPU-only, trivial | **RISKY** — the Artistic-License fork avoids GPL, but it's a non-standard, rarely-vetted licence; no reason to take the risk when `anyascii` (ISC) does the same job cleanly. | [GitHub](https://github.com/kmike/text-unidecode/blob/master/LICENSE) |
| indic-transliteration (`sanscript`) | `indic-transliteration` | MIT | pure Python, small | CPU-only, trivial | **OK** | [PyPI](https://pypi.org/project/indic-transliteration/) |
| aksharamukha | `aksharamukha` | **AGPL-3.0** | pure Python + data | CPU-only | **RISKY/effectively BANNED** — AGPL is viral network-copyleft; most orgs ban it outright for anything shipped in a deliverable. Prefer `ai4bharat-transliteration` or `indic-transliteration` (both MIT) for Devanagari/Telugu↔Latin. | [GitHub](https://github.com/virtualvinodh/aksharamukha) |
| ai4bharat-transliteration (IndicXlit) | `ai4bharat-transliteration` | MIT | ~11M-param NN | GPU nice-to-have, CPU fine (small transformer) | **OK** — best fit for the 18% cross-script India pairs. | [GitHub](https://github.com/AI4Bharat/IndicXlit) |
| faiss-cpu | `faiss-cpu` | MIT | index lib, N/A params | CPU-only, 16 cores plenty for 5-13M rows | **OK** | [Socket](https://socket.dev/pypi/package/faiss-cpu) |
| hnswlib | `hnswlib` | Apache-2.0 | header-only C++ | CPU-only, fast build | **OK** | [GitHub LICENSE](https://github.com/nmslib/hnswlib/blob/master/LICENSE) |
| sparse_dot_topn | `sparse-dot-topn` | Apache-2.0 | Cython ext | CPU-only, good for TF-IDF blocking at 5M+ rows | **OK** | [GitHub LICENSE](https://github.com/ing-bank/sparse_dot_topn/blob/master/LICENSE) |
| polars | `polars` | MIT | Rust ext | CPU-only, multi-core, RAM-efficient (fits the ~5GB free budget far better than pandas) | **OK** | [PyPI](https://pypi.org/project/polars/) |
| duckdb | `duckdb` | MIT | embedded engine | CPU-only, out-of-core (good for 5M+ row joins under RAM pressure) | **OK** | [duckdb.org/faq](https://duckdb.org/faq) |
| splink | `splink` | MIT | pure Python + SQL backend | CPU-only (DuckDB backend) | **OK** — good fit for the Fellegi-Sunter probabilistic blocking/scoring sub-problem | [PyPI license expression](https://pypi.org/project/splink/) |
| dedupe | `dedupe` | MIT | pure Python + C ext | CPU-only | **OK** | [PyPI](https://pypi.org/project/dedupe/) |
| recordlinkage | `recordlinkage` | BSD-3-Clause | pure Python | CPU-only | **OK** | [Libraries.io](https://libraries.io/pypi/recordlinkage) |
| libpostal / postal | `postal` (bindings to C lib `libpostal`) | MIT (both lib and bindings) | C lib + ~2GB data model | CPU-only; **~2GB on-disk data + real RAM to load** — tight against the ~5GB free budget, test early | **OK, with an argument** (below) | [libpostal GitHub](https://github.com/openvenues/libpostal), [pypostal PyPI](https://libraries.io/pypi/postal) |
| sentence-transformers | `sentence-transformers` | Apache-2.0 | framework, N/A | CPU or GPU | **OK** | [GitHub LICENSE](https://github.com/UKPLab/sentence-transformers/blob/main/LICENSE) |
| pyahocorasick | `pyahocorasick` | BSD-3-Clause | C ext, tiny | CPU-only | **OK** | [PyPI](https://pypi.org/project/pyahocorasick/) |

**libpostal/OSM argument:** libpostal (MIT) is a static, pre-trained CRF address parser that ships its own weights and runs **fully offline** — no network call, API, or registry lookup happens at inference. Its *training* data included OpenStreetMap/OpenAddresses, but that's true of essentially every pretrained NLP model in this audit (XLM-R and MuRIL train on CommonCrawl; embedding models train on scraped web text) — training-time data provenance isn't what "no external APIs/services/geocoding/registries/internet data **for resolving entities**" is banning; that rule targets *live* lookups during inference (e.g., calling a geocoder or a business registry to resolve a specific test-set entity). libpostal never does that — it only normalizes address strings using fixed, shipped weights. Verdict: **OK to use**, same footing as any other pretrained model in this list. The real risk is operational, not legal: the ~2GB data model against a ~5GB free-RAM laptop — load it once, don't reload per-process.

## Models

| Model | HF repo | SPDX licence | Params | CPU/GPU feasibility | Verdict | Source |
|---|---|---|---|---|---|---|
| LaBSE | `sentence-transformers/LaBSE` | Apache-2.0 | 471M | Fits RTX 3050 6GB easily; CPU usable but slower | **OK** | [HF](https://huggingface.co/sentence-transformers/LaBSE) |
| BGE-M3 | `BAAI/bge-m3` | MIT | 568M | Fits 6GB GPU; good multilingual dense+sparse retrieval | **OK** | [HF](https://huggingface.co/BAAI/bge-m3) |
| multilingual-e5-small | `intfloat/multilingual-e5-small` | MIT | 118M | Trivial CPU/GPU | **OK** | [HF](https://huggingface.co/intfloat/multilingual-e5-small) |
| multilingual-e5-base | `intfloat/multilingual-e5-base` | MIT | 278M | Easy CPU/GPU | **OK** | [HF](https://huggingface.co/intfloat/multilingual-e5-base) |
| gte-multilingual-base | `Alibaba-NLP/gte-multilingual-base` | Apache-2.0 | 305M | Easy CPU/GPU, **but requires `trust_remote_code=True`** (custom modeling code) — operational/security caveat, not a licence one, and worth avoiding on a shared SageMaker container | **OK (licence)** / flag on `trust_remote_code` | [HF](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) |
| paraphrase-multilingual-MiniLM-L12-v2 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 | 118M | Trivial | **OK** | [HF](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) |
| all-MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | 22M | Trivial | **OK licence, poor fit** — English-only training data, won't handle Devanagari/Telugu cross-script pairs | [HF discussion](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/discussions/34) |
| Qwen3-Embedding-0.6B | `Qwen/Qwen3-Embedding-0.6B` | Apache-2.0 | 0.6B | Fits 6GB GPU fp16 | **OK** | [HF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | Apache-2.0 (confirmed via LICENSE file, standard unmodified Apache 2.0 text) | 1.5B | Fits 6GB GPU fp16 comfortably | **OK** | [LICENSE file](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE) |
| Qwen3-4B | `Qwen/Qwen3-4B` | Apache-2.0 (confirmed via LICENSE file) | 4B | fp16 ≈8GB — **too big for RTX 3050 6GB unquantized**; needs 4-bit/8-bit quant on laptop, or run on SageMaker T4(16GB)/A10G(24GB) | **OK (licence, ≤8B)**, feasibility needs quantization on laptop | [LICENSE file](https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE) |
| deberta-v3-small | `microsoft/deberta-v3-small` | MIT | 44M backbone / 142M incl. embeddings | Trivial | **OK licence, English-only** — not directly useful for raw Indic script without transliteration first | [HF](https://huggingface.co/microsoft/deberta-v3-small) |
| deberta-v3-base | `microsoft/deberta-v3-base` | MIT | 86M backbone / 184M incl. embeddings | Trivial | **OK licence, English-only** — same caveat | [HF](https://huggingface.co/microsoft/deberta-v3-base) |
| xlm-roberta-base | `FacebookAI/xlm-roberta-base` | MIT | 270M | Easy CPU/GPU | **OK** | [HF discussion](https://huggingface.co/FacebookAI/xlm-roberta-base/discussions/20) |
| MuRIL | `google/muril-base-cased` | Apache-2.0 | ~236M | Easy CPU/GPU | **OK** — purpose-built for 17 Indian languages incl. transliterated pairs, strong fit for the India cross-script sub-problem | [HF](https://huggingface.co/google/muril-base-cased) |
| IndicBERTv2 | `ai4bharat/IndicBERTv2-MLM-only` (family) | MIT | 278M | Easy CPU/GPU | **OK** | [GitHub](https://github.com/AI4Bharat/IndicBERT) |
| MS-MARCO MiniLM cross-encoder | `cross-encoder/ms-marco-MiniLM-L6-v2` (note: HF repo is `L6-v2`, not `L-6-v2` as spelled in the brief) | Apache-2.0 | 22M | Trivial | **OK licence, English-only training** — usable as an architecture/init to fine-tune, but for the multilingual pair-scoring sub-problem, fine-tuning a cross-encoder head on `xlm-roberta-base` or `muril-base-cased` is the better-fitting base | [HF](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) |

**Qwen licence trap (confirmed):** Qwen2.5's 3B and 72B variants ship under a separate, non-Apache "Qwen/Tongyi" licence — only the sizes actually named in this brief (1.5B, and all Qwen3 dense sizes incl. 4B) were verified Apache-2.0 directly from each repo's `LICENSE` file. Don't swap in a different Qwen2.5 size without re-checking.

## 5-line shortlist (safest per role)

1. **String similarity:** `rapidfuzz` (MIT) — fast, no licence risk, covers Levenshtein/Jaro-Winkler/token-set for the fuzzy-name-match sub-problem.
2. **Transliteration (18% cross-script India pairs):** `ai4bharat-transliteration`/IndicXlit (MIT) for neural Devanagari/Telugu↔Latin, `indic-transliteration` (MIT) as a rule-based fallback — skip `unidecode` (GPL) and `aksharamukha` (AGPL) entirely.
3. **ANN index (5–13M-row blocking):** `faiss-cpu` (MIT), `hnswlib` (Apache-2.0) as the lighter alternative — both CPU-only, fit the 16-core/no-GPU-needed constraint.
4. **Multilingual embedder:** `intfloat/multilingual-e5-base` (MIT, 278M) as the default; `google/muril-base-cased` (Apache-2.0, ~236M) if fine-tuning specifically for the Indic-transliteration sub-problem.
5. **Cross-encoder/fine-tune base:** `xlm-roberta-base` or `google/muril-base-cased` (both MIT/Apache-2.0, ~250-270M, genuinely multilingual) as the backbone to fine-tune a pairwise scorer on S1↔S2/S3 — don't start from the English-only `ms-marco-MiniLM-L6-v2` for this multilingual task.