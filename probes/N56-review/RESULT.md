# N56-review — RESULT (Code & Evidence Review)

## TL;DR
- **`p54.build` / `n55.build`**: No bugs found. Handled missing newlines, duplicate SXs, empty rows correctly. Adversarial tests added.
- **`n55.analyse`**: **BUG**. The US/India holdout lens is incorrectly computed on holdout pairs only (`d[d.holdout & d.am]`). This artificially suppresses the noise words list because `n >= 20` is much harder to reach on the 10-20% holdout split. (Testing confirms US lens drops from 32 words to 9 words).
- **`loco_n54_n55.py` / `loco_chain.py`**: No bugs found. F0.5 per-S1 computation and ID mapping match P28/P46. No truth leakage.
- **N54 'all' logic**: **FLAWED**. Removing pairs at `p >= 0.99` deletes high-confidence true positives that legitimately swapped an org-word (e.g. `ecole` -> `club`) at the same address.
- **src port (`decoy_mask` & `run.py`)**: No bugs found. Variables align perfectly at the hook, default recipe behavior is preserved.

---

## 1. `p54.build` / `n55.build`
**Severity**: None
**Issues**: "no bugs found". 
- `p54.build` safely reads and modifies `matching_results.tsv` lines. Missing newlines on the last row are repaired, empty rows are cleanly skipped, and missing candidates are ignored.
- `n55.build` correctly enforces `CAP`, checks `cand` globally for candidates, and checks `owned` for already-taken SXs.
- **Deliverable**: Adversarial selftests testing these exact edge cases have been written to `probes/N56-review/test_builds.py` and pass.

## 2. `p54.cells` / `n55.analyse`
**Severity**: High
**Issues**: 
- **BUG (`n55.py:59-60`)**: In `n55.analyse`, the holdout analogue for US and India computes the lens over `d = d[d.holdout & d.am]`. By restricting `lens_table` to holdout pairs only, the `n_hn >= 20` support threshold filters out a large number of legitimate true noise words (as they only appear a fraction of times in the holdout vs the full train set). This severely underestimates the in-domain analogue cell sizes, skewing the bounds and making France look 8x larger than it really is. 
  - **Fix**: Compute the lens on all argmax pairs, like in `p46.py`: 
    ```python
    a = d_all[d_all.am].reset_index(drop=True)
    lt = p46.lens_table(a[["s1", "sx"]], "train", c)
    ```
- **Alignment / Truth leakage**: "no bugs found". `p54.cells` and `n55.analyse` correctly align `s1` / `sx` via positional assignment when modifying subset arrays. The lens in France is correctly label-free (`argmax`). `p` is not used in rules incorrectly.

## 3. `loco_n54_n55.py` / `loco_chain.py`
**Severity**: None
**Issues**: "no bugs found". 
- ID mapping uses `OFF1`/`OFF2` and `CI` exactly as in `P28`. 
- Per-S1 F0.5 `fbeta` is imported and used exactly identically to `P28`/`P46`. 
- No truth (`y`) is leaked into the rule masks (`hnx`, `lt`, etc.).

## 4. N54 'all' removing `p >= 0.99` true France pairs
**Severity**: Medium
**Issues**: 
- **FLAWED REASONING**: The lens words list (`decoy_words.csv`) includes org words like `club`, `ecole`, `amicale`. A true France business might legitimately change its org-word (e.g. from an Ecole to a Club) while staying at the exact same address. The matcher gives these true variants `p >= 0.99` when the remaining core name is highly distinctive. By using the `all` variant, N54 indiscriminately removes these high-confidence true matches. N54's own mixture test confirms this: the `p >= 0.9868` band contains ~3% true pairs (~22 pairs out of 724). Deleting them reduces recall unnecessarily. 
  - **Fix**: Stick to `ruled` to only drop pairs added by post-threshold rules, or impose a ceiling `p` cap for `all` (e.g., `p < 0.99`) to preserve matcher-accepted true variants.

## 5. src port (`decoy_mask` & `run.py`)
**Severity**: None
**Issues**: "no bugs found". 
- The `decoy_mask` implementation in `france.py` uses `np.ndarray` inputs, which perfectly match the arrays sliced by `ci` in `run.py`.
- The hook `acc = acc & ~FR.decoy_mask(...)` executes sequentially after both the `CHAIN` and `complete()` paths, cleanly filtering both. 
- The condition `DECOY is not None` safely preserves default behavior when N54 is not explicitly enabled in the recipe. 
- `FR.C` and `qx` are correctly set prior to evaluation.

## MASTER VERDICT (08:30)
- Items 1, 3, 5: accepted (no bugs; adversarial selftests pass).
- Item 2: not a correctness bug — the in-domain analogue lens learned on holdout argmax pairs has less support (US 9 /
  India 5 noise words); the cell precision (0.998 / 0.997) is valid for the words it covers. N55's France evidence is
  the France lens + the slot test, as stated in N55 RESULT.
- Item 4: already priced in — N54 'all' LOCO removes pairs that are 0.9% (US→India) / 7% (India→US) true and is still
  positive in both directions and halves (addenda 3–5, loco_gc70.py). Keep 'all' as primary, 'ruled' as fallback.
