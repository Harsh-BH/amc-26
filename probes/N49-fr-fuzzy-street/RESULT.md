# N49-fr-fuzzy-street — RESULT

## TL;DR
The brief hypothesized that the base matcher rejects France argmax pairs where the only difference is the street name, caused by normalization bugs (elision/apostrophes, abbreviations, accents/case) or generator typos.

1. **Cell Size and Address Noise**:
   - The cell size per S1 in France is **2.34**, which is significantly larger than US (**1.48**) and India (**1.27**).
   - This inflation is heavily driven by **fuzzy-street noise (generator typos)**. The fuzzy-street cell per S1 in France is **0.33**, compared to just **0.10** in the US and **0.04** in India. 

2. **Validation on the Revised Bar (In-Domain Negatives per S1)**:
   - **Exact street key additions** (P28-like) would add 10,501 pairs to France. They **PASS** the validation bar:
     - US precision bound: 99.6% ($\ge$ 0.90)
     - India precision bound: 87.6% ($\ge$ 0.75)
   - **Fuzzy street key additions** would add 5,011 pairs to France. They **FAIL** the validation bar:
     - US precision bound: 92.5% ($\ge$ 0.90)
     - India precision bound: **72.4%** ($<$ 0.75)

3. **Deduplication and Label-Free Check**:
   - P28 (`bxneg`) already targets exact street key matches, subsuming the exact arm.
   - For the fuzzy arm, after deduplicating against P42 (hn-extra) and P37 (gc), there are only **2,443 truly NEW pairs**.
   - A label-free check on these 2,443 new pairs shows a massive skew in `name_rel`: **30%** of the new pairs are `add1` (mostly P20 R1 filler words like *associés*), compared to just **0.6%** in the cell's accepted pairs. The new pairs do not look like a true-variant cell.

## Evidence

### Cell Size (Task 1)
| Country | S1 Count | Exact pairs per S1 | Fuzzy pairs per S1 | Total Cell per S1 |
|---------|----------|--------------------|--------------------|-------------------|
| US      | 129,822  | 1.316              | 0.097              | 1.478             |
| India   | 86,635   | 0.991              | 0.036              | 1.274             |
| France  | 256,912  | 1.948              | 0.327              | 2.339             |

France's cell is inflated by a much higher rate of fuzzy street differences (generator typos), confirming it has larger address noise.

### Revised Validation Bar (Task 1)
*Negatives per S1* bounding on pairs the rule would add (i.e. rejected & free argmax in base).
- **Exact Additions** (France pool: 10,501 pairs)
  - US rate: 21 falses $\rightarrow$ 0.00016 neg/S1 $\rightarrow$ 42 exp. France negatives $\rightarrow$ **99.6% precision**.
  - India rate: 434 falses $\rightarrow$ 0.0050 neg/S1 $\rightarrow$ 1,300 exp. France negatives $\rightarrow$ **87.6% precision**.
- **Fuzzy Additions** (France pool: 5,011 pairs)
  - US rate: 187 falses $\rightarrow$ 0.0014 neg/S1 $\rightarrow$ 374 exp. France negatives $\rightarrow$ **92.5% precision**.
  - India rate: 462 falses $\rightarrow$ 0.0053 neg/S1 $\rightarrow$ 1,384 exp. France negatives $\rightarrow$ **72.4% precision**.

The fuzzy additions fail the India bar of 75%.

### Deduplication (Task 2)
Filtering the 5,011 fuzzy additions against P42 and P37 leaves **2,443 truly NEW pairs**. 
- Fills empty S1: 236 pairs.
- By p-band:
  - $p \ge 0.9$: 23 pairs (Exp. negatives 64 based on US rate $\rightarrow$ bound $<$ 0%)
  - $p \ge 0.8$: 31 pairs (Exp. negatives 2.5 based on US rate $\rightarrow$ bound 92%)
  - $p \ge 0.7$: 46 pairs (Exp. negatives 0 $\rightarrow$ bound 100% — small sample)
  - $p \ge 0.6$: 626 pairs (Exp. negatives 0 $\rightarrow$ bound 100% — small sample)
  - $p \ge 0.5$: 430 pairs (Exp. negatives 0 $\rightarrow$ bound 100% — small sample)
  - $p \ge 0.4$: 327 pairs (Exp. negatives 2.5 based on US rate $\rightarrow$ bound 99%)
  - $p < 0.4$: 204 pairs (Exp. negatives 287 based on US rate $\rightarrow$ bound $<$ 0%)
*Note: Due to the very small number of additions in the upper bands, the in-domain false counts (e.g. 32 falses in US at $p\ge 0.9$ projecting to 64 France negatives) dwarf the additions, making the bounds fail or become highly unstable.*

### Label-Free Check (Task 3)
Comparing the 2,443 newly added fuzzy pairs against a sample of the cell's accepted pairs:
- **New pairs `name_rel`**: eq (68.5%), **add1 (29.9%)**, sub1 (0.9%), drop1 (0.7%)
- **Accepted pairs `name_rel`**: eq (82.8%), web (7.6%), sub1 (3.4%), drop1 (2.6%), **add1 (0.6%)**
The new pairs are heavily skewed towards `add1` (likely France R1 stripped filler words + fuzzy streets). This deviates massively from the true-like accepted distribution, indicating they are likely distinct entities.

## Recommendation
**Do not stage the fuzzy street rule.** It decisively fails the revised validation bar on the India rate (72.4% vs 75% required) and its label-free profile exhibits a massive `add1` skew (30%) inconsistent with accepted pairs. The exact-street additions pass the bar, but they are functionally equivalent to the already-validated P28 (same-address) rule and should be handled there.
