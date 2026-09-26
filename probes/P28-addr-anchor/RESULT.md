# P28-addr-anchor — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
The rule as briefed (one core word **swapped** + same address ⇒ accept) loses everywhere there are labels.
- On #8c's own holdout, the pairs it flips are 0.2–0.7% true (bx: US −0.0037, India −0.0063).
- In the unseen-country proxy (LOCO), swaps flip at 61–65% true, below the 0.79 break-even.

What works is the same-address idea for names that differ by an **insert / delete, or not at all**, with a label-free sibling-word guard (`bxneg`).
- LOCO India→US: **+0.0049 ± 0.00016**; US→India: **+0.0012 ± 0.00013** (flips 99.9% / 90% true).
- In-domain on #8c's US holdout it flips only 21 pairs (−0.00002): #8c already accepts every true same-address pair. `sub28a` applies the rule to France only; US/India rows are untouched.

On France (#9ab) it adds 15,546 pairs to 14,417 S1, drops nothing, and stays inside P6's bands.
**Upload `data/work/p28/sub28a` (validator PASS) against #9ab.** It is one lever: US/India rows are identical to #9ab's.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
1. **LB slot: `data/work/p28/sub28a/`**, both TSVs. Official validator: PASS.
   - It is #9ab plus the rule on France only. vs `output/sub9ab`: 14,417 France rows differ and every one is a superset of the old row. 0 US/India rows differ. Pairs 5,762,680 → 5,778,226 (+15,546).
   - Rebuild it (≈6 min, ≤2.3 GB). Re-running the final `fr.py` reproduced the staged France file exactly (16:39 check):
     ```
     .venv/bin/python probes/P28-addr-anchor/fr.py --blocked bxneg        # -> data/work/p28/prob_test_France_9ab_bxneg_blk.parquet
     .venv/bin/python probes/P14-france-shift/combine.py /home/harsh1/github-repos/amc-26/output/sub8c \
         data/work/p28/prob_test_France_9ab_bxneg_blk.parquet 0.985 data/work/p28/sub28a
     ```
   - France profile: 6.71% → 6.48% empty, 3.018 → 3.078 matches per S1 (P6 bands: 4.5–7.5%, ≥ 2.9).
   - Expected LB effect if France behaves like LOCO India→US: France +0.005–0.007, so LB +0.0008–0.001. If it behaves like US→India: LB +0.0002.
2. **The rule (`bxneg`)**, in `probes/P28-addr-anchor/anchor.py`. It is label-free and general code; nothing is keyed on a country name. A pair qualifies when all of these hold:
   - **name:** the cores (normalize.name_core) are equal, or one token is inserted / deleted (P20's `name_rel` kinds eq / add1 / drop1). Swaps (sub1) and typo-only pairs are excluded.
   - **house number:** present on both sides and equal.
   - **city:** the two records share a place segment (rules.py's `po`).
   - **street:** the street key of the hn segment is exactly equal after `N._seg` canonicalisation (articles / bis / ter dropped).
   - **sibling-word guard:** the insert / swap word is not a learned sibling word. `sib_words`: 'kind:word' whose hn-equal-or-SIB3-offset pairs take a generator sibling offset ≥ 50% of the time, with ≥ 20 pairs. Learn it on the **blocked** candidate list (P3), not the pruned one.
   - **eligibility:** the pair is its SX's argmax, or its SX is matched nowhere (best p < t). So the rule never steals an SX from an accepted S1.
   - **action:** `p := max(p, 0.99)` before the usual argmax + t decision.
   - API: `m = rule_masks(d[["s1","sx","p"]], "test", c, t, learn=blocked_pairs)`, then `p = boosted(p, m["bxneg"])`.
3. **Where it goes in src (if #28a wins):** as the last stage of `run.model_parts` for **labels without training data** (France), after `after_ce`: `thresholds` gives t, `rule_masks` is called on the country's scored pairs and `p3_cands` blocked pairs, and the result feeds `boosted`. Do **not** turn it on for US/India: in-domain it can only find the model's few residual errors (bxneg −0.00002 US; unguarded variants down to −0.027 India).
4. **It stacks with R1 (P20's sub20a).** 5,447 of the 6,125 added insert pairs are R1's filler words (fils 2,735 / associés 1,428 / cie 1,284). The core-equal (7,792) and delete (1,629) pairs are new.
   - If sub20a wins on the LB, apply the rule to its France file instead: `fr.py --blocked --src data/work/p20/prob_test_France_idf_noisesib.parquet --t 0.975 bxneg`, then combine.py at 0.975.
   - After R1 strips the fillers, those pairs become core-equal, which the rule also covers.
5. **Conservative fallback: `cxneg`** = bxneg + chain guard (skip when another S1 has the same city + hn + street).
   - France: +12,073 pairs, 11,205 S1, 6.52% empty, 3.065 per S1.
   - LOCO: lower gain than bxneg in both directions (+0.0042 / +0.0009) but higher flip precision in the India regime (0.94 vs 0.90).
   - Build: `fr.py --blocked cxneg`, then combine.py.

## Evidence         (table: ledger name | world | key metrics | verdict)
Variants: a = 1 word (swap / insert / delete) + hn + city; b = a + street fuzzy (ratio ≥ 85); bx = a + street exact; c = b + chain guard (the brief's a / b / c); bxs = bx swaps only; bxn = bx insert / delete only; bxe = core equal + same address; bxt = typo only; bxne = bxn ∪ bxe; g = + sibling-word guard; cx… = + chain guard (city + hn + street).
ΔF0.5 is paired per S1, half 1 ± SE. "Flip" = a pair the rule turns from rejected to accepted; it is the only thing the rule changes. Flip precision is the share of flipped pairs that are true.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| addr-anchor-holdout-US (last row 16:30) | #8c dense holdout (P23 dump, t 0.71), base 0.98924 | a −0.00526 ± 0.00020 (2,627 flips, 0.6% true); bx −0.00374 (1,781, 0.7%); c −0.00253; bxs −0.00365; bxn −0.00010 (38 flips); bxe −0.00001; **bxneg −0.00002 ± 0.00001 (21 flips; its 212,137 rule pairs are 100.0% true, and #8c already accepts all but 21)** | drop (nothing to gain in-domain) |
| addr-anchor-holdout-India (16:31) | same, base 0.99004 | a −0.02707 (9,248 flips, 0.1% true); bx −0.00628; c −0.00084; bxn −0.00384 (rule pairs only 87.0% true); bxe −0.00114; bxneg −0.00477 (the guard learns 1 word from the **pruned** list) | drop |
| addr-anchor-loco-India>US (16:36, loco6) | P6 LOCO: India-only M2 on US fold-0 (France-like, US-like hn); t_profile 0.81 / oracle 0.87; base 0.9285 / 0.9296 | a +0.00114 / +0.00241 (flips 74% true); bx +0.00186 / +0.00295 (80%); c +0.00111 / +0.00181; **bxs −0.00111 / −0.00089 (61% / 63%)**; bxt −0.0005; bxn +0.00299 / +0.00387 (99%); bxe +0.00204 / +0.00294 (99.9%); bxne +0.00500 / +0.00676; **bxneg +0.00485 ± 0.00016 / +0.00661 ± 0.00019 (5,638 / 7,584 flips, 99.9% true)**; cxneg +0.00415 / +0.00564; bxsg −0.00094 | keep (bxneg) |
| addr-anchor-loco-US>India (16:36, loco6) | US-only M2 on India fold-0 (India hn regime); t 0.97 / 0.98; base 0.9021 / 0.9054 | a −0.02460 / −0.02648; bx −0.00391; bxn −0.00358 (add1 flips 18% true: India siblings keep the hn); bxe +0.00107 / +0.00162 (94%); bxne −0.00250 / −0.00204; **bxneg +0.00122 ± 0.00013 / +0.00192 ± 0.00017 (1,462 / 2,045 flips, 90% true)**; cxneg +0.00087 / +0.00133 (94%); bxsg +0.0003 | keep (bxneg); the guard turns −0.0025 into +0.0012 |
| addr-anchor-France-on-9ab (16:29) | test France, #9ab file (P19 idf144 + bge, t 0.985; reproduces uploaded #9ab France rows exactly) | base 6.71% empty, 3.018 per S1. Guard learned on the file (31 words). bxneg +16,423 pairs, 15,192 S1. bxne (no guard) +20,344. bxn +12,554. bxe +7,799. cxne +15,814. bxsg +50,154 / bx +62,707 (swap-dominated, not recommended). All in P6 bands | inconclusive (no labels) |
| addr-anchor-France-on-9ab_blk (16:33) | same; guard learned on P3's blocked France list (4.3M pairs, 1,197 words, 65 of them insert words) | **bxneg +15,546 (eq 7,792 / add1 6,125 / drop1 1,629), 14,417 S1, 0 dropped, 6.48% empty, 3.078 per S1**; cxneg +12,073, 11,205 S1; bxng +7,756 | keep → sub28a |
| addr-anchor-sub28a-build | test | validator PASS; 14,417 France rows differ from sub9ab (all supersets); 0 non-France rows differ | **keep → LB** |

Label-free check that France sits in the LOCO regime, not the in-domain one. The measure is the share of the rule's own pairs that the base model rejects:
- in-domain US #8c: 0.01%;
- LOCO US→India: 1.5%;
- LOCO India→US: 2.7%;
- **France #9ab: 3.0%** (15.7k of 523.7k).

On insert / delete pairs alone, France rejects 27.5% vs 9.8% in LOCO India→US. France's model is at least as shifted as the proxy on exactly these pairs. The rule's precision over all its pairs is a property of the data (US 100.0% / India 98.2% for bxne), not of the model.

Learned sibling words: US 188 / India 139 (LOCO lists), France 31 (file) / 1,197 (blocked list).
- They recover P20's hand analysis without labels: groupe, holding, distribution, participations, international, developpement, and the org words ecole / comite / college / union.
- On the US side they include branch words (holdings, group, north / south / downtown).

## What failed      (approaches tried and dropped, with the number that killed them)
- **The user's swap rule (sub1).**
  - In-domain, #8c already accepts the true swap pairs, and the 1,743 it rejects are 0.6% true (−0.0037).
  - In LOCO India→US the flipped swaps are 61–65% true (−0.0011 / −0.0009). The sibling guard does not rescue it: bxsg is −0.0009 US-like, +0.0003 India-like.
  - Reason: a swapped org word at the same address is often a *different* entity sharing the address (a club at a school, a branch in a mall). Under a shifted model these sit at 60%, below the 0.79 break-even.
  - On France, swaps would add 50k pairs (5.6% empty) on this evidence. Do not.
- **Brief variants (a) hn + city and (b) + fuzzy street:** worse than exact street in every world. Fuzzy street ≥ 85 admits neighbouring-number streets and typo'd distinct streets. LOCO India→US: a +0.0011, b +0.0015, bx +0.0019.
- **Typo-only names (bxt):** flips 53–58% true in LOCO (−0.0005). A near-identical but different word is a different business.
- **Chain guard keyed on (city_key, hn):** flagged 95% of France pairs. France S1 cluster in a few cities with small house numbers. It is keyed on (city, hn, street) now, and even then it costs LOCO gain (cxneg < bxneg in both directions).
- **Guard learned on a pruned list:** #8c's pruner removes most sibling-offset pairs, so the India in-domain list yields 1 sibling word and bxneg = bxne (−0.0048). Learn it on the blocked list (P3); the LOCO frames and `fr.py --blocked` do.
- **A US / India variant (brief step 3): not built.** In-domain bxneg is −0.00002 (US) and −0.0048 (India, guard starved). #8c already accepts every true same-address pair.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `anchor.py` (self-check: `.venv/bin/python probes/P28-addr-anchor/anchor.py`):
  - `street_key(addr: Series) -> Series`: the street of the house-number segment, canonical and sorted; '' if there is no hn.
  - `sib_words(key, sib, support=20, share=0.5) -> set` and `sib_table(pairs(s1, sx), xs, c) -> set`: the guard.
  - `rule_masks(d(s1, sx, p), split, c, t, am=None, best=None, learn=None) -> {variant: bool mask, 'kind', 'info', 'sibw'}`. Pass `am` / `best` when `d` holds only part of each SX's candidates.
  - `boosted(p, mask) -> p` (BOOST 0.99; it must stay above t) and `decide(sx, p, t) -> mask` (argmax per SX & p ≥ t).
- `ho.py [US India]`: #8c dense holdout from `data/work/p23/pairs_{c}.parquet`. Asserts it reproduces #8c's decision, then prints per-variant ΔF / flips. 3 min, 2.4 GB per country.
- `loco.py ["India>US" "US>India"]`: P6 LOCO scores (`data/work/p27/p6/S_holdout_{src}.parquet`, codes decoded via `F.ids`). 4 min, 2.6 GB per direction.
- `fr.py [--blocked] [--src FILE --t T] variants…`: applies the rule to a France (s1, sx, p) file. Writes `data/work/p28/prob_test_France_9ab_<v>[_blk].parquet` (or `<src stem>_<v>…`) and prints the profile. 3–6 min, ≤2.3 GB.
- Logs: `data/work/p28/{ho5,loco6,fr2,fr3}.log` (aggregates only).
- Staged: `data/work/p28/sub28a/` (217 MB; delete once uploaded).
- Other rule files in `data/work/p28/`: bxn, cxn, bxe, bxne, cxne, bxneg, bxnesg, bxsg, cxneg (file guard); bxneg_blk, cxneg_blk, bxng_blk.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **France precision is not measured; there are no labels.** The case rests on three things:
  - LOCO in the US-like hn regime (flips 99.9% true) and in the India regime with the guard (90%).
  - France rejecting the rule's pairs at the LOCO rate, not the in-domain one.
  - P20's proxy that France siblings keep the hn only 2–6% (US-like).
  - The LB is the test. A drop would mean France has more same-address different entities than US/India.
- **France-specific exposure: delete pairs.** The dropped word is an org word (club, amicale, sportive, ecole, comite…), 1,629 pairs.
  - An association and the school it belongs to can share an address.
  - The argmax condition bounds this: the SX goes to the S1 that already scores highest.
  - `cxneg` is the hedge.
- **Label-free and no external data.**
  - The guard is learned from each country's own candidate pairs; unsupervised use of test data is allowed by organiser ruling 2.
  - SIB3 offsets come from train (P12).
  - Street types / articles come from normalize (existing hand lists), plus 'bis / ter / quater'.
  - No model and no new dependency (rapidfuzz is already used).
- **Thresholds:** BOOST 0.99 must exceed France's t (0.985 now). If t goes above 0.99, raise BOOST.
- **LOCO proxy limits:** the P6 LOCO frames are M2-only (no CE, no pruner) on blocked pairs; #9ab's France has a bge CE and a pruner. The regime match rests on the rejection-rate check above.
- **Ledger:** P28 rows went to the main ledger (`/home/harsh1/github-repos/amc-26/ledger/experiments.jsonl`).
  - Each script was re-run as variants were added, so earlier rows hold subsets.
  - The last row per name is final: holdout 16:30–16:31, loco 16:36, France 16:29 (file guard) + 16:39 (blocked guard, the rebuild check), build 16:37.
  - The `keep` verdict on loco-US>India comes from its best variant (bxneg / bxe), not from the brief's variants, which are negative.

## Open questions   (for the master; each with your best guess)
- **Promote the rule into src for unseen labels?**
  - Best guess: yes, if #28a beats #9ab on the public LB.
  - It is ~120 lines of general code and adds a stage only for labels without training data.
  - Put it after `after_ce` in `model_parts` and learn the guard on `p3_cands`.
- **Stack on R1 (sub20a) or on #9ab?**
  - Best guess: on whichever France file is best on the LB.
  - The overlap is the ~5.4k filler-insert pairs, and the rest is additive.
  - `fr.py --src … --t …` does it in one command.
- **Lower France's t instead?**
  - No. The added pairs have a median p of 0.958 (10% below 0.63), so a lower t would also admit the swap and typo pairs the LOCO shows to be 55–65% true.
  - The rule is the targeted version of "lower t where the address agrees".
- **Is the core-equal part (7.8k pairs) the safest piece?**
  - Probably yes: LOCO flips are 99.9% true (US-like) and 94% (India-like), and it needs no guard.
  - If the master wants half the risk, `fr.py --blocked bxe` + combine gives +7,799 pairs alone.
