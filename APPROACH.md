# Our approach — Amazon ML Challenge 2026 (explained simply)

*Team **Yeti** — Harsh Bhatt, Shriya Sinha, Aditya Kashyap, Anurag Pokhariyal (IIT Kharagpur). Last updated: 26 Sep 2026, morning. Deadline: 27 Sep 2026, 23:59 IST.*

---

## 1. The challenge in one minute

Amazon gets information about the same businesses from **3 different sources**. The sources don't share an ID, and
each one writes things differently: short forms, typos, missing parts, even different scripts (Hindi, Telugu…).

- **Source 1** is the clean master list: every business appears once.
- **Sources 2 and 3** are messy copies.

**Our job:** for every business in Source 1, find all the rows in Sources 2 and 3 that are the *same real business*.

Example:

| Source | Name | Address | |
|---|---|---|---|
| Source 1 | Ram Marketing Private Limited | 12 MG Road, Pune | the master record |
| Source 2 | Ram Mktg Pvt Ltd | MG Rd, Pune | ← same business |
| Source 2 | राम मार्केटिंग प्राइवेट लिमिटेड | Near SBI ATM, Pune | ← same business, written in Hindi |
| Source 3 | Ram Traders | 5 FC Road, Pune | ← different business (a look-alike) |

The answer for that Source 1 business is the first two rows. The third looks similar but is a trap.

**Size:** the test set has **1.73 million** Source 1 businesses and about **10 million** rows in Sources 2 and 3.
The training data (with the correct answers) covers the **US and India**. The test set **also contains France**, which
has no training answers at all.

## 2. How we are scored

For each Source 1 business, the organisers compare our list with the true list:

- **Precision:** of the matches we gave, how many were right?
- **Recall:** of the true matches, how many did we find?

They combine these into a score called **F0.5**, which counts **precision twice as much as recall**: a wrong match hurts
more than a missed one. A business that truly has no match scores 1.0 if we leave it empty and 0 if we match anything.
The final number is the average over all businesses. 1.0 is perfect.

A new rule (25 Sep): the **candidate list** we produce (the shortlist we consider for each business) also counts.
**Smaller shortlists rank higher**, on top of the leaderboard score.

---

## 3. How we organised the work (and why)

We had 72 hours, one laptop, and a big dataset, so we split the problem into **small sub-problems** and attacked them
**in parallel**:

- One **"master"** session (the coordinator) plans the work, checks every result and decides what to keep.
- Several **"probes"** each solve one sub-problem and write a short report plus their numbers into a shared **results
  ledger**. A probe is a separate AI session (Claude Opus 5.5) running in its own terminal window.
- Each probe works in its own copy of the code (a **git worktree**), so they can't break each other. The master merges
  their work carefully and runs the tests after every merge.

The sub-problems:

| When | Probe | Question it answered |
|---|---|---|
| 25 Sep | P1 data analysis | What kinds of mess are in the data, and where do we lose points? |
| 25 Sep | P2 scripts | How do we compare a Hindi/Telugu name with an English one? |
| 25 Sep | P3 candidates | How do we quickly find a short list of possible matches for each business? |
| 25 Sep | P7 scale | Can the full run fit on a 15 GB laptop? |
| 25 Sep | P4 matcher | Which model best tells a true match from a look-alike? |
| 25 Sep | P6 France | How do we stay accurate on France with no answers to learn from? |
| 25 Sep | P5 decisions | How do we turn match scores into final answer lists? |
| 25 Sep | P4b reranker | Can a small language model fix the hardest cases? |
| 25–26 Sep | P8 / P9 pipeline | One command that goes from raw data to the final files |
| 25 Sep | P10 candidates | Make the shortlists smaller (the new rule) |
| 25 Sep | P11 bigger reranker | Does a bigger or longer-trained language model help? |
| 25 Sep | P12 fake France data | Can we invent training examples that look like France? |
| 25–26 Sep | P13 reranker on AWS | Does a stronger reranker (bge), trained on a cloud GPU, help? |
| 26 Sep | P14 France shift | *Why* is France worse, and can we fix it without answers? |
| 26 Sep | P15 research | What did winners of similar contests do? What have we missed? |
| 26 Sep | P16 matcher on AWS | Does training the main model on twice the data help? |
| 26 Sep | P17 house numbers | Can we catch the data's fake "sibling" businesses by their house number? |
| 26 Sep | P18 documents | Bring this document, the methodology and the package up to date |

Why this plan: every stage of the problem has its own way of failing. So each probe could be measured on its own
number, and the master only kept what the numbers supported.

---

## 4. What the data taught us (the important discoveries)

1. **Every messy row belongs to at most one master business.** A row in Source 2 can't match two Source 1 businesses.
   We use this everywhere: each messy row goes to its single best match.
2. **Half the businesses have look-alikes.** About 50% of Source 1 businesses share their cleaned name with another
   business ("Classic Software" in two cities). So the **address** decides, especially the **house number and street**.
3. **About 26% of the messy rows match nothing.** They are decoys, often built to look like real businesses.
4. **The Hindi/Telugu names are word-for-word translations** of the English name from a fixed vocabulary. We learned a
   small dictionary from the training data that turns them back into English (95% exactly).
5. **Postcodes almost never appear**, so they are useless. The city and house number carry the address signal.
6. **France looks like the same kind of mess**, just in French (SARL, rue, accents), but with no answers to learn from.
7. **The decoys follow a pattern (found 26 Sep).** A fake "sibling" of a real business sits at the *same street, a few
   numbers away*: house number + 1, 2, 3, 4, 5, 7, 9, 11, 13 or 21. A real typo instead drops a digit or is off by 1–2.
   Our main model only knew "same number / different number", so it could not tell a sibling from a typo.
8. **France's words look "rare" to our model (found 26 Sep).** The model learned which words are common from US/India
   text. French words like "rue" and French city names never appeared there, so it treats them as rare, meaningful
   words. Two unrelated shops on the same French street then look alike. Also, 35% of French messy rows have no region
   (state), against 3% in training.

---

## 5. How our system works (step by step)

```
raw data
  │
  ├─ 1. Translate Hindi/Telugu/… names to English      (learned dictionary)
  ├─ 2. Clean names and addresses                     (remove "Pvt Ltd", fix short forms, find house number & city)
  ├─ 3. Find candidates: for each messy row, the few   (fast text search, within the same state)
  │     most similar master businesses                 → about 16 candidates per business, 99.2% of true matches kept
  ├─ 4. Prune the shortlist                            (a small model throws away hopeless candidates)
  │                                                    → 4.8 candidates per business (70% fewer), almost no loss
  ├─ 5. Score each candidate pair with a model          (LightGBM, 65 signals: name similarity, same house number,
  │                                                      same city, how many look-alikes exist, …)
  ├─ 6. Re-check the uncertain pairs (India & US)       (a multilingual language model, bge-reranker, reads both records)
  ├─ 7. US only: the house-number check                 (a small model that knows the "sibling" number pattern)
  ├─ 8. Decide                                          (each messy row goes to its best match, only if the score is
  │                                                      high enough; France gets its own, stricter bar)
  └─ 9. Write the two files and check them with the organisers' validator
```

Everything runs with **one command**: `run.py --split test --stage model`. With the saved model scores it takes about
4 minutes.

**Rules we respect:** every model is open-source under MIT/Apache and far below 8 billion parameters (the biggest is
0.57 billion). We use no internet lookups, no address APIs and no outside data, only the provided training data. We
rented an AWS GPU to *train* the language model faster; nothing online is used to make a decision.

---

## 6. How we made sure our numbers are honest

- We kept a slice of the training data aside (the **holdout**) and scored it exactly like the organisers score the test.
- **A mistake we caught:** our first scoring used a small sample of the data, which had fewer decoys than the real
  test. It made our first system look like 0.82 when it was really 0.77. From then on we always score against the
  **full** data (every business and every messy row competing): the "full-corpus holdout".
- When our model suddenly jumped to 0.98, we did not trust it. **Two independent checks** looked for cheating (label
  leakage) and recomputed the score from scratch. Both confirmed it (0.981).
- Anything we tune on the holdout is tuned on one half and reported on the other half.
- A change is only kept if it beats the current best by a clear, repeatable margin. Many ideas were dropped this way
  (section 9).
- Every experiment is written in the ledger (`ledger/*.jsonl`), so every number in this document can be traced.

---

## 7. What happened on 25–26 September

**1. Finding France's problem with the leaderboard (25 Sep evening).** Our first upload (#5) scored **0.980** on the
leaderboard against **0.989** on our holdout. Our holdout has no France, so we suspected France. We used four uploads
that changed *only* France:
- **France left completely empty:** 0.849. From that drop we could compute France's own score: about **0.93**. India +
  US came out at about **0.989**, exactly what the holdout promised. So the whole gap is France.
- **France stricter (bar 0.88):** 0.981, better. France was giving too many matches.
- **France stricter still (bar 0.95):** **0.981142**, our best so far. The gain was flattening out.

**2. A stricter bar for France.** France now uses the 0.95 bar. When a new France method changes its scores, we set
the bar so that France leaves the same share of businesses empty (6.85%) as that best upload. That way no France
answers are needed.

**3. Smaller shortlists (the pruner).** For the new rule, a small model now drops hopeless candidates:
- **16 → 4.8 per business**, and the candidate file went from 377 MB to 130 MB.
- It costs 0.0002, which the leaderboard confirmed (0.980954, exactly the predicted small loss).

**4. A stronger language model, trained on AWS.** We swapped the small e5 language model (118M parameters) for
**bge-reranker-v2-m3** (568M, Apache-2.0):
- It was trained for 76 minutes on a rented cloud GPU (an A10G on Amazon SageMaker) for **$4.70**.
- It gains **+0.0006** on India/US.
- Training the small model for 2 rounds instead of 1 had already given +0.0004.

**5. House-number siblings.** Once we knew the "sibling" pattern (discovery 7), a small extra model learned it for the
US: +0.0005 on the US.
- For India it did not help, because the stronger language model already catches those cases.
- For France we prepared a simple rule instead: drop matches at a sibling distance (+3, +5, +7…). The leaderboard tells
  us whether France behaves like the US here.

**6. Fixing France's "rare words" (the IDF fix).** For France we recompute which words are common using France's own
(unlabelled) messy rows. So "rue" and "Lille" stop looking special.
- We tested the idea on India, where we have answers: pretending India's words are unknown costs 1.2 points, and this
  fix wins back 96% of it.
- It reads unlabelled test text (no answers), which may or may not count as "only the training data". So we also built
  a train-data-only version.

**Now:** all France options sit on the same India/US base, so each leaderboard score tells us what that one France
change is worth.

---

## 8. Results

| Submission | What changed | Our holdout score | Leaderboard |
|---|---|---|---|
| #2 | Simple rules: exact cleaned name + address check | 0.771 | — |
| #3 | A trained model (LightGBM) instead of rules | 0.981 | — |
| #4 | + the language model re-checks uncertain pairs (India/US) | 0.988 | — |
| #5 | + the model trained on 4× more data | 0.9888 | 0.980017 (rank 33) |
| D1 | #5 with France left empty (a test, not a real answer) | — | 0.849163 → France ≈ 0.93 |
| D2 | #5 with a stricter France bar (0.88) | — | 0.981024 |
| **D4** | #5 with France stricter still (0.95) | — | **0.981142** (best so far) |
| #6c | + the pruner (4.8 candidates instead of 16) + France 0.95 | 0.9885 | 0.980954 |
| #7 | + the language model trained 2 rounds | 0.9888 | not uploaded (superseded by #8c) |
| **#8c** | bge language model (AWS) + US house-number check | **0.9899** | TBD (26 Sep) |
| #9a / #9b / #9s | #8c + one France fix each: rare-word (IDF) fix / bge for France / sibling rule | = #8c (no France answers) | TBD (26 Sep) |
| final | #8c + whichever France fixes win | 0.9899 | TBD |

The jump from rules to a trained model (+21 points) was the biggest win: the model learned when a same-name business
at a different street is *not* the same business.

**Why the leaderboard is lower than our holdout:** France, now confirmed by upload D1. India + US score on the
leaderboard what our holdout predicted (≈ 0.989). France started at ≈ 0.93, and every France fix above targets that gap.

---

## 9. Things we tried and dropped (and why)

| Idea | Result | Why dropped |
|---|---|---|
| A label-free statistical matcher (Fellegi–Sunter) | 0.27 | Far worse than a trained model |
| A neural transliterator (IndicXlit) | 93% vs our dictionary's 99.6% | Slower and worse |
| A multilingual embedding model for Hindi names | 69% vs 99.6% | Much worse on short names |
| A fancier decision layer (P5 "stack") | +0.0001 | Too small to justify the complexity |
| A two-stage model | +0.0008 (before the language model) | The language model already covers most of it |
| The language model on France, trusted as much as on India/US | −0.0016 | It doesn't transfer to a country it never saw. Trusting it less works (option #9b) |
| A bigger e5 language model (2.4× the size) | +0.0001 | Size was not the problem |
| Mixing two language models | below the best single one | No gain |
| Invented France-like training examples (P12) | −0.8 points on a test country | Only works if we already know the country's error rates, which we can't for France |
| Training the main model on twice the data (P16, on AWS, $0.63) | +0.00014 | Too small; the language model already fixes those pairs |
| Filling France's missing region from the city (P14) | +0.00008 | The missing region barely matters |
| The house-number model on India | −0.0001 | The stronger language model already catches those cases |
| A fixed "top 5 per business" shortlist | −0.021 | True matches hide deep in long lists of look-alikes; the learned pruner is far better |
| Qwen3 reranker | never ran (our cloud job crashed on a typo) | bge had already won |

## 10. Problems we hit along the way (and fixed)

- **Laptop memory:** probes crashed when a job used too much memory. We added memory caps and let jobs spill to swap.
  The crashing job, not the probe, is now the one that stops, and a guard pauses the biggest job when memory runs low.
- **Disk:** it filled up to 1.8 GB free. We cleaned old files and caches and added a disk guard.
- **Too many candidates at test scale:** one rule-based step exploded to 10 GB of memory. We fixed it by limiting the
  search to the same state, which was also slightly more accurate.
- **The laptop GPU slowed to 1/15 speed** after a battery drain (early on 26 Sep). Every language-model score is now saved,
  so the final run does not need the GPU, and the big model was trained on AWS instead.
- **A cloud job died at the very end** because a comment swallowed part of a settings line. The model itself was saved;
  re-scoring cost $0.83. Lesson: rehearse with the exact real settings.

---

## 11. What we are doing right now (26 Sep)

1. **Leaderboard uploads, one France change each:** #8c (the new base), then #9a (IDF fix), #9b (bge for France), #9s
   (sibling rule). The fifth slot combines the winners.
2. **Final package:** this document, the methodology document and `Yeti_submission.zip`, all brought up to the current
   pipeline (P18). The zip was built and checked (97 MB, validator PASS).

## 12. The plan for the rest of the challenge

| When | What |
|---|---|
| 26 Sep | 5 uploads (above); pick the final France option from the scores |
| 27 Sep, by 14:00 | Freeze. Re-run everything from a clean copy to prove it reproduces |
| 27 Sep, by 22:00 | Final upload + the final zip (`Yeti_submission.zip`: outputs + code + methodology document), with 2 hours of buffer before the deadline |

---

## Glossary (plain words)

| Word | Meaning |
|---|---|
| **Candidate / shortlist** | The few rows we seriously consider as possible matches for a business |
| **Blocking** | The fast step that builds the shortlists, so we never compare everything with everything |
| **Pruner** | A small model that removes hopeless candidates from the shortlists |
| **Holdout** | Training data we hide from the model and use to measure it honestly |
| **Full-corpus holdout** | Scoring the holdout while *all* businesses and rows compete: the fair mirror of the test |
| **LightGBM** | A fast, popular machine-learning model made of many small decision trees |
| **Cross-encoder / language model** | An AI model that reads two records side by side and says how likely they match (ours: e5-small, then bge-reranker) |
| **Threshold / bar** | The minimum score a pair needs before we call it a match |
| **IDF** | A number saying how rare (and so how telling) a word is; common words like "street" count little |
| **Sibling** | A decoy business at the same street as a real one, a few house numbers away |
| **AWS SageMaker** | Amazon's service for renting a GPU for a training job, paid by the second |
| **Probe** | One AI helper session working on one sub-problem |
| **Worktree** | A separate copy of the code so parallel work doesn't collide |
| **Ledger** | The file where every experiment's settings and scores are recorded |

*Details for specialists: `docs/PROBLEM.md` (rules and data facts), `Documentation_template.md` (technical methodology),
`probes/MERGE-*.md` (every decision with its evidence), `SUBMISSIONS.md` (the upload queue).*
