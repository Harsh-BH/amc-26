# Our approach — Amazon ML Challenge 2026 (explained simply)

*Last updated: 25 Sep 2026, evening. Deadline: 27 Sep 2026, 23:59 IST.*

---

## 1. The challenge in one minute

Amazon gets information about the same businesses from **3 different sources**. The sources don't share an ID, and
each one writes things differently — short forms, typos, missing parts, even different scripts (Hindi, Telugu…).

- **Source 1** is the clean master list: every business appears once.
- **Sources 2 and 3** are messy copies.

**Our job:** for every business in Source 1, find all the rows in Sources 2 and 3 that are the *same real business*.

Example:

| Source | Name | Address |
|---|---|---|
| Source 1 | Ram Marketing Private Limited | 12 MG Road, Pune |
| Source 2 | Ram Mktg Pvt Ltd | MG Rd, Pune | ← same business |
| Source 2 | राम मार्केटिंग प्राइवेट लिमिटेड | Near SBI ATM, Pune | ← same business, written in Hindi |
| Source 3 | Ram Traders | 5 FC Road, Pune | ← different business (a look-alike) |

The answer for that Source 1 business is: the first two rows. The third looks similar but is a trap.

**Size:** the test set has **1.73 million** Source 1 businesses and about **10 million** rows in Sources 2 and 3.
Training data (with the correct answers) covers the **US and India**. The test set **also contains France**, which has
no training answers at all.

## 2. How we are scored

For each Source 1 business, the organisers compare our list with the true list:

- **Precision** — of the matches we gave, how many were right?
- **Recall** — of the true matches, how many did we find?

They combine these into a score called **F0.5**, which counts **precision twice as much as recall**: a wrong match hurts
more than a missed one. A business that truly has no match scores 1.0 if we leave it empty and 0 if we match anything.
The final number is the average over all businesses. 1.0 is perfect.

A new rule (25 Sep): the **candidate list** we produce (the shortlist we consider for each business) also counts —
**smaller shortlists rank higher**, on top of the leaderboard score.

---

## 3. The first plan we decided on (and why)

We had 72 hours, one laptop, and a big dataset, so we split the problem into **small sub-problems** and attacked them
**in parallel**:

- One **"master"** session (the coordinator) plans the work, checks every result and decides what to keep.
- Several **"probes"** — separate AI sessions (Claude Opus 5.5), each running in its own terminal window (tmux) — each
  solve one sub-problem and write a short report plus their numbers into a shared **results ledger**.
- Each probe works in its own copy of the code (a **git worktree**), so they can't break each other; the master merges
  their work carefully and runs the tests after every merge.

The sub-problems (the "waves"):

| Wave | Probe | Question it answered |
|---|---|---|
| A | P1 data analysis | What kinds of mess are in the data, and where do we lose points? |
| A | P2 scripts | How do we compare a Hindi/Telugu name with an English one? |
| A | P3 candidates | How do we quickly find a short list of possible matches for each business? |
| A | P7 scale | Can the full run fit on a 15 GB laptop? |
| B | P4 matcher | Which model best tells a true match from a look-alike? |
| B | P6 France | How do we stay accurate on France with no answers to learn from? |
| C | P5 decisions | How do we turn match scores into final answer lists? |
| C | P4b reranker | Can a small language model fix the hardest cases? |
| — | P8 / P9 pipeline | One command that goes from raw data to the final files |
| — | P10 candidates | Make the shortlists smaller (new rule) |

Why this plan: every stage of the problem has its own way of failing, so each probe could be measured on its own
number, and the master only kept what the numbers supported.

---

## 4. What the data taught us (the important discoveries)

1. **Every messy row belongs to at most one master business.** A row in Source 2 can't match two Source 1 businesses.
   We use this everywhere — each messy row goes to its single best match.
2. **Half the businesses have look-alikes.** About 50% of Source 1 businesses share their cleaned name with another
   business ("Classic Software" in two cities). So the **address** decides — especially the **house number and street**.
3. **About 26% of the messy rows match nothing.** They are decoys, often built to look like real businesses.
4. **The Hindi/Telugu names are word-for-word translations** of the English name from a fixed vocabulary. We learned a
   small dictionary from the training data that turns them back into English exactly (95% exact).
5. **Postcodes almost never appear**, so they are useless; the city and house number carry the address signal.
6. **France looks like the same kind of mess**, just in French (SARL, rue, accents) — but with no answers to learn from.

---

## 5. How our system works (step by step)

```
raw data
  │
  ├─ 1. Translate Hindi/Telugu/… names to English      (learned dictionary)
  ├─ 2. Clean names and addresses                     (remove "Pvt Ltd", fix short forms, find house number & city)
  ├─ 3. Find candidates  — for each messy row, the few  (fast text search, within the same state)
  │     most similar master businesses                 → about 16 candidates per business, 99.2% of true matches kept
  ├─ 4. Score each candidate pair with a model          (LightGBM, 65 signals: name similarity, same house number,
  │                                                      same city, how many look-alikes exist, …)
  ├─ 5. Re-check the uncertain pairs (India & US)       (a small multilingual language model reads both records)
  ├─ 6. Decide                                          (each messy row goes to its best match, only if the score is
  │                                                      high enough; France gets its own, stricter bar)
  └─ 7. Write the two files and check them with the organisers' validator
```

Everything runs with **one command**: `run.py --split test --stage model`.

**Rules we respect:** every model is open-source under MIT/Apache and far below 8 billion parameters; we use no
internet lookups, no address APIs, no outside data — only the provided training data.

---

## 6. How we made sure our numbers are honest

- We kept a slice of the training data aside (the **holdout**) and scored it exactly like the organisers score the test.
- **A mistake we caught:** our first scoring used a small sample of the data, which had fewer decoys than the real
  test. It made our first system look like 0.82 when it was really 0.77. From then on we always score against the
  **full** data (every business and every messy row competing) — the "full-corpus holdout".
- When our model suddenly jumped to 0.98, we did not trust it: **two independent checks** looked for cheating (label
  leakage) and recomputed the score from scratch. Both confirmed it (0.981).
- Every experiment is written in the ledger (`ledger/*.jsonl`), so every number in this document can be traced.

---

## 7. Results so far

| Submission | What changed | Our holdout score | Leaderboard |
|---|---|---|---|
| #2 | Simple rules: exact cleaned name + address check | 0.771 | — |
| #3 | A trained model (LightGBM) instead of rules | 0.981 | — |
| #4 | + the language model re-checks uncertain pairs (India/US) | 0.988 | — |
| **#5** | + the model trained on 4× more data | **0.9885** | **0.980** |

The jump from rules to a trained model (+21 points) was the biggest win: the model learned when a same-name business
at a different street is *not* the same business.

**Why the leaderboard (0.980) is lower than our holdout (0.9885):** our best guess is **France**. Our holdout only has
US and India. If those score as expected, France must be around **0.93** — which is exactly what our "train on one
country, test on another" experiments predicted for an unseen country. We are now checking this with the leaderboard.

---

## 8. Things we tried and dropped (and why)

| Idea | Result | Why dropped |
|---|---|---|
| A label-free statistical matcher (Fellegi–Sunter) | 0.27 | Far worse than a trained model |
| A neural transliterator (IndicXlit) | 93% vs our dictionary's 99.6% | Slower and worse |
| A multilingual embedding model for Hindi names | 69% vs 99.6% | Much worse on short names |
| A fancier decision layer (P5 "stack") | +0.0001 | Too small to justify the complexity |
| A two-stage model | +0.0005 | Too small for the extra complexity |
| Using the language model for France | −0.0016 | It doesn't transfer to a country it never saw |

## 9. Problems we hit along the way (and fixed)

- **Laptop memory:** probes crashed when a job used too much memory. We added memory caps, let jobs spill to swap,
  made the crashing job (not the probe) be the one that stops, and built a guard that pauses the biggest job when
  memory runs low.
- **Disk:** it filled up to 1.8 GB free; we cleaned old files and caches (now ~10 GB free) and added a disk guard.
- **Too many candidates at test scale:** one rule-based step exploded to 10 GB of memory; we fixed it by limiting the
  search to the same state, which was also slightly more accurate.

---

## 10. What we are doing right now

1. **Confirming the France gap** with three quick test uploads that change *only* France (empty / stricter / looser).
2. **P9** — making the one-command pipeline reproduce submission #5 exactly (for the final package).
3. **P10** — making the candidate shortlists **smaller** (from ~16 to ~4–6 per business) with a learned filter, without
   losing accuracy — because of the organisers' new rule.

## 11. The plan for the rest of the challenge

| When | What |
|---|---|
| 25 Sep night | France diagnosis from the leaderboard; merge P9; P10's smaller shortlists (submission #6) |
| 26 Sep | Improve France (French-specific cleaning rules, better threshold); add the shortlist filter to the pipeline and the document; one or two more improvements, each kept only if it measurably helps |
| 27 Sep, by 14:00 | Freeze. Re-run everything from a clean copy to prove it reproduces |
| 27 Sep, by 22:00 | Final upload + the final zip (`<team_name>_submission.zip`: outputs + code + methodology document) — 2 hours of buffer before the deadline |

---

## Glossary (plain words)

| Word | Meaning |
|---|---|
| **Candidate / shortlist** | The few rows we seriously consider as possible matches for a business |
| **Blocking** | The fast step that builds the shortlists, so we never compare everything with everything |
| **Holdout** | Training data we hide from the model and use to measure it honestly |
| **Full-corpus holdout** | Scoring the holdout while *all* businesses and rows compete — the fair mirror of the test |
| **LightGBM** | A fast, popular machine-learning model made of many small decision trees |
| **Cross-encoder / language model** | A small AI model that reads two records side by side and says how likely they match |
| **Threshold** | The minimum score a pair needs before we call it a match |
| **Probe** | One AI helper session working on one sub-problem |
| **Worktree** | A separate copy of the code so parallel work doesn't collide |
| **Ledger** | The file where every experiment's settings and scores are recorded |

*Details for specialists: `docs/PROBLEM.md` (rules and data facts), `Documentation_template.md` (technical methodology),
`probes/MERGE-*.md` (every decision with its evidence), `SUBMISSIONS.md` (the upload queue).*
