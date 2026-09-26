# N48-fr-filler-tier — France same-address variants whose name differences are French filler phrases
Read probes/NIGHT/AGY_COMMON.md first, then probes/P46-noise-word-tier/RESULT.md and p46.py, and P20's R1.

Observation (master, reading unaccepted France argmax pairs, p 0.3–0.99, P31 base): many are at the IDENTICAL address
(hn + street + city) and the SX name = S1 name with one descriptive word dropped and/or a filler phrase added, e.g.
patterns like "<X> <Word> SARL" -> "<X> SARL & Fils" / "<X> SARL Et Fils" / "<X> + Fils" / "<X> SARL Cie" /
"<X> SARL & Associés" / "<X> Section" / "<X> Groupe" / "<X> Développement". P46's lens only selected the single gained
word "développement" (and "groupe"); R1 strips some fillers from the core, so these pairs may fall outside P46's
one-word-swap cell (they look like word DROPS, or multi-token gains).

Task:
1. Quantify, on data/work/p31/pairs_France.parquet (argmax, not acc, SX free, hn_cls != 1), the cells:
   identical hn+street+city (define "identical" as P28/P46 do), name relation = drop-one-word / swap-one-word /
   add-filler-phrase, grouped by the gained token(s) (raw, lowercased, accents folded, incl. '&', '+', 'et' phrases)
   and the dropped word. Report counts and P46-style label-free lens stats per gained phrase (sibling-offset share s,
   same-hn share e) — computed on ALL argmax pairs carrying that phrase, as P46 does.
2. Measure the US and India analogues on holdout truth (p23 dumps, holdout==True): same-address + drop-one-word
   (+ any gained filler/legal token) pairs, precision overall and among not-accepted (pred==False) pairs, by p band.
   Also check train truth in full (gt_pairs) for "same exact address, S1 word dropped" SX: fraction true for that S1.
3. Pick the rule (phrases/cells) that passes the VALIDATION BAR. Also make sure decoy-marker words (the generator's
   sibling decoys carry words like holding/participations/distribution/france/services WITH a changed hn) are excluded
   by the lens, not by a hand list.
4. p48.py build <base> <out> + selftest; report how many pairs it would add on the P31 proxy base and how many of those
   P46's rule (développement/groupe) would also add (de-duplicate: the build must skip pairs already in the base).
Output folder for any staged test file: output/sub48_<name>/ (only if you can build from a local base; otherwise just
the script + counts). Deadline: RESULT.md final by 07:30 IST.
