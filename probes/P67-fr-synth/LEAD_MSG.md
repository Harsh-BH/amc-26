# P67 → lead, 15:45 (the checks you asked for; SendMessage to your socket timed out twice)

1. **Strict LOCO** (target generated with the SOURCE country's op rates, job p67h-bge-loco-0927-1501): **gate PASS**.
   - syn−base at the shipped cell: +0.00119 ± 0.00012 (India→US) / +0.00117 ± 0.00012 (US→India).
   - syn−pre: +0.00113 / +0.00036. The own-rate v1 run gave +0.0019 / +0.0024.
   - On the final chain's analogue (p55n veto; P55 add arm c > .95 on top), strict:

   | adds from | India→US | US→India | precision |
   |---|---|---|---|
   | syn combiner | +0.00266 | +0.00352 | .94 / .876 |
   | old combiner | +0.00236 | +0.00278 | .961 / .836 |
   | **syn combiner, Qwen-gated** | **+0.00270** | **+0.00396** | — |

   - The gate drops adds whose source-only Qwen r < .5. Those adds are only 64% / 49% precise on LOCO.
2. **Add types** (sub67 = 3,302 adds; sub67q = 2,793 after the gate):

   | type | sub67 / sub67q | Qwen median, share r ≥ .5 | LOCO precision (I→U / U→I) |
   |---|---|---|---|
   | extra same-source member | 2,903 / 2,518 | 0.994, 85% (gated 100%) | .938 / .912 |
   | missing-src | 330 / 222 | 0.79, 54% | .94 / .855 |
   | empty S1 | 69 / 53 | 0.993 | .953 / .659 (n 85 / 91) |

   - The P64 legal-form veto is already applied: 94 adds skipped.
3. **Overlap with sub54's LB-rejected France set** (40,888 France pairs in sub54 that sub66_final lacks):
   - 2,418 of the 3,302 sub67 adds are in it; 2,176 of the 2,793 sub67q adds.
   - They are that set's c > .95 high-confidence tail, and the LB never tested them alone. **Main risk.**
   - sub66_t975ce's 5,270 adds share only 359 pairs with sub67, plus 1 SX that goes to a different S1. The two sets are nearly disjoint and can be combined.

**Files**
- `data/work/p67/changes.parquet` (3,302 rows): s1, sx, action, type (empty / missing-src / extra), p, c_new, c_old, qwen_r, bge_r, r_syn, qgate_keep (True = in sub67q).
- `output/sub67q`: validator --check-ids **PASS** vs sub66_final_small's candidates, md5 d7199116. India/US byte-identical; 2,669 France S1 changed.
- `output/sub67` (ungated): PASS, md5 a610c9ac.

**Expected LB for sub67q ≈ +0.0001** = strict LOCO dF × France/LOCO adds per S1 (0.26 / 0.16) × 0.1498.

**Qwen arm:** no g6e capacity after 75 min; the jobs are stopped at 16:00.
