# P52-fr-calib — RESULT (DRAFT 2, 27 Sep 06:10 — France IW grid running)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Arms 1, 3 and 4 fail as briefed.** Arm 2 fails source→target on the chain analogue, but it is the only lever with real
  signal. The LOCO target is **over-confident** in the band, not prior-shifted: π_T = π_S = 0.726, and p 0.5–0.9 pairs
  are only 20–44% true. So every "calibrate, then apply the source-optimal bars" arm admits false pairs. EM π̂ is
  0.76–0.83 against a true 0.726; even the true-π shift loses −0.009 / −0.013.
- Frame→frame transfer (tune on one LOCO frame, transfer to the other): **importance weighting beats P42's density map
  by +0.0036 / +0.0068 on chain** (raw bars) and **+0.0017 / +0.0057 when it goes through the density map**.
- Pending: the France form of arm 2 (P42's France-bar grid with LOCO S1s weighted toward France: AUC 0.956 / 0.945,
  ESS 12% / 11%), then the ship gate.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- Pending the France IW grid.

## Evidence         (table: ledger name | world | key metrics | verdict)
France-F0.5 dF vs the current recipe analogue:
- P42 density-mapped t .99 / r .9 / gc .7 / hn .8, caps 5/6, τ 0.1, plus P48's rescue/gc band veto at r < 0.2;
- each cell reads US→India / India→US (the LOCO target country is after the arrow);
- SE ≈ 0.0002–0.0006.

**Arms as briefed (source = the source country's own fold 0, labelled; target = P42 LOCO frame)**
| ledger name | arm | chain | m2 | verdict |
|---|---|---|---|---|
| arm1-em-prior-shift-src2tgt | EM (BCTS + Saerens) | −0.0159 / −0.0142 | −0.0461 / −0.0023 | drop |
| (same row) | true-π shift (the ceiling) | −0.0091 / −0.0130 | −0.0371 / −0.0014 | — |
| (same row) | BCTS only | −0.0071 / −0.0129 | −0.0351 / −0.0013 | — |
| (same row) | π̂ vs true π_T | 0.796 vs 0.726; 0.758 vs 0.726 | 0.829 / 0.754 vs 0.726 | EM over-estimates |
| arm4-qmap-top1-src2tgt | top-1 quantile map + source bars | +0.0016 / −0.0058 | −0.0055 / −0.0020 | drop |
| (same row) | naive (source bars, raw) | −0.0105 / −0.0089 | −0.0327 / −0.0005 | — |
| arm2-iw-bars-src2tgt | IW source bars (S1 clf AUC 0.89 / 0.92, ESS 0.18 / 0.17) | +0.0020 / **−0.0091** | +0.0059 / +0.0098 | drop (chain fails) |

**Transfer rules between unseen-country frames** (A's oracle bars → B; B's labels only score) — `xfer-rule-tournament-frameA2frameB`
| rule | chain A=India-tgt → B=US-tgt | chain US-tgt → India-tgt | m2 India-tgt → US-tgt | m2 US-tgt → India-tgt |
|---|---|---|---|---|
| B oracle (in-sample ceiling) | +0.0022 | +0.0034 | +0.0183 | +0.0140 |
| raw (same bars) | −0.0041 | −0.0029 | +0.0034 | +0.0006 |
| **density (P42's rule)** | −0.0039 | −0.0041 | −0.0031 | −0.0009 |
| qmap top-1 | −0.0039 | −0.0031 | +0.0023 | +0.0036 |
| em | −0.0028 | −0.0045 | −0.0021 | −0.0062 |
| **iw (raw bars)** | **−0.0003** | **+0.0027** | **+0.0161** | **+0.0074** |
| **iw → density** | −0.0022 (+0.0017 vs density) | +0.0016 (+0.0057) | −0.0012 (+0.0019) | +0.0006 (+0.0015) |

**Arm 3** — `arm3-adversarial-rank5-refit`:
- Adversarial AUC on P39's 20 frame columns: France vs LOCO **0.9995**; US target vs India target 0.994.
- Top shifted columns: lt 0.27, q 0.21, best_oth_lt 0.20, state_eq 0.14, best_src_ex_lt 0.05 (gain share).
- P39 scorer refit, add at 0.8 on top of the recipe:

| train labels → apply | raw | rank5 | drop5 | verdict |
|---|---|---|---|---|
| US → India | −0.087 | −0.058 | −0.083 | drop |
| India → US | +0.0023 | +0.0014 | +0.0006 (addrm 0.9: +0.0080) | drop |

## What failed      (approaches tried and dropped, with the number that killed them)
- See the table: EM, the true-π oracle, BCTS, qmap, naive, and arm 3.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `prep52.py US|India|france`:
  - `US|India` → `data/work/p52/src_<src>.npz`;
  - `france` → `data/work/p52/france_hn_rel.npy`.
- `p52.py loco` → `loco.json`; `p52.py xfer` → `xfer.json`; `p52.py xfer_iwd`; `p52.py fr_iw` → `fr_iw.json` + `data/work/p52/fr_iw_grid.parquet`.
- `arm3.py` → `arm3.json`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Source-side chain p uses P45's in-domain CE (r_all), not P30's source-only CE (no source-side scores exist).

## Open questions   (for the master; each with your best guess)
- Pending.
