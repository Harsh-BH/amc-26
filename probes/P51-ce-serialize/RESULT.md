# P51-ce-serialize — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- DRAFT (04:35 IST): serializer built and self-checked; P13's exact LOCO pairs re-texted (India/US train/val/holdout,
  France/India/US test bands); LOCO jobs launched 04:21 (US on ml.g6e.2xlarge, India racing g6e.4xlarge / g6.xlarge /
  g5.4xlarge). Gate verdict due ~06:30.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- Pending the gate (pass = serialized − raw bge ≥ +0.0010 France-F0.5 in both LOCO directions, mean-OOD weights).

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| (serialize self-check) | synthetic | France '(8) R. Konrad Adenauer' → hn 8 / street 'adenauer konrad r' / région; India 'Pvt Ltd' + MH; US LLC + IL; gib brand + noaddr flags | pass |
| (token lengths, 3k train pairs) | train | bge tokens per pair, serialized vs raw: India p50 119 vs 54, p99 172 vs 88; US p50 92 vs 37, p99 124 vs 54. maxlen 160 truncates 3.3% India / 0% US (longest_first → the trailing raw name goes first) | – |
| (gate code check, P51_NOLOG) | holdout-dense-half1 | `p51.py gate` with raw bge in both arms: difference exactly 0; bge over no-CE +0.0028 / +0.00148 = P13's frproxy; 52 s, 1.54 GB | pass |
| (sub51 chain check) | test-France | `sub51.sh bge <P13 W_FR>` (build38 blend → tpick 0.9868 → P31 post → on43 (t42 0.99) → P46 swap lens (+3,185) → splice) vs output/sub44r: **2 France lines differ** (= P38's repro43 gap). `veto51.py` (P48's veto with the chain's own r / blend) + splice into output/sub48: 1,159 vetoed (rescue 44 / gc 1,115) vs P48's 1,138 (41 / 1,097) → **23 France lines differ** from sub48 (set level; byte order inside lines differs, P46 appends unsorted) | pass |
| (LOCO job AUCs, log.jsonl) | holdout band | **Serialized bge trained on US → India (unseen) band AUC r 0.8654** (mid 0.8536; stack 0.9534) vs raw bge 0.821 (P13) and Qwen3-4B 0.856 (P38); US in-domain 0.9535 (stack 0.9655). Train US 483,712 pairs at 501 pairs/s on one L40S (936 s, 1 epoch, micro 64 at L 160, 17.6 GB) | – |
| (channel check) | holdout | holdout rows = P13 LOCO r files' rows exactly (193,751); train India 294,766 / US 483,754 = p13.build's selection (seed 0) | pass |

## Jobs (AWS, --profile harsh, ap-south-1, on-demand)
| job | instance | purpose | status | $ |
|---|---|---|---|---|
| p51-us-g6e2-0927-0421 | ml.g6e.2xlarge | LOCO: train US (1 epoch) → holdout band (mid + end) | Completed, 1,417 billable s | 1.33 |
| p51-us-g62-0927-0421 | ml.g6.2xlarge | duplicate | stopped at Downloading (g6e got capacity too) | ~0 |
| p51-in-g6x-0927-0421 | ml.g6.xlarge (1× L4) | LOCO: train India (294,656 pairs, 120 pairs/s, 2,333 s) → holdout band | running | |
| p51-in-g6e4-0927-0421 / p51-in-g54-0927-0421 | g6e.4xlarge / g5.4xlarge | India duplicates | stopped while Pending / at Downloading | ~0 |
| p51-in-g5x (not created) | ml.g5.xlarge | – | refused: quota | 0 |

## What failed      (approaches tried and dropped, with the number that killed them)
- (none yet)

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `serialize.py` — `serialize(df[name, business_name, business_address, country], M) -> np.ndarray[str]`:
  `name: <name_core> | legal: <canonical legal forms> | hn: <hn> | street: <street_key> | city: <city> | area: <other
  segments> | state: <code> | pin: <postcode> | brand | noaddr | raw: <P3 Latin name>`; empty fields omitted, flags only
  when true. Parsers: normalize.name_core / name_norm / _is_legal / address_fields, france.paren_hn / street_key,
  P36 gib.is_gib (split's learned syllables). No country branching (the label only picks normalize's state table).
- `build51.py build [India,US]` — P13's train/val/holdout pairs with serialized texts (~1 min per country, 1.3 GB);
  `build51.py test <c>` — m2_big's test band (chunked); `build51.py show` — 20 examples per country →
  data/work/p51/examples.txt (data rows: not in git).
- `launch51.py <phases> <instance> <job> [hp.json]` — P13's solution_p13.py + bundle on `amc26-p51/data/v1/`;
  hyperparameters from `hp/loco.json` (maxlen 160, cap 9000 s so the full epoch runs, deadline 13,500 s).
- `p51.py fetch <job> <tag>` / `p51.py gate <tag_US> <tag_India> [control tags]` — P38's gate with an own ledger row.
- `poll.sh <jobs…>` — status + log events.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Assumption: the smoke pass is the first minutes of the real LOCO jobs (tokens + train_speed events). The job code is
  P13's unchanged solution_p13.py (ran these exact phases in P13), only the channel and maxlen differ.
- Assumption: empty fields omitted (token budget: serialized text is already 2.2–2.5× the raw text).

## Open questions   (for the master; each with your best guess)
- (pending)
