# P2-script: cross-script names and addresses (India)
Goal: make Indic-script S2/S3 names and addresses comparable to their Latin S1 counterparts.
Facts: S1 is always Latin; ~23.5% of India S2 names and ~13% of S3 names are Indic script; 18% of India true
pairs cross scripts (~550k labelled cross-script pairs in train — use them).
1. Inventory which scripts occur (Devanagari, Telugu, Tamil, Bengali, Gujarati, Kannada, …) and their shares.
2. Compare, on cross-script true pairs vs same-country hard negatives:
   a) indic-transliteration (MIT) rule romanization + schwa deletion + phonetic folding (aa→a, w→v, sh→s, z→j …)
   b) ai4bharat IndicXlit (MIT, ~11M params) neural romanization — measure throughput on ~1.2M strings (CPU and GPU under flock)
   c) a mapping learned from the train cross-script pairs (char n-gram alignment or a tiny seq2seq) — train only
   d) multilingual embeddings (multilingual-e5-small MIT, LaBSE Apache-2.0) cosine
   Metrics: similarity AUC, and recall@{1,5,20} of the true S1 when retrieving among that country's dev-world S1s.
3. Indic-script addresses: can transliterated city names be matched to Latin city_keys? Learn a city
   dictionary from train pairs if needed.
Deliverable: probes/P2-script/translit.py: to_latin(s: pd.Series) -> pd.Series, plus speed and RAM at full scale
and a recommendation (which method or combination, and where: blocking key, feature, or both).

Note: ai4bharat-transliteration is NOT in the shared .venv (heavy deps). If you evaluate it, install it into
your own venv (uv venv probes/P2-script/.venv) — never pip-install into the shared .venv.
