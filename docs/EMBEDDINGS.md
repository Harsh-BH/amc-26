# Embedding models — what to reach for

Written before the problem dropped. Bedrock availability was checked live against
`ap-south-1`; model rankings are from MTEB v2 / public benchmarks as of Sep 2026.

## The workflow that wins a 72-hour run

**Freeze the backbone, cache the vectors, iterate on the head.** Embedding 250k images
takes an hour once. Re-running it for every experiment is how teams lose Saturday.

```
raw text/images  ──(once)──>  embeddings.npy  ──(seconds per experiment)──>  LightGBM / MLP head
```

Fine-tuning the backbone is a *day-three* move, and only if the frozen-feature score has
plateaued. A frozen SigLIP2 + GBDT head beats a rushed end-to-end fine-tune almost every time.

Cache to disk keyed by a hash of (model name, preprocessing), so a changed preprocessor
invalidates the cache instead of silently serving stale vectors.

## Pick by problem shape

| Problem | Reach for | Why |
|---|---|---|
| Text → number (price, length, rating) | `bge-m3` or `gte-multilingual-base` + LightGBM | Strong, 1024-dim, cheap to run |
| Text → label / attribute | same, or fine-tune `deberta-v3-base` | Frozen features first, fine-tune if plateaued |
| Image → number / label | **DINOv2/v3 ViT-B/14** | Vision-only beats CLIP-family on pure image→image |
| Image + text → target | **SigLIP2**, concat both towers | Aligned space, strong on product imagery |
| Text query finds images | CLIP / SigLIP2 | Cross-modal alignment is the whole point |
| Near-duplicate / product matching | DINOv2/v3 | Self-supervised wins fine-grained matching |

The rule from the benchmarks: **a human types words to find pictures → cross-modal
(CLIP/SigLIP). A picture finds pictures → vision-only (DINO).** Test two on 5k rows before
committing the full run.

## Text

| Model | Dim | Notes |
|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | Top of MTEB v2 open-weight; 8B variant is SOTA but far too heavy here |
| `BAAI/bge-m3` | 1024 | Best all-rounder, 100+ languages, dense+sparse hybrid |
| `Alibaba-NLP/gte-multilingual-base` | 768 | Fast, strong, small — good default under time pressure |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 22M params. Use as the *baseline*, not the answer |

On a 6 GB RTX 3050, `gte-multilingual-base` and `bge-m3` run fine at batch 32 in fp16.
`Qwen3-Embedding-0.6B` fits but leaves little headroom — drop batch to 8–16.

## Image

| Model | Notes |
|---|---|
| `facebook/dinov2-base` (ViT-B/14) | Vision-only SSL. Benchmarks show ~93% classification with **no fine-tuning** |
| DINOv3 | Newer SSL backbone, stronger on dense + fine-grained tasks |
| `google/siglip2-base-patch16-224` | Sigmoid loss, multilingual, dynamic resolution. Beats original CLIP/SigLIP |
| MetaCLIP 2 / Perception Encoder | Meta's CLIP successors, also ahead of original CLIP |
| `openai/clip-vit-large-patch14` | The known quantity. Fine, but no longer the best |

Product images are the 2024/2025 pattern — **start with SigLIP2 if text is also in play,
DINOv2 if the target is purely visual.**

## Bedrock — verified live in ap-south-1

| Model ID | Input | Access |
|---|---|---|
| `amazon.titan-embed-text-v2:0` | TEXT | `ON_DEMAND` — works immediately |
| `amazon.titan-embed-image-v1` | TEXT + IMAGE | `ON_DEMAND` — works immediately |
| `cohere.embed-v4:0` | TEXT + IMAGE | **`INFERENCE_PROFILE` only** — extra setup |
| `cohere.embed-english-v3` | TEXT | on-demand |
| `cohere.embed-multilingual-v3` | TEXT | on-demand |

**The gotcha:** `cohere.embed-v4:0` needs a cross-region inference profile — you cannot just
call `invoke_model` with that id. Titan works straight away. Do not discover this at hour 3.

Bedrock is billed per token, not per hour, so it does not eat GPU credits — but 250k images
through Titan Multimodal is a real cost. Price a 1k sample before committing.

```python
import base64, boto3, json
br = boto3.client('bedrock-runtime', region_name='ap-south-1')

def titan_text(text):
    r = br.invoke_model(modelId='amazon.titan-embed-text-v2:0',
                        body=json.dumps({'inputText': text, 'dimensions': 1024}))
    return json.loads(r['body'].read())['embedding']

def titan_image(path, text=None):
    body = {'inputImage': base64.b64encode(open(path, 'rb').read()).decode()}
    if text:
        body['inputText'] = text          # joint text+image vector
    r = br.invoke_model(modelId='amazon.titan-embed-image-v1', body=json.dumps(body))
    return json.loads(r['body'].read())['embedding']
```

## Local, the way you will actually use it

```python
# text - sentence-transformers handles pooling and normalisation for you
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True, device='cuda')
X = m.encode(df.catalog_content.tolist(), batch_size=64, normalize_embeddings=True,
             show_progress_bar=True, convert_to_numpy=True)
np.save('cache/text_gte.npy', X)

# image
import torch
from transformers import AutoImageProcessor, AutoModel
proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
net = AutoModel.from_pretrained('facebook/dinov2-base', torch_dtype=torch.float16).cuda().eval()
with torch.inference_mode():
    out = net(**proc(images=batch, return_tensors='pt').to('cuda'))
    feats = out.last_hidden_state[:, 0]        # CLS token
```

Always `normalize_embeddings=True` for text. For images, L2-normalise the CLS token yourself
before feeding a GBDT — unnormalised vectors make tree splits behave badly.

## Traps

- **Do not** re-embed on every run. Cache, keyed by model + preprocessing hash.
- **Do not** feed raw 1024-dim vectors to LightGBM and stop there — PCA to 128–256 often
  scores the same and trains 5× faster, which matters when the clock is the constraint.
- **Do not** fine-tune the backbone before a frozen baseline exists. You will have no floor to compare against.
- Truncation kills long product text. Check the model's max sequence length against your
  99th-percentile token count before trusting the vectors.
