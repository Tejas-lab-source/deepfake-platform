# Build: Filter-Invariant Deepfake Detection Platform

You are building a complete, working project for a B.Tech major project. Read this whole brief before writing code. Then reply with a short implementation plan and the repo tree, and **wait for my "go"** before Phase 1. After that, work phase by phase. At the end of each phase, run its checks, show me the results and stop for review.

---

## 1. What the product does

A web app where a user uploads a face image (or a short video) and gets one of three verdicts: **FAKE**, **REAL** (shown as "No manipulation found", never "authentic") or **UNCERTAIN**.

The research goal: the verdict must stay stable when a face has gone through **cosmetic beauty filters** (skin smoothing, makeup, face reshaping, GAN restoration), including filter types the system never saw in training. Beauty filters break deepfake detectors in two directions:

- real faces get flagged as fake (false alarms);
- fakes pass as real (missed detections).

The system must handle four cases: deepfake, deepfake + filter, real photo, real photo + filter.

Every upload is checked **twice**:

1. **Our pipeline** (FastAPI) gives the main verdict.
2. **Hive's deepfake API** gives an independent second opinion, run in parallel.

A reconciler in Node merges the two. **Groq** (an LLM API) then writes a plain-English explanation of the final result. Groq never decides the verdict.

---

## 2. Hard constraints. Do not deviate.

1. **No training of backbones.** Use pretrained detectors, frozen. The only training allowed:
   - a linear head on cached features;
   - calibration numbers;
   - a logistic-regression fusion.
2. **Stack:**
   - React 18 + Vite + TypeScript + Tailwind (frontend)
   - Node.js + Express + TypeScript (the only public backend)
   - MongoDB (Mongoose)
   - Redis + BullMQ (job queue)
   - FastAPI + Python 3.11 + PyTorch (ML service, private network only)
   - Docker Compose for everything
3. **Groq never sees images and never sets a verdict.** It receives result JSON only.
4. **Hive never flips our verdict.** It can only confirm it, or downgrade it to UNCERTAIN (rules in section 6).
5. **Never fabricate:**
   - model weights, download URLs or API response formats;
   - metrics or results;
   - library APIs. If unsure of an API, check the installed package's source or docs, or ask me.
   
   If a weight file needs a manual download (Google Drive, a dataset form), write a clear script plus README instructions and stop for me.
6. **Everything must run without a GPU and without API keys** through a `MOCK_MODE=true` flag. Mock mode uses deterministic fake scores, so the full app (upload → verdict → explanation) works end-to-end in development. Real mode is used when weights and keys are present.
7. Secrets only in `.env` files (provide `.env.example`). Never commit keys.

---

## 3. Repo layout

```
deepfake-platform/
  docker-compose.yml
  .env.example
  README.md
  web/                      # React + Vite + TS + Tailwind
  server/                   # Node + Express + TS (API, workers, reconciler, Groq, Hive)
  ml-service/
    app/
      main.py               # FastAPI routes
      config.py             # pydantic-settings, MOCK_MODE, K, paths
      pipeline.py           # per-face pipeline, steps 1-10
      faces.py              # MTCNN detection + crops, MediaPipe landmarks
      detectors/gend.py     # GenD (CLIP ViT-L/14) wrapper -> (features 1024-d, logit)
      detectors/sbi.py      # SBI EfficientNet-B4 wrapper -> logit
      beautify/f1_smooth.py f2_makeup.py f3_reshape.py f4_restore.py
      head.py               # linear head + fusion + calibration (loads head.pt, model_card.json)
      explain.py            # Grad-CAM for CNN and ViT, heatmap IoU
      mock.py               # deterministic mock scores
    offline/
      extract_faces.py cache_features.py train_head.py calibrate.py eval_lofo.py hive_batch.py
    tests/
    weights/                # gitignored; download scripts fill it
```

---

## 4. ML service (FastAPI): our main verdict

### Models (pretrained, frozen)

- **GenD (CLIP)**, the primary detector and feature extractor. Hugging Face `yermandy/GenD_CLIP_L_14` (MIT). The model card loads it with `from src.hf.modeling_gend import GenD; GenD.from_pretrained("yermandy/GenD_CLIP_L_14")`; the code lives in GitHub `yermandy/GenD`.
  - Vendor the minimal modeling file, or install from the repo.
  - Expose a function that returns **both** the 1024-d pre-classifier feature and the logit.
  - Verify the preprocessing (crop size and normalisation) from the repo; don't guess.
- **SBI**, the second detector. EfficientNet-B4 weights from the official Self-Blended Images repo (FF++ c23 checkpoint, research-use licence).
  - Write `scripts/download_weights.sh`.
  - If the weights are behind Google Drive, document the manual step.

### Beautification library: identity-preserving filters, each with a `strength` in [0, 1]

| Family | What it does | Use |
|---|---|---|
| F1, skin smoothing | Bilateral, guided and Gaussian filters, only inside a skin mask built from MediaPipe Face Landmarker landmarks (face oval minus eyes, brows and lips) | Training and inference |
| F2, makeup and tone | Skin brightening, lip tint, blush, warmth/saturation, on landmark-defined regions | Training and inference |
| F3, geometric reshaping | Jaw slimming, eye enlargement, nose narrowing, via landmark-guided piecewise-affine warp (scikit-image), with capped magnitudes | Training and inference |
| F4, generative restoration | GFPGAN or CodeFormer | **Test only. Never used in training or in inference copies.** Optional install; skip gracefully if missing |

All four families are applied to both real and fake faces. F3 must also return its transform, so heatmaps can be warped back.

### Per-face pipeline (`pipeline.py`)

1. Validate the file by magic bytes; enforce size limits.
2. Detect faces with MTCNN (facenet-pytorch). Take two crops from the same box: 224 px for GenD, 380 px for SBI. Faces not detected are **excluded**, never scored as real.
3. Compute MediaPipe landmarks once per crop and cache them.
4. Make K = 4 beautified copies (F1, F2, F3, one random 2-filter combination) at random strengths. K is configurable; K = 0 is the fast mode.
5. Run GenD on the clean crop + K copies in one batch (fp16 on GPU), and SBI on the clean crop only.
6. Apply the invariance-trained linear head (`head.pt`) to the GenD features: one logit per copy. If `head.pt` is missing, use GenD's own logit.
7. Aggregate:
   - mean logit over copies (filter-TTA, "M1");
   - spread = max − min ("M2");
   - a training-free `beautifiedLikelihood`: high-frequency energy inside the skin mask relative to hair and background.
8. Fuse and calibrate: logistic fusion over [head logit, SBI logit, spread, beautifiedLikelihood], then temperature + threshold. All values are loaded from `model_card.json`; the defaults are identity fusion, T = 1 and threshold 0.5.
9. Decide. UNCERTAIN if any of these holds:
   - the probability is within `uncertain_margin` of the threshold;
   - spread > `spread_limit`;
   - the copies disagree on the verdict;
   - GenD and SBI strongly disagree.
   
   Otherwise FAKE or REAL. Also return `confident` (a boolean) and `uncertainReason`.
10. Explain:
    - Grad-CAM on SBI's conv head and on GenD's ViT (use pytorch-grad-cam with a ViT reshape transform; target the single logit with a custom target);
    - heatmap IoU (top 20% regions) between clean and one beautified copy;
    - for F3, warp the beautified heatmap back with the inverse transform before computing IoU.

**Video:** sample up to 32 frames uniformly. Aggregate with the mean of the top-k frame logits (configurable). Explain only the most-fake and most-real frames. Report progress.

### Endpoints

Private network only; require an `X-Internal-Secret` header.

- `GET /health`, `GET /ready` (ready once models are loaded), `GET /model-info`
- `POST /analyze/image {path, k?, explain?}`
- `POST /analyze/video {path, maxFrames?, k?}`
- `POST /compare {path, family, strength}`: returns the clean and filtered verdicts, a flip flag, both heatmaps and IoU
- `POST /beautify {path, family, strength}`: returns the filtered image

`/analyze/image` returns:

```json
{"verdict":"UNCERTAIN","probability":0.58,"threshold":0.47,"confident":false,
 "uncertainReason":"unstable_under_beautification",
 "faces":[{"box":[112,80,260,260],"scores":{"gendHead":1.21,"sbi":-0.40},
   "variantLogits":{"clean":0.31,"F1":1.02,"F2":0.44,"F3":-0.12,"mix":0.88},
   "spread":1.14,"beautifiedLikelihood":0.72,
   "heatmaps":{"gend":"hm_1.png","sbi":"hm_2.png"},"heatmapIoU":0.41}],
 "timingsMs":{"detect":38,"beautify":95,"score":610,"explain":240},
 "modelCardVersion":"..."}
```

Models load once at startup, and inference runs behind a lock.

---

## 5. Offline training: the only training, on cached features

The scripts go in `ml-service/offline/` and run in Colab, Kaggle or a local GPU. Each takes CLI args and writes to `weights/` and `artifacts/`.

1. **`extract_faces.py`**
   - Input: Celeb-DF v2 (access requires the authors' form, so document it).
   - Use the official test list (518 videos), about 8 frames per video, MTCNN crops.
   - Build train/val splits **grouped by celebrity identity**.
   - Run a leakage assertion that refuses to write a manifest if any identity or video crosses splits.
2. **`cache_features.py`**
   - For each face, save GenD 1024-d features (float16) for: clean, 2 strengths × each of F1, F2, F3, one generic-augmentation copy (blur, JPEG, colour jitter), and F4 for test faces only.
   - Also save SBI logits for clean faces.
   - Forward-only, batched, resumable.
3. **`train_head.py`** trains a linear head on the cache with:

   ```
   L = BCE(h(z_x), y) + BCE(h(z_e(x)), y) + λp · (h(z_x) − h(z_e(x)))²
   ```

   - λp = 1.0; tune it on validation only.
   - Hardest-of-K: of the cached beautified copies of a face, use the one whose logit moves most.
   - Implement six schemes, each run on 3 leave-one-family-out folds (hold out one of F1/F2/F3; F4 is always held out) × 3 seeds:

   | Scheme | Head trained on | Invariance loss |
   |---|---|---|
   | H-a | No head (GenD as released) | — |
   | H-b | Clean + bilateral smoothing only | No |
   | H-c | Clean + random F1–F3 (the critical control) | No |
   | H-d | Clean + generic augmentations | Yes |
   | H-e | Clean + random F1–F3 | Yes |
   | H-f | Clean + random F1–F3, hardest-of-K (**the headline**) | Yes |

4. **`calibrate.py`** fits, on the validation split only: temperature, threshold, uncertain margin, spread limit and fusion weights. It writes `model_card.json` and records where each value came from.
5. **`eval_lofo.py`**: per scheme × fold × seed, report:
   - clean AUC;
   - FPR on beautified real faces;
   - FNR on beautified fakes;
   - verdict flip rate;
   - UNCERTAIN rate;
   - heatmap IoU.
   
   Give mean ± std over seeds plus bootstrap 95% CIs. Also run the inference-time ablation (+M1, +M2, +calibration, +fusion), and report errors with UNCERTAIN forced to a decision. Write results to JSON and post them to the server's `evalRuns` collection.
6. **`hive_batch.py`**: run Hive on about 500 validation faces in four groups (real, real + filter, fake, fake + filter). Respect the daily budget, resume across days and save raw responses. Report:
   - Hive's FPR and FNR;
   - the agreement rate with our pipeline;
   - who was right when the two disagree.

Pre-registered success criteria, printed in the eval report (report honestly whether each is met):

- flip rate on the held-out family at least 30% lower than H-c;
- FPR on beautified real faces below both H-a and H-c;
- clean AUC within 1 point of H-a.

---

## 6. Node server: API, workers, Hive, reconciler, Groq

### General

- **Auth:** JWT (access + refresh), bcrypt; roles `user` and `admin`.
- **Uploads:** multer; magic-byte check with `file-type`; limits of 10 MB per image and 100 MB per video; compute sha256.
- **Jobs:** BullMQ. Each analysis job:
  1. calls FastAPI **and** Hive in parallel (Hive timeout 15 s);
  2. runs the reconciler;
  3. saves to MongoDB;
  4. queues a Groq explanation job.
- **Progress:** Socket.IO room per job. Stages: queued → detecting faces → scoring n/N frames → verifying → explaining → done.
- **Rate limits:** express-rate-limit backed by Redis. Errors use RFC 7807 problem+json.

### Hive client (`server/src/providers/hive.ts`)

Behind a `Verifier` interface (`verify(filePath) → {status, deepfakeScore, aiGeneratedScore, raw}`), so another provider can be swapped in later.

- Endpoint: `POST https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection`
- Header: `authorization: Bearer ${HIVE_API_KEY}`
- Send the original file as multipart field `media`.
- Read `output[0].classes`: find `class === "deepfake"` for the score, and store `ai_generated` too. For video, iterate `output[]`.
- **Verify the exact response format on the first real call and adapt the parser.** Hive's docs show `classes` items with a value field; confirm whether it is named `value` or `score`. Don't assume.
- Default limit: 100 requests a day. To stay within it:
  - cache by file sha256 in the `externalChecks` collection, so a file is never sent twice;
  - keep a Redis daily budget counter with a reserved demo pool;
  - when the budget is spent, return status `quota`.
- External verification is a per-upload toggle (`externalCheck` field), because uploads go to a third party.

### Reconciler (`server/src/reconcile.ts`)

A pure function, unit-tested for every row below. Hive's verdict is FAKE when `deepfakeScore ≥ HIVE_THRESHOLD` (default 0.9, configurable).

| Ours | Hive | Final verdict | Badge |
|---|---|---|---|
| FAKE | FAKE | FAKE | `verified_by_2` |
| REAL | REAL | REAL | `verified_by_2` |
| FAKE, confident | REAL | FAKE | `second_check_disagrees` |
| REAL, confident | FAKE | REAL | `second_check_disagrees` |
| FAKE or REAL, not confident | disagrees | UNCERTAIN | `checks_disagree` |
| UNCERTAIN | anything | UNCERTAIN | `hive_hint` |
| anything | error, timeout, quota or opted out | ours | `single_check` |

### Groq (`server/src/providers/groq.ts`)

- OpenAI-compatible base URL `https://api.groq.com/openai/v1`.
- Model IDs come from env vars with a fallback list (Groq retires models at short notice):
  - `GROQ_MODEL_EXPLAIN=openai/gpt-oss-120b`
  - `GROQ_MODEL_CHAT=openai/gpt-oss-20b`
- Uses:
  1. explanation as JSON `{summary, why, caveats[]}`; it must cover a disagreement with Hive when one exists;
  2. an "ask about this result" chat grounded only in the result JSON;
  3. PDF report text.
- System prompt rules:
  - use only numbers present in the JSON;
  - never say "authentic" or "proven fake";
  - always state that the check is automated.
- Validate the reply against the schema. On failure or timeout, fall back to a template explanation built from the JSON. Cache results in the `explanations` collection.

### MongoDB collections

| Collection | Key fields |
|---|---|
| `users` | email, passwordHash, role, createdAt |
| `media` | ownerId, kind, storageKey, sha256, sizeBytes, expiresAt (TTL index) |
| `analyses` | mediaId, status, final {verdict, badge, ruleApplied}, ours {verdict, probability, threshold, confident, uncertainReason, detectorScores, spread}, externalCheckId, faces[], frames[], heatmapKeys, modelCardVersion |
| `comparisons` | analysisId, family, strength, clean, filtered, flipped, heatmapIoU |
| `externalChecks` | mediaSha256, provider, status, deepfakeScore, aiGeneratedScore, threshold, verdict, latencyMs, rawResponse |
| `explanations` | analysisId, groqModel, promptVersion, summary, why, caveats[], fallbackUsed |
| `evalRuns` | scheme, fold, seed, metrics, ci95 |

### Public API (`/api/v1`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/auth/register`, `/auth/login`, `/auth/refresh` | Accounts |
| POST | `/analyses` | Upload (multipart, with the `externalCheck` field) → `202 {analysisId, jobId}` |
| GET | `/analyses/:id` | Returns `{final, ours, external, explanation, ml}` |
| GET | `/analyses` | The user's history (paged) |
| POST | `/analyses/:id/compare` | Body `{family, strength}` |
| POST | `/analyses/:id/ask` | Body `{question}` |
| GET | `/analyses/:id/report.pdf` | Reviewer report |
| DELETE | `/analyses/:id` | Delete media + results |
| GET | `/model-card`, `/eval-runs` | Model card and evaluation tables |

---

## 7. React frontend

| Page | Contents |
|---|---|
| **Detect** | Drag-and-drop upload, the external-check toggle, live progress via Socket.IO. Result view: final verdict + verification badge; a ConfidenceGauge with the threshold marked; Hive's score beside ours; heatmap viewer (blend and side-by-side, opacity slider); face gallery; Groq explanation; "Ask about this result" chat; download JSON or PDF |
| **Filter lab** | Choose F1–F4 + strength; original vs filtered shown side by side (verdict, confidence, heatmap); a flipped verdict is highlighted |
| **History** | The user's past analyses |
| **Results** | Leave-one-family-out tables and charts from `evalRuns`, the inference ablation and the Hive comparison |
| **Model card** | Detectors, thresholds and their sources, calibration, licences |

Video results also need a frame timeline in which frames with no face are drawn as gaps. UNCERTAIN has its own colour. Every result page states that the check is automated and is not proof.

---

## 8. Phases. Stop after each one for my review.

| Phase | Build | Done when |
|---|---|---|
| 0 | Plan + repo tree | I say go |
| 1 | Docker Compose skeleton (web, server, ml-service, mongo, redis), `.env.example`, health checks, MOCK_MODE end to end: upload → job → mock verdict → reconciler (with mock Hive) → template explanation → UI | `docker compose up` works; e2e smoke test passes |
| 2 | Real ML: weight download scripts; GenD + SBI wrappers; MTCNN + MediaPipe; F1–F4; pipeline steps 1–10; `/analyze/image`, `/compare`, `/beautify` | Unit tests on sample images; latency printed for CPU (and GPU if available) |
| 3 | Offline scripts: extract → cache → train heads → calibrate → eval LOFO, runnable on a tiny sample dataset folder | Runs end to end on a 20-image fixture; outputs `head.pt` + `model_card.json` |
| 4 | Real Hive + Groq clients, budget counter, cache, reconciler tests (all 7 rules), Groq schema validation + fallback | Tests pass; one real call each, if keys are present |
| 5 | Full frontend: all pages, video path, Socket.IO progress, PDF report | Manual test checklist in the README |
| 6 | Hardening: rate limits, upload security, TTL cleanup, CI (lint, type-check, tests), README with setup, dataset access and licences | CI green |

---

## 9. Quality bar

- **Tests:**
  - pytest for beautify ops (output shape, identity preserved, strength = 0 leaves the image unchanged), pipeline decisions, IoU and calibration math;
  - Jest/Vitest for the reconciler, budget counter and parsers.
- **Typing:** TypeScript strict; Python type hints; ruff + black + mypy; eslint + prettier.
- **Config and logging:** all thresholds and K are config, not code constants. Use structured logs.
- **Licences in the README:** SBI weights are research-only; DeepfakeBench weights are CC BY-NC; CodeFormer is non-commercial; Hive and Groq have their own terms. This is an academic, non-commercial project.
- **When something is unknown** (a weight URL, an exact API field, a preprocessing detail), say so, check the source, or ask me. Never invent it.
