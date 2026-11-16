# Filter-invariant deepfake detection platform

A web app that checks a face image or short video for manipulation and returns **FAKE**,
**REAL** (shown as "No manipulation found") or **UNCERTAIN**. The research goal is a verdict
that stays stable when the face has gone through cosmetic beauty filters.

Every upload is checked by our own pipeline (FastAPI) and, if the user opts in, by Hive's
deepfake API as a second opinion. A reconciler merges the two, and an explanation is written
from the result JSON. The full specification is in [CLAUDE.md](CLAUDE.md).

This is an academic, non-commercial B.Tech project. Every result is an automated check and is
not proof either way.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Compose skeleton, mock mode end to end | Done |
| 2 | Real ML pipeline (GenD, SBI, filters, Grad-CAM) for images | Done |
| 3 | Offline training and evaluation scripts | Not started |
| 4 | Real Hive and Groq clients | Not started |
| 5 | Full frontend, video path, PDF report | Not started |
| 6 | Hardening and CI | Not started |

Mock mode is the default. In mock mode every score is derived from the file's sha256: the
scores are repeatable but say nothing about the image.

Real mode (`MOCK_MODE=false`) runs the real image pipeline in the ML service. Still missing
in real mode: video analysis (Phase 5), the Hive second check and Groq explanations
(Phase 4), and heatmaps in the UI (Phase 5). There is no trained head or calibration yet
(Phase 3), so real-mode verdicts use GenD's own logit with **uncalibrated defaults**
(threshold 0.5, T = 1). Do not read them as results.

## Run it

Requires Docker with Compose v2. No GPU, weights or API keys are needed.

```bash
docker compose up --build
```

Then open http://localhost:8080, create an account and upload an image.

To change ports or secrets, copy `.env.example` to `.env` and edit it. The defaults are for
development only; the server refuses to start with them when `NODE_ENV=production`.

### Smoke test

With the stack running (needs Node 22+ on the host):

```bash
node scripts/smoke_e2e.mjs
```

It registers a user, uploads the fixture image, waits for the verdict and explanation, and
checks repeatability, the external-check toggle, upload rejection, history and delete.

## Services

| Service | Role | Exposed |
|---|---|---|
| `web` | React + Vite + Tailwind, served by nginx, which proxies `/api` and `/socket.io` | `8080` |
| `server` | Express API, BullMQ workers, reconciler, Socket.IO | `4000` |
| `ml-service` | FastAPI pipeline; requires `X-Internal-Secret` | private |
| `mongo`, `redis` | Storage and job queue | private |

Request flow: upload → `media` + `analyses` documents → BullMQ job → ML service and second
check in parallel → reconciler → explanation job → result. Progress is pushed to a Socket.IO
room per analysis, and the UI also polls as a fallback.

## Real mode

1. Install the ML stack and download the weights (about 1.2 GB):

   ```bash
   cd ml-service
   python3.11 -m venv .venv
   .venv/bin/pip install -r requirements.txt -r requirements-ml.txt
   cd ..
   bash scripts/download_weights.sh
   ```

   On Linux, MediaPipe also needs `libegl1` and `libgles2` (`apt-get install libegl1 libgles2`).

2. **SBI weights may need a manual step.** They are on Google Drive, which sometimes refuses
   scripted downloads. If the script says it could not fetch them, open the "c23" link in the
   [SBI README](https://github.com/mapooon/SelfBlendedImages#2-pretrained-model), download
   the file and save it as `ml-service/weights/sbi/FFc23.tar`. Without it the pipeline still
   runs, with GenD only (`scores.sbi` is `null`).

   The file used to validate this project was 141,290,933 bytes with SHA-256
   `2b05c2ba36ccb9e9f4f9e1aae9acd443ae2e6400ce725f56104fdb175d3c3267`. The authors publish no
   checksum, so this only tells you whether you have the same file. Despite its name it is
   a PyTorch checkpoint (a zip container), not a tar. It is loaded with `weights_only=True`.

3. Run with Docker (builds a larger image, about 2.5 GB). **This real-mode image has not been
   built or run yet** (see "Phase 2 validation" below); the direct command after it is the
   verified path.

   ```bash
   MOCK_MODE=false WITH_ML=true docker compose up --build
   ```

   or run the ML service directly:

   ```bash
   cd ml-service
   MOCK_MODE=false UPLOAD_DIR=/path/to/uploads .venv/bin/uvicorn app.main:app --port 8000
   ```

`GET /ready` answers 503 while models load (about 10 s on CPU) and names the cause if
loading fails. `python scripts/bench.py` (from `ml-service/`) prints latency.

### What the pipeline does per face

1. MTCNN detects faces. Undetected faces are excluded; an image with no face is UNCERTAIN.
2. Two crops per face, each built the way its detector's repository does it: a 224 px
   five-point aligned crop for GenD and a 380 px box crop for SBI.
3. MediaPipe landmarks, then K beautified copies (F1, F2, F3 and a random two-filter mix) at
   random strengths. The random choices are seeded from the file, so results repeat.
4. GenD scores the clean crop and the copies in one batch; SBI scores the clean crop.
5. Mean logit, spread, the `beautifiedLikelihood` cue, fusion, calibration and the decision.
6. Grad-CAM for both detectors, and the IoU of the top 20% heatmap regions between the clean
   face and the copy whose score moved most (warped back first if F3 was applied).

### Filters

| Family | What it does | Where it is used |
|---|---|---|
| F1 | Bilateral, guided or Gaussian smoothing inside the skin mask | Inference copies, training |
| F2 | Skin brightening, lip tint, blush, warmth and saturation | Inference copies, training |
| F3 | Jaw slimming, eye enlargement, nose narrowing (piecewise-affine warp, capped) | Inference copies, training |
| F4 | GFPGAN restoration | Test only, through `/compare` and `/beautify` |

F4 is optional and **has not been run in this project yet**. It needs `pip install gfpgan`
and `GFPGANv1.4.pth` saved as `ml-service/weights/gfpgan/GFPGANv1.4.pth`. GFPGAN 1.3.8 depends
on `basicsr`, which is known to break on recent torchvision, so expect to patch or pin it.
Without it, F4 requests return 501 and everything else works.

### CPU latency

Measured with `scripts/bench.py` on a 4-core Intel Xeon 2.1 GHz VM, PyTorch 2.2.2 CPU, 4
threads, one face, median of 5 runs. No GPU was available, so there is no GPU figure. K only
applies to GenD; SBI always scores the clean crop once.

Detectors alone:

| Step | Time |
|---|---|
| GenD forward, 1 crop (K = 0) | 461 ms |
| GenD forward, 5 crops (K = 4) | 2336 ms |
| GenD Grad-CAM, 1 crop | 1274 ms |
| GenD Grad-CAM, 2 crops (clean + one copy, for the IoU) | 2775 ms |
| SBI forward, 1 crop | 112 ms |
| SBI Grad-CAM, 1 crop | 2356 ms |

Full pipeline (ms):

| Detectors | K | Heatmaps | Total | Detect | Beautify | Score | Explain |
|---|---|---|---|---|---|---|---|
| GenD only | 0 | off | 589 | 48 | 16 | 523 | 0 |
| GenD only | 0 | on | 2104 | 54 | 20 | 471 | 1557 |
| GenD only | 4 | off | 2463 | 42 | 276 | 2145 | 0 |
| GenD only | 4 | on | 5055 | 39 | 303 | 2029 | 2683 |
| GenD + SBI | 0 | off | 626 | 38 | 13 | 573 | 0 |
| GenD + SBI | 0 | on | 4139 | 52 | 15 | 571 | 3500 |
| GenD + SBI | 4 | off | 2567 | 51 | 293 | 2223 | 0 |
| GenD + SBI | 4 | on | 7408 | 49 | 285 | 2241 | 4832 |

### Phase 2 validation

Run on a 4-core CPU VM with the real `yermandy/GenD_CLIP_L_14` and SBI FF++ c23 weights
loaded, the ML service started directly (not in Docker).

| Check | Result |
|---|---|
| ML service pytest | 59 passed, 0 skipped |
| Same suite without the ML stack installed | 17 passed, 3 files skipped (before the last 3 SBI tests were added) |
| black, ruff, mypy | Clean |
| Server tests (16, of which 13 cover the reconciler) and type-check | Passed |
| GenD wrapper against GenD's original code, same 4 images | Identical logits |
| SBI checkpoint through the `weights_only=True` loader | Strict key match, finite scores |
| Upload to verdict to explanation through the Node API, 6 images | All completed, both heatmaps written |

Known limitations:

- **Real-mode Docker image: pending.** `WITH_ML=true` has never been built successfully. The
  system package names in `ml-service/Dockerfile` are untested.
- **F4 (GFPGAN): pending.** The wrapper was checked against GFPGAN 1.3.8's source but never run.
- **Uncalibrated.** No trained head or model card exists, so verdicts use GenD's own logit at
  threshold 0.5, and SBI only acts through the detectors-disagree rule.
- **Images only.** Real video analysis returns 501 until Phase 5; heatmaps are written to the
  upload volume but not shown in the UI yet.
- **GenD is sensitive to the filters.** On the three real faces tried, smoothing and makeup
  copies moved GenD's logit far toward "fake"; the pipeline returned UNCERTAIN for them. That
  is a handful of images, not an evaluation.
- **SBI Grad-CAM is slow** on CPU (about 2.4 s against a 0.1 s forward pass).
- **No GPU figures.** fp16 on GPU is implemented but has not been run.

## Development

```bash
# ML service (Python 3.11)
cd ml-service
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest && .venv/bin/ruff check . && .venv/bin/black --check . && .venv/bin/mypy app
# Tests that need the ML stack or the weights are skipped until both are installed.

# Server (Node 22)
cd server
npm ci && npm test && npm run typecheck && npm run lint

# Web
cd web
npm ci && npm run typecheck && npm run lint && npm run build
```

## Decision rules

The ML service returns UNCERTAIN when the probability is within `uncertain_margin` of the
threshold, the spread across filtered copies exceeds `spread_limit`, the copies disagree, or
the two detectors strongly disagree. All values load from `weights/model_card.json`; without
that file the defaults are identity fusion, T = 1 and threshold 0.5, and they are
uncalibrated.

The reconciler (`server/src/reconcile.ts`) never flips our verdict. The second check can
confirm it, or downgrade a non-confident verdict to UNCERTAIN.

## Licences and terms

This is an academic, non-commercial project.

| Component | Licence or terms |
|---|---|
| GenD code and `yermandy/GenD_CLIP_L_14` weights | MIT |
| CLIP ViT-L/14 (the backbone inside GenD) | MIT (OpenAI) |
| SBI code and weights | Research use only; commercial use needs a licence from the authors |
| MediaPipe Face Landmarker | Apache-2.0 |
| facenet-pytorch (MTCNN) | MIT |
| GFPGAN (optional F4) | Apache-2.0; check the terms of its weights before use |
| CodeFormer (not used) | Non-commercial (S-Lab License) |
| DeepfakeBench weights (not used) | CC BY-NC |
| Hive, Groq | Their own terms of service |

Tests use scikit-image's bundled astronaut photo (NASA, public domain). No dataset images are
stored in this repository.
