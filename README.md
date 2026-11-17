# Filter-invariant deepfake detection platform

A web app that checks a face image or short video for manipulation and returns **FAKE**,
**REAL** (always shown as "No manipulation found") or **UNCERTAIN**. The research goal is a
verdict that stays stable when the face has gone through cosmetic beauty filters (skin
smoothing, makeup, reshaping, generative restoration).

Every upload is checked by our own pipeline (FastAPI, frozen pretrained detectors) and, only if
the user opts in, by Hive's deepfake API as a second opinion. A reconciler merges the two, and
an explanation is written from the result data. The full specification is in
[CLAUDE.md](CLAUDE.md).

> **Automated deepfake detection is probabilistic.** A result is one piece of evidence, never
> proof that media was or was not manipulated. Detectors miss manipulations and flag unedited
> media, especially after compression, unusual lighting or new editing tools.

This is an academic, non-commercial B.Tech project.

## Contents

[Status](#status) · [Architecture](#architecture) · [Quick start](#quick-start-no-keys-no-gpu) ·
[Configuration](#configuration) · [Running services separately](#running-each-service-on-its-own) ·
[API](#api) · [Image and video workflows](#image-and-video-workflows) · [Real mode](#real-mode) ·
[Training and evaluation](#phase-3-offline-training) · [External APIs](#phase-4-external-apis) ·
[Security and privacy](#security-and-privacy) · [Testing](#testing) · [CI](#continuous-integration) ·
[Limitations](#known-limitations) · [Troubleshooting](#troubleshooting) · [Demo](#demo-script) ·
[Licences](#licences-and-terms)

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Compose skeleton, mock mode end to end | Done |
| 2 | Real ML pipeline (GenD, SBI, filters, Grad-CAM) for images | Done, validated with the real weights |
| 3 | Offline training, calibration and evaluation scripts | Done on the fixture; real data awaits Celeb-DF access |
| 4 | Hive and Groq clients, budgets, explanation validation | Done. Hive verified live once (M7.1, synthetic image). Groq never called live (M7.2 skipped by decision) |
| 5 | Full frontend, real video, live progress, PDF, Ask, Filter Lab | Done, verified end to end in mock mode (Docker Compose) and real mode (real weights) |
| 6 | Hardening, reliability, CI, documentation | Done; CI configured (see [CI](#continuous-integration) for its run status) |

Mock mode is the default: every score is derived from the file's SHA-256, repeatable but
meaningless. Real mode (`MOCK_MODE=false`) runs the real detectors. No head has been trained on
real data yet, so real-mode verdicts use GenD's own logit with **uncalibrated defaults**
(threshold 0.5, T = 1). Do not read them as accuracy results.

## Architecture

```
 Browser (React, Vite, Tailwind)
   │  HTTPS: /api/v1/*  (JWT)          WebSocket: /socket.io (JWT handshake, per-analysis rooms)
   ▼
 web: nginx ── serves the app, proxies /api and /socket.io, security headers
   │
   ▼
 server: Node + Express (the only public backend)
   ├─ REST API: auth, uploads (magic bytes, size limits), results, heatmaps, Ask, PDF, compare
   ├─ Socket.IO: progress events (stage + measured done/total) to the owner only
   ├─ BullMQ workers ──► ML service and, with consent, the second check, in parallel
   │                  └─► reconciler ──► explanation (template; Groq only if enabled)
   ├─ maintenance: stale-job reconciliation, expired-upload sweep
   ├─ MongoDB (users, media with TTL, analyses, explanations, externalChecks,
   │           comparisons, evalRuns)
   └─ Redis (job queue, API budgets, rate limits, locks)
         │ private network, X-Internal-Secret
         ▼
 ml-service: FastAPI + PyTorch
   ├─ /analyze/image  MTCNN → crops → MediaPipe → K beautified copies → GenD + SBI
   │                  → linear head → fusion, calibration → decision → Grad-CAM
   ├─ /analyze/video  OpenCV decode → even frame sampling → image pipeline per frame
   │                  → top-k mean → decision → Grad-CAM on the extreme frames
   ├─ /compare, /beautify  (Filter Lab), /progress/{id}, /model-info, /ready
   └─ shared upload volume: uploads and derived images (heatmaps, crops)

 External, optional: Hive (second check, consent + budget), Groq (wording only, off by default)
```

**Tech stack:** React 18, Vite, TypeScript, Tailwind, React Router · Node 22, Express 5,
TypeScript, Socket.IO, BullMQ, Mongoose 8, ioredis, pdfkit, zod · Python 3.11, FastAPI,
PyTorch 2.2 (CPU), facenet-pytorch (MTCNN), MediaPipe, OpenCV, scikit-image, pytorch-grad-cam ·
MongoDB 7, Redis 7 · Docker Compose · GitHub Actions.

### Repository layout

```
docker-compose.yml  .env.example  .github/workflows/ci.yml
web/          React app: src/pages (Detect, History, Filter Lab, Results, Model card,
              analysis page), src/components (result view, heatmaps, timeline, Ask), src/test
server/       src/routes (auth, analyses, meta), src/data (repositories: MongoDB),
              src/queue (jobs), src/ask, src/report (PDF), src/explain, src/providers
              (ML, Hive, Groq), src/sockets.ts, src/maintenance.ts, tests/, scripts/smoke.ts
ml-service/   app/ (main, pipeline, video, faces, detectors, beautify, head, explain, mock,
              progress), offline/ (Phase 3 training and evaluation), tests/, weights/ (gitignored)
scripts/      download_weights.sh, Phase 1 smoke test
```

## Quick start (no keys, no GPU)

Requires Docker with Compose v2.

```bash
docker compose up --build
```

Open http://localhost:8080, create an account and upload an image or a short video. This runs
in **mock mode**: no model weights, no API keys, no external network calls and no cost. The
Hive second check returns placeholder scores (`hive-mock`) and every explanation and Ask answer
comes from the built-in template. Groq is `off` and Hive is `mock` by default in Compose, so no
paid service is ever called unless you opt in explicitly.

To change ports or secrets, copy `.env.example` to `.env`. The development secrets are
refused when `NODE_ENV=production`.

## Configuration

All settings are environment variables with safe defaults; `.env.example` lists them with
explanations. The most important:

| Variable | Default | Meaning |
|---|---|---|
| `MOCK_MODE` | `true` | `false` runs the real models (needs the weights; see Real mode) |
| `HIVE_MODE`, `GROQ_MODE` | `mock`, `off` (Compose) | `mock`, `real` or `off`; real needs a key in `.env` |
| `K`, `VIDEO_K` | `4`, `0` | Beautified copies per image face, and per video frame |
| `MAX_FRAMES`, `VIDEO_TOP_K` | `32`, `5` | Frames sampled per video; frames averaged for the verdict |
| `MAX_IMAGE_MB`, `MAX_VIDEO_MB` | `10`, `100` | Upload limits (checked again by the ML service) |
| `MEDIA_TTL_HOURS` | `24` | Uploads expire after this; results stay until deleted |
| `CORS_ORIGINS` | empty | Same origin only unless origins are listed |
| `RATE_LIMIT_*` | 20/15 min auth, 30/h uploads, 60/h Ask, 20/h compare | Per user (per address for auth) |
| `JOB_ATTEMPTS`, `STALE_JOB_MINUTES` | `2`, `30` | Retries for transient failures; when a job counts as stuck |
| `JWT_*_SECRET`, `INTERNAL_SECRET` | dev values | Must be changed for production |

## Running each service on its own

```bash
# Infrastructure
docker run -d --name mongo -p 127.0.0.1:27017:27017 mongo:7
docker run -d --name redis -p 127.0.0.1:6379:6379 redis:7-alpine

# ML service (mock mode needs only requirements.txt)
cd ml-service && python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
UPLOAD_DIR=/tmp/df-uploads .venv/bin/uvicorn app.main:app --port 8000

# Server (shares UPLOAD_DIR with the ML service)
cd server && npm ci && npm run build
UPLOAD_DIR=/tmp/df-uploads node dist/index.js        # or: npm run dev

# Web (proxies /api and /socket.io to localhost:4000)
cd web && npm ci && npm run dev                     # http://localhost:5173
```

## API

All routes are under `/api/v1`, need `Authorization: Bearer <access token>` unless marked, and
answer errors as RFC 7807 `application/problem+json`. Another user's analysis is always 404.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and mock mode (no auth) |
| POST | `/auth/register`, `/auth/login`, `/auth/refresh` | Accounts (no auth; rate-limited) |
| POST | `/analyses` | Multipart `file` + `externalCheck` (`true` sends the file to Hive) → `202 {analysisId, jobId}`; the same file while still running → the running analysis (`duplicate: true`) |
| GET | `/analyses?page&limit` | Your history, newest first |
| GET | `/analyses/:id` | `{status, stage, progress, final, ours, external, explanation, ml}` |
| GET | `/analyses/:id/heatmaps/:name` | A heatmap, face crop or filtered image of this analysis (PNG) |
| POST | `/analyses/:id/ask` | `{question}` → `{answer, source, intent, fallbackReason}` |
| GET | `/analyses/:id/report.pdf` | Reviewer PDF (finished analyses only) |
| POST | `/analyses/:id/compare` | `{family: F1-F4, strength: 0-1}` → clean vs filtered verdicts (real pipeline only; 501 in mock mode) |
| GET | `/analyses/:id/comparisons` | Earlier Filter Lab comparisons |
| DELETE | `/analyses/:id` | Deletes the upload, derived images and all results |
| GET | `/model-card` | Detectors, calibration values and their sources, rules, licences |
| GET, POST | `/eval-runs` | Latest imported evaluation report; POST (admin) imports `eval_lofo.py` JSON |

**Socket.IO** (path `/socket.io`): connect with `auth: {token}`; refused with
`connect_error` `unauthorized` and `data.reason` `missing`, `invalid` or `expired`.
`emit('subscribe', analysisId, ack)` joins only your own analysis and immediately sends a
`progress` snapshot of the current state. Events: `progress {analysisId, stage, status,
progress?: {done, total, unit}, detail?, snapshot?}` with stages `queued`, `detecting_faces`,
`extracting_frames` (video), `scoring`, `aggregating`, `verifying`, `explaining`, `done` or
`failed`. `emit('unsubscribe', analysisId)` leaves.

## Image and video workflows

**Image:** upload (magic bytes and size checked, stored under a random name, SHA-256 kept
server-side only) → job queued → ML service detects faces, scores the clean crop and K
beautified copies, fuses and calibrates, decides, and makes heatmaps for the deciding face →
the optional second check runs in parallel → reconciler → explanation → done. The browser sees
each stage live (faces scored n of N), with polling as a fallback.

**Video:** the ML service decodes the file with OpenCV (MP4, WebM, MOV), counts the frames
that really decode, samples up to `MAX_FRAMES` evenly (the middle frame of equal segments)
and keeps each frame's index and timestamp (`index / fps`). Every sampled frame goes through
the image pipeline. Frames without a face are kept in the timeline as gaps and **excluded**
from the verdict, never counted as real. The video logit is the mean of the top
`VIDEO_TOP_K` frame logits, decided by the same rules as an image. Only the most and least
suspicious frames get heatmaps. Empty, corrupt, unsupported and over-long videos are refused
with a clear message. The UI shows a frame timeline with the threshold, gaps, the extreme
frames and their heatmaps.

**Ask** answers questions about one finished analysis from a whitelisted set of facts
(verdict, scores, thresholds, second check, frame summary): deterministic templates, no LLM.
Unrelated questions are refused. Every answer is checked so that it contains no number that
is not in the facts and no "authentic" or "proven". With `GROQ_MODE=real`, Groq may word the
answer from the same facts and the same checks apply; otherwise, and on any failure, the
template is used.

**PDF reports** contain the analysis ID, timestamps, verdict and badge, scores and thresholds,
pipeline checks, second check, rule applied, model card version, explanation and the video
frame table, with "automated finding, not proof" on every page. They never contain email
addresses, file names, paths, hashes, storage keys, keys or raw provider responses.

**Filter Lab** applies one filter (F1 to F4, chosen strength) to the face of an analysed image
and shows both verdicts, both heatmaps and their overlap; a changed verdict is highlighted.

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

3. Run with Docker (builds a larger image, about 2.5 GB). **The real-mode image
   (`WITH_ML=true`) has still not been built in this project**; the verified real-mode path is
   the ML service run directly, as below (Phase 5's real end-to-end run used it).

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


Video and end-to-end figures (Phase 5, same VM, `MOCK_MODE=false`, real GenD + SBI weights,
single runs, frames containing one face, `VIDEO_K=0`, heatmaps for the most and least
suspicious frames):

| Workload | Total | Scoring | Heatmaps |
|---|---|---|---|
| Image, K = 4, heatmaps on | 10.0 s | 2.9 s | 6.7 s |
| Image, K = 0, heatmaps off | 0.8 s | | |
| Video, 8 frames sampled of 8 | 30.9 s | 5.3 s | 25.0 s |
| Video, 32 frames sampled of 32 | 33.0 s | 22.2 s | 9.3 s |
| Video, 32 frames sampled of 250 | 32.5 s | | |

Model loading took 6.3 s; peak memory was about 4.3 GB. Scoring costs about 0.7 s per frame;
decoding is negligible (under 0.1 s), so a long video costs the same as a 32-frame one. The
heatmap time varied between runs (25 s and 9 s for one or two frames), which these single
runs cannot explain; treat it as noisy. No optimisation was made that could change results:
the measurements show Grad-CAM and per-frame scoring dominate, which is why videos default to
`VIDEO_K=0` and explain only two frames.

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
- *(Resolved in Phase 5: real video analysis exists and heatmaps are shown in the UI.)*
- **GenD is sensitive to the filters.** On the three real faces tried, smoothing and makeup
  copies moved GenD's logit far toward "fake"; the pipeline returned UNCERTAIN for them. That
  is a handful of images, not an evaluation.
- **SBI Grad-CAM is slow** on CPU (about 2.4 s against a 0.1 s forward pass).
- **No GPU figures.** fp16 on GPU is implemented but has not been run.

## Phase 3: offline training

The scripts in `ml-service/offline/` train the invariance head, calibrate the service and
evaluate the six head schemes. They are CPU-only and resumable. Full documentation is in
[ml-service/offline/README.md](ml-service/offline/README.md).

**What has been validated, and what has not:**

| Kind of validation | Status | What it shows |
|---|---|---|
| Fixture validation (plumbing) | **Done.** `python -m offline.run_fixture` runs every stage on a generated 20-face fixture in about 5 minutes on CPU | The code runs end to end, files flow between stages, leakage guards hold, the service loads the outputs |
| Real-data accuracy validation | **Not done.** No Celeb-DF v2 access yet, and real-data split assignment is blocked until the official list's identity graph has been reviewed | Nothing yet: no accuracy, AUC, error rate or success criterion has been measured on real faces |

Fixture numbers (AUC, error rates, criteria marked MET or NOT MET) come from one photo with
arbitrary labels. They are not accuracy results and must not be reported as such.

- **Protocol label for any future result:** internal identity-disjoint split of the Celeb-DF
  v2 official test list, not comparable with published benchmark results.
- **Showing results in the app:** an admin imports the evaluation JSON, and the Results page
  shows it (fixture reports are labelled as such):

  ```bash
  curl -X POST http://localhost:8080/api/v1/eval-runs -H "authorization: Bearer $ADMIN_TOKEN" \
       -H 'content-type: application/json' --data @ml-service/artifacts/eval/lofo.json
  ```
- **Service safeguards:** the service refuses a `head.pt` that doesn't match
  `model_card.json`, and refuses fixture or insufficiently validated cards unless
  `ALLOW_UNVALIDATED_MODEL_CARD=true` (tests only).

## Phase 4: external APIs

Each upload can get two external services: **Hive** as an independent second check, and
**Groq** for a plain-English explanation. Neither is needed to run the project, and neither
can change our verdict on its own terms: Hive can only confirm it or downgrade a
non-confident verdict to UNCERTAIN (the reconciler, unchanged since Phase 1), and Groq only
writes text. Full reference: [server/README.md](server/README.md).

> **Validation status.** Both clients are built from the official API documentation
> (checked 16 Nov 2026) and tested against fake responses, a real Redis and the full mock
> stack. **Hive was verified with exactly one live request** (M7.1, a synthetic abstract
> image): the response matched the documented v3 format (`output[].classes[]` with
> `class`/`value`) and the parser. **Groq has never been called live**: that test (M7.2) was
> skipped by decision, so the Groq path is verified against fake responses only.

### Modes

Each service has its own switch: `HIVE_MODE` and `GROQ_MODE`, each `mock`, `real` or `off`.
Docker Compose and `.env.example` set `HIVE_MODE=mock` and `GROQ_MODE=off`, so turning
`MOCK_MODE` off for the real models never starts paid calls by itself. (Outside Compose, an
empty value follows `MOCK_MODE`: `mock` when true, `real` when false.)

| Mode | Hive second check | Groq explanation |
|---|---|---|
| `mock` | Placeholder scores from the file hash; no network, no cost | Built-in template |
| `real` | Hive v3 API, budgeted; needs `HIVE_API_KEY` | Groq API, validated; needs `GROQ_API_KEY` |
| `off` | Not run; recorded as `disabled` | Built-in template |

Real mode without a key does not fail: the check is recorded as `not_configured` (the result
is then a single check) and the explanation uses the template.

### Privacy: consent before anything is sent to Hive

- A file is sent to Hive **only** when the uploader ticks the per-upload consent box
  (`externalCheck=true`). It is off by default.
- Without consent nothing is sent, cached or recorded about Hive for that upload.
- Only the file is sent, under the neutral name `upload`: never the original file name, the
  user's identity or the analysis.
- Groq never receives the image. It receives only a fixed list of result fields (verdicts,
  scores, thresholds, badge), never file bytes, paths, file names or hashes.

### Budgets and caching

Neither API is treated as free. Every request that is about to be sent takes one unit from a
daily budget in Redis, atomically, so concurrent uploads cannot overspend.

| Setting | Default | Meaning |
|---|---|---|
| `HIVE_DAILY_LIMIT` | 100 | Hive requests per UTC day in total (Hive v3's documented default limit) |
| `HIVE_DEMO_RESERVE` | 20 | Part of the daily limit only users with the `admin` role may use |
| `HIVE_USER_DAILY_LIMIT` | 10 | Hive requests per regular user per day |
| `HIVE_RETRIES` | 0 | Each retry is a separately billed request |
| `GROQ_DAILY_LIMIT` | 900 | Groq requests per UTC day |

- **Never refunded.** A unit is counted when a request is sent and kept even if it fails,
  because neither provider documents billing for failed requests.
- **Provider limit hit.** A Hive 429 response stops all Hive requests until the next UTC day.
- **Cache hits are free.** A successful Hive result is reused for any later upload of the same
  file (matched by SHA-256), and a Groq explanation is reused for identical result facts.
  Failures are not cached, so a later upload tries again.
- **Duplicate uploads.** Two uploads of the same file at the same time trigger one Hive
  request; the second waits for its result.

### Fallbacks

Nothing external can fail an analysis. A Hive problem (no consent, budget, timeout, auth,
rate limit, unexpected response) gives a single-check result with the reason stored and
shown in the API. Any Groq problem, including a response that fails validation, gives the
template explanation with the reason stored. Groq responses are rejected as a whole if they
break the schema, use the words "authentic" or "proven", omit that the check is automated,
contradict the verdict, skip a disagreement, hide mock mode, contain a link, or contain any
number that is not in the result.

### Enabling the real APIs

1. Put the keys in `.env` (never commit it): `HIVE_API_KEY` (a Hive V3 key from "Service API
   Keys") and `GROQ_API_KEY`.
2. Set `HIVE_MODE=real` and/or `GROQ_MODE=real` explicitly (`MOCK_MODE=false` alone does not
   enable them under Docker Compose).
3. Restart: `docker compose up --build`. The server log shows the active modes and warns,
   without printing any key, if a real mode has no key.
4. Upload with the consent box ticked to use Hive. The analysis response shows the second
   check's status and, if unavailable, why (`external.reason`).

## Security and privacy

- **Accounts:** bcrypt password hashes; short-lived JWT access tokens (15 min) and refresh
  tokens (7 days), HS256 only; expired tokens get a clear 401 (`Access token expired`) and the
  web app refreshes them, including for the live-progress socket.
- **Ownership everywhere:** every analysis, heatmap, report, question, comparison and socket
  room is checked against the signed-in user; another user's data is answered as 404.
- **Uploads:** type decided from the file's bytes, never its name or MIME header; per-kind
  size limits (checked twice: server and ML service); random storage names; the ML service only
  opens files inside the upload directory. Rejected uploads are deleted at once.
- **Paths:** derived images are served only for strict `<32 hex>.png` names that belong to the
  analysis, read from the derived directory and verified as PNG. No user input becomes a path.
- **Headers and CORS:** API: `nosniff`, `DENY` framing, no referrer, `default-src 'none'` CSP,
  `no-store`; web: a strict CSP. CORS is same-origin unless `CORS_ORIGINS` lists origins.
- **Rate limits** (Redis, per user): login/registration, uploads, Ask, Filter Lab. If Redis is
  down the limiter fails open and logs it, rather than taking the API offline.
- **Secrets:** only in `.env` (gitignored). Production refuses the development secrets. Keys are
  scrubbed from logs and stored errors; request logs record method and path only (no tokens or
  query strings); internal errors are logged but shown to users only as a generic message.
- **What leaves the server:** with consent only, the upload goes to Hive under the neutral
  name `upload`. Groq (if ever enabled) receives a whitelisted result document only: never
  media, file names, paths, hashes, emails or provider responses. PDFs and Ask answers use the
  same whitelist.
- **Retention:** uploads expire after `MEDIA_TTL_HOURS` (the MongoDB TTL index removes the
  record, and the server deletes the file). Results stay until the user deletes the analysis,
  which removes the upload, heatmaps, comparisons, explanation and an unshared second-check
  record.

## Reliability

- Jobs retry transient failures (ML service restarting) up to `JOB_ATTEMPTS` times with
  backoff; invalid media fails at once with a clear reason. Re-running a job is idempotent.
- Every `CLEANUP_INTERVAL_MINUTES`, analyses stuck in `queued`/`processing` for
  `STALE_JOB_MINUTES` without a live job are requeued, sent to the explanation step, or marked
  failed with a clear message, so nothing stays stuck after a crash or restart.
- Uploading a file that is still being analysed returns the running analysis.
- The ML service caps frames, decoded frames (`MAX_VIDEO_DECODE_FRAMES`) and sizes, and runs
  one inference at a time; progress entries expire.

## Testing

| Suite | Command | Count |
|---|---|---|
| Server: unit, routes (in-memory repositories), sockets, PDF, Ask, jobs, Redis and **MongoDB** integration | `cd server && npm test` | 220 tests |
| Web: components and pages (Vitest, Testing Library) | `cd web && npm test` | 23 tests |
| ML: filters, pipeline, video, API, offline scripts, **real models** | `cd ml-service && .venv/bin/pytest -rs` | 192 tests |
| End to end, mock mode, Docker Compose | `docker compose up -d --build --wait` then `cd server && npx tsx scripts/smoke.ts` | 24 checks |
| End to end, real models (ML service run directly) | `SMOKE_EXPECT=real SMOKE_BASE=http://127.0.0.1:4000 SMOKE_IMAGE=<face image> SMOKE_VIDEO=<face video> SMOKE_VIDEO_FRAMES=<n> npx tsx scripts/smoke.ts` | 26 checks |
| Phase 1 smoke test (backwards compatibility) | `node scripts/smoke_e2e.mjs` | 10 checks |

The server's integration tests need Redis on `localhost:6379` and MongoDB on
`localhost:27017` (or `TEST_REDIS_URL`, `TEST_MONGO_URL`) and **fail** without them; set
`SKIP_REDIS_TESTS=1` / `SKIP_MONGO_TESTS=1` only to skip them on purpose. The HTTP route
tests use in-memory repositories and are not database tests; the MongoDB integration tests
cover the production repositories. ML tests that need the model weights are skipped, with the
reason shown by `-rs`, when the weights are absent.

Static checks: `npm run typecheck`, `npm run lint`, Prettier (`server`, `web`), and `ruff`,
`black --check`, `mypy app offline` (`ml-service`).

## Continuous integration

`.github/workflows/ci.yml` runs on pushes to `main` and `phase-*` branches and on pull
requests: **server** (type-check, lint, format, all tests with Redis and MongoDB service
containers, build), **web** (type-check, lint, format, tests, build), **ml** (ruff, black,
mypy, pytest) and **compose-e2e** (`docker compose config`, `up --build --wait`, the mock smoke
test). No job needs Hive or Groq credentials. Model weights are not in the repository: the
ML job downloads the public ones (GenD, CLIP config, MediaPipe; cached between runs) and so
runs the real-model tests too. The SBI checkpoint comes from Google Drive, which often refuses
scripted downloads; then only the SBI-specific tests are skipped, each with its reason. With
all weights present (locally), the ML suite runs 192 tests with none skipped.

## Known limitations

- **Uncalibrated real mode.** No head has been trained on real data (Celeb-DF v2 access is
  pending), so real verdicts use GenD's logit at threshold 0.5 and say nothing about accuracy.
  The Results page shows only imported evaluations; none exist on real data.
- **Groq never verified live**, and Hive only with one synthetic image (see above).
- **CPU speed.** About 10 s per image with K = 4 and heatmaps, about 30 s per video; one
  inference at a time, so a long queue waits. No GPU was available to measure.
- **Real-mode Docker image** (`WITH_ML=true`) has not been built in this project; real mode
  was verified with the ML service run directly.
- **F4 (GFPGAN)** is optional and has never been installed or run; F4 requests answer 501.
- **Mock mode** decodes videos for real frames and timestamps but its scores are placeholders,
  and it has no faces, heatmaps or Filter Lab.
- **Ask** understands questions by keywords: unusual phrasings may be refused or matched to a
  nearby topic; it never answers from anything but the result.
- Orphaned derived images from failed jobs are not swept (they are small and owned by no one).
- **Dependency advisory accepted:** zod 3 has a high-severity advisory (memory exhaustion when
  validating oversized constrained collections; fixed only in zod 4.6.6+). It is mitigated
  here because every user request body is capped at 100 KB (the admin-only evaluation import
  at 5 MB) and provider responses at 2 MB before validation. Moving to zod 4 changes the API
  across the Phase 1-4 code and is left as a follow-up. `file-type` (ASF infinite loop) and
  `react-router` (open redirect) advisories were fixed by upgrading. The remaining `npm audit`
  findings are in test tooling (Vitest), not in anything that runs in production.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `docker compose up` waits on `ml-service` | Real mode loads models first (`start_period` 60 s); see `docker compose logs ml-service` |
| `GET /ready` answers 503 "Models failed to load" | Weights missing: run `scripts/download_weights.sh`, place the SBI file by hand (Real mode, step 2) |
| Server tests fail "Redis is not reachable" / "MongoDB is not reachable" | Start them (see Running each service), or skip on purpose with `SKIP_REDIS_TESTS=1` / `SKIP_MONGO_TESTS=1` |
| Filter Lab says it needs the real pipeline | Expected in mock mode; set `MOCK_MODE=false` with the weights |
| "Live updates unavailable" in the UI | The socket could not connect (proxy or network); progress is still polled every 2 s |
| An analysis fails with "not a supported video" or "could not be decoded" | The file is not an MP4/WebM/MOV OpenCV can read, is empty or corrupt |
| 429 Too Many Requests | A rate limit; wait for `Retry-After` seconds or raise the `RATE_LIMIT_*` values |
| The server refuses to start in production | Set real `JWT_ACCESS_SECRET`, `JWT_REFRESH_SECRET` and `INTERNAL_SECRET` in `.env` |

## Demo script

1. `docker compose up --build`, open http://localhost:8080, register.
2. **Detect:** upload a portrait. Watch the stages and progress, then the verdict, both
   gauges, the pipeline checks and the explanation. Ask "Why was this flagged?" and
   "What is the capital of France?" (refused). Download the PDF.
3. Upload a short MP4: the frame timeline shows timestamps, gaps for frames without a face and
   the most and least suspicious frames.
4. **History:** open, page through and delete analyses.
5. **Model card** and **Results:** decision values and their sources; evaluation results once
   imported.
6. For heatmaps and the **Filter Lab**, run real mode (see Real mode) and repeat step 2:
   switch heatmaps between GenD and SBI, blend with the opacity slider, then try F1 at 60 %
   in the Filter Lab.
7. Tick "Get a second opinion from Hive" to see the reconciler and badge (mock scores unless
   `HIVE_MODE=real` with a key).

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
| Hive, Groq | Their own terms of service and pricing; uploads go to Hive only with consent |

Tests use scikit-image's bundled astronaut photo (NASA, public domain). No dataset images are
stored in this repository.
