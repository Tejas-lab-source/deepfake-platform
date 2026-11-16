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
| 2 | Real ML pipeline (GenD, SBI, filters, Grad-CAM) | Not started |
| 3 | Offline training and evaluation scripts | Not started |
| 4 | Real Hive and Groq clients | Not started |
| 5 | Full frontend, video path, PDF report | Not started |
| 6 | Hardening and CI | Not started |

**Only mock mode works today.** In mock mode every score is derived from the file's sha256.
The scores are repeatable but say nothing about the image. With `MOCK_MODE=false` the ML
service reports "not ready" until Phase 2, the second check is unavailable until Phase 4, and
explanations always come from a fixed template until Phase 4.

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

## Development

```bash
# ML service (Python 3.11)
cd ml-service
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest && .venv/bin/ruff check . && .venv/bin/black --check . && .venv/bin/mypy app

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

Details for model weights and datasets are added in the phases that introduce them. In
short: SBI weights are research-only, DeepfakeBench weights are CC BY-NC, CodeFormer is
non-commercial, and Hive and Groq have their own terms of service.
