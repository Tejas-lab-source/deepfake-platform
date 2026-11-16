# Server

The Node.js + Express + TypeScript API, the BullMQ workers, the reconciler, and the clients
for the two external services: Hive (second check) and Groq (explanations).

Run it through Docker Compose from the repository root (see the main README). For local
development: `npm ci`, then `npm run dev`, with MongoDB, Redis and the ML service running.

## How an analysis flows

```
POST /api/v1/analyses  (multipart: file, externalCheck=true|false)
  -> magic-byte check, sha256, media + analysis documents, analysis job
analysis worker
  -> ML service ─────────────────────────────────────────────┐
  -> second check (only with consent), in parallel:          │
       cached ok result for this sha256? -> reuse (free)     │
       same file already in flight?      -> wait for it      │
       reserve budget -> Hive v3 -> validate -> store        │
  -> reconcile(ours, second check)  (unchanged since Phase 1) <┘
  -> analysis saved, explanation job queued
explanation worker
  -> whitelisted result JSON -> cache? -> budget -> Groq -> validate
  -> any failure: built-in template (reason recorded)
```

## Modules

| File | Purpose |
|---|---|
| `src/config.ts` | All settings (zod-validated); empty variables count as unset |
| `src/secrets.ts`, `src/logger.ts` | Key scrubbing; every log line is scrubbed before it is written |
| `src/http.ts` | Fetch with a hard timeout, retries that honour `retry-after`, a header-free error type |
| `src/budget.ts` | Daily budgets (one atomic Redis script) and the one-request-per-file lock |
| `src/providers/hiveSchema.ts` | Validation of Hive v3 responses; per-frame and maximum scores |
| `src/providers/hive.ts` | Hive v3 client |
| `src/providers/hiveMock.ts` | Mock Hive: placeholder scores from the file hash, no network |
| `src/providers/externalCheck.ts` | Consent, cache, lock, budget, provider call, storage; never throws |
| `src/providers/groq.ts` | Groq chat-completions client with strict JSON output and model fallback |
| `src/explain/prompt.ts` | The only data Groq sees (whitelisted result JSON) and the system prompt |
| `src/explain/validate.ts` | Rules a Groq explanation must pass, or it is rejected whole |
| `src/explain/explainer.ts` | Groq when valid, template otherwise; never throws |
| `src/explainTemplate.ts` | The template explanation (Phase 1) |
| `src/reconcile.ts` | Merges our verdict with the second check (Phase 1, unchanged) |
| `src/queue/workers.ts` | Analysis and explanation jobs |

## Settings

Keys go only in `.env`, which is never committed. `.env.example` lists every setting with
empty keys.

| Setting | Default | Notes |
|---|---|---|
| `MOCK_MODE` | `true` | Default for both provider modes and the ML service |
| `HIVE_MODE` | follows `MOCK_MODE` | `mock`, `real` or `off` |
| `HIVE_API_KEY` | empty | Hive V3 key; real mode without it records `not_configured` |
| `HIVE_API_URL` | official v3 endpoint | `https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection` |
| `HIVE_THRESHOLD` | `0.9` | Hive says FAKE at `deepfake >= 0.9` (Hive's recommendation) |
| `HIVE_TIMEOUT_MS` | `15000` | Per request |
| `HIVE_DAILY_LIMIT` | `100` | Per UTC day, all users |
| `HIVE_DEMO_RESERVE` | `20` | Reserved for users with the `admin` role; cannot exceed the daily limit |
| `HIVE_USER_DAILY_LIMIT` | `10` | Per regular user per day |
| `HIVE_RETRIES` | `0` | Each retry is billed; a startup warning appears if set above 0 |
| `GROQ_MODE` | follows `MOCK_MODE` | `mock`, `real` or `off`; `mock` and `off` use the template |
| `GROQ_API_KEY` | empty | Real mode without it uses the template (`not_configured`) |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible |
| `GROQ_MODEL_EXPLAIN` | `openai/gpt-oss-120b` | Tried first |
| `GROQ_MODEL_FALLBACKS` | `openai/gpt-oss-20b` | Comma-separated; tried when a model is missing or rejects the request |
| `GROQ_MODEL_CHAT` | `openai/gpt-oss-20b` | For "Ask about this result" (Phase 5) |
| `GROQ_TIMEOUT_MS` | `20000` | Per request |
| `GROQ_RETRIES` | `2` | For 429, 5xx, timeouts and network errors only |
| `GROQ_DAILY_LIMIT` | `900` | Per UTC day |
| `GROQ_MAX_COMPLETION_TOKENS` | `1024` | Includes the model's reasoning tokens |
| `GROQ_REASONING_EFFORT` | `low` | `low`, `medium` or `high` |
| `GROQ_TEMPERATURE` | `0.2` | |

## Hive (second check)

Built from Hive's v3 documentation ("AI-Generated and Deepfake Content Detection
Playground", checked 16 Nov 2026):

- **Request:** `POST` to `HIVE_API_URL` with `authorization: Bearer <key>` and the file as
  multipart field `media`, named `upload` (never the original file name).
- **Response:** `output[]` holds one entry per frame, each with
  `classes: [{class, value}]` and `extra: [{name: "frame_index"}, {name: "timestamp"}]`.
  The `deepfake` and `ai_generated` classes are required in every frame.
- **Legacy field rejected:** the older v2 `score` field is rejected, not silently accepted.
- **Video:** the reported scores are the maximum over frames (Hive's "any frame" rule). Every
  frame's index, timestamp and scores are kept and returned in `external.frames`.
- **Statuses:** a stored check has a status and, when it produced no scores, a reason:

| Status | Reasons |
|---|---|
| `ok` | — |
| `quota` | `budget:daily_limit`, `budget:user_limit`, `budget:exhausted`, `hive_rate_limited` |
| `timeout` | `timeout`, `peer_in_flight` |
| `error` | `not_configured`, `disabled`, `auth`, `too_large`, `upstream`, `http_<code>`, `network`, `invalid_json`, `invalid_response:<why>`, `client_error` |

Anything but `ok` makes the result a single check.

### Consent

- `externalCheck` must be exactly `true` on the upload; it defaults to `false`.
- Without it the worker never calls the second-check code, and that code also refuses to
  send, cache or record anything unless consent is `true`.

## Budgets (Redis)

Keys are `budget:{hive|groq}:{UTC date}:…` and expire after 48 hours.

- **Atomic reservation:** one Lua script checks and increments in a single step, so
  concurrent requests cannot exceed the limit. In a test, 200 simultaneous reservations
  gave exactly 100 allowed, at most 80 of them for regular users.
- **Demo reserve:** regular users share `HIVE_DAILY_LIMIT − HIVE_DEMO_RESERVE` and are each
  capped at `HIVE_USER_DAILY_LIMIT`. Users whose role in MongoDB is `admin` may use the whole
  limit and are not capped per user. There is no API to grant the role.
- **Counting:** one unit per request about to be sent, never refunded. A request refused by
  the budget sends nothing and costs nothing.
- **Provider refusal:** a Hive 429 marks the day exhausted, so no more Hive requests go out
  that UTC day.
- **One request per file:** a token lock with an expiry (`lock:hive:{sha256}`) ensures only
  one request per file is in flight; a crashed worker cannot hold it forever.
- **Redis outage:** a budget or lock failure means nothing is sent, and the result is a single
  check.

## Groq (explanations)

- **Request:** `POST {GROQ_BASE_URL}/chat/completions` with strict structured output
  (`response_format.json_schema.strict = true`), `reasoning_effort`, `temperature` and
  `max_completion_tokens` from the settings.
- **Input:** a system prompt with the rules, and the whitelisted result JSON from
  `src/explain/prompt.ts`. Nothing else is sent.
- **Model fallback:** a model answering 400, 404 or 422 moves on to the next model.
- **Errors:** 401/403 stops at once; 429, 5xx, timeouts and network errors are retried, then
  give up.
- **Rejected outputs:** truncated output (`finish_reason: length`), a refusal, an empty
  answer or invalid JSON.
- **Validation:** any failure rejects the whole response:
  - **Schema:** exactly `summary`, `why`, `caveats` (one to five), with length limits.
  - **Banned wording:** no "authentic", no "proven", "proves" or "proof that", and no links.
  - **Automated:** must say the check is automated.
  - **Verdict wording:** REAL must be worded "no manipulation", never "genuine" or "real";
    FAKE must mention manipulation; UNCERTAIN must say no reliable verdict was reached.
  - **Disagreement:** if the two checks disagree, the explanation must say so.
  - **Mock mode:** if the scores are mock placeholders, the explanation must say so.
  - **Numbers:** every number must come from the result JSON, as given, rounded to the
    written precision, or as a percentage. A score of 0.9 allows "90" or "0.9", but not "1".
- **Fallback and cache:** a rejected or failed response uses the template; the reason is
  stored in `explanations.fallbackReason`. Valid explanations are cached by a SHA-256 of
  (result JSON, models, prompt version).

## Security

- **Logs:** every line is scrubbed of the configured keys and of credential-shaped tokens
  (`Bearer …`, `gsk_…`) before it is written, on top of pino's field redaction.
- **Errors:** HTTP errors carry only the provider and status, never request headers or
  response bodies.
- **Stored responses:** Hive responses are scrubbed before storage; Groq request bodies are
  never stored.
- **No secrets in the repository:** keys never appear in API responses, stored records,
  commits or documentation. `.env` is gitignored.

## Tests

```bash
npm test          # needs Redis on localhost:6379 (database 15 is used and cleaned up)
npm run typecheck && npm run lint && npx prettier --check src tests
```

- **Redis tests fail if Redis is missing.** The budget and second-check tests do not skip
  silently; `SKIP_REDIS_TESTS=1` skips them on purpose, and `TEST_REDIS_URL` points them
  elsewhere.
- **No network.** Hive and Groq are tested with fake HTTP responses; the Hive fixture is
  Hive's documented example.
- **End to end:** `node scripts/smoke_e2e.mjs` from the repository root, against a running
  stack.

## Known limitations

- **Live behaviour unverified.** No live Hive or Groq call has been made. Response details
  (for example extra fields or a different error body) may differ from the documentation.
- **Hive details not documented:** pricing, whether the daily limit resets at midnight UTC
  or on a rolling 24 hours, and the video length limit for multipart uploads. A mismatch
  in the reset rule shows up as `hive_rate_limited` and stops requests for the day.
- **Groq's published limits conflict:** the rate-limits page lists 30 requests a minute and
  1,000 a day for these models, the models page lists higher Developer-plan limits.
  `GROQ_DAILY_LIMIT` defaults to 900 to stay under the lower figure.
- **The number check is strict by design.** Explanations that round differently fall back
  to the template; expect some fallbacks until the prompt has been tuned on real outputs.
- **No admin-role API.** The demo reserve depends on the `admin` role, which can only be
  set directly in MongoDB.
- **Phase 5.** "Ask about this result" (`/ask`) and the PDF report text are not part of
  Phase 4.
