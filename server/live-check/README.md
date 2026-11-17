# M7: live Hive and Groq verification

**Status: prepared, not run.** Nothing here has contacted Hive or Groq. A live run needs
explicit approval.

The goal is to confirm that the real APIs behave as their documentation says: one Hive
request and one Groq request, using synthetic data only, through the app's own clients,
parser and validation.

## What gets sent

| Service | Data | Never sent |
|---|---|---|
| Hive | A generated abstract image (gradients and rings: no person, no face), as multipart field `media` named `upload`, type `image/png` | User images, file names, paths, user identity |
| Groq | The system prompt and the whitelisted result JSON built from `synthetic-result.json`, exactly as the app builds it | Any image, file name, path, hash or personal data |

At most **one billed request per service**: no retries and a single Groq model.

## Safeguards in a live run

| Safeguard | Behaviour |
|---|---|
| Live gate | Needs both `--live` and `M7_LIVE=1`; otherwise only a dry run is possible. `runCheck()` enforces `M7_LIVE=1` itself, so importing the script cannot bypass the gate, and the real network re-checks it on every request |
| Credentials | A key required by a service in `real` mode must be set (a placeholder is fine in the sandbox) before anything starts: otherwise no run marker, no budget reservation and no request. Errors name the variable, never its value |
| Destinations | Only `https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection` and `https://api.groq.com/openai/v1/chat/completions`: HTTPS, default port, no credentials or query in the URL. A custom `HIVE_API_URL` or `GROQ_BASE_URL` is refused before anything starts, and every request is checked again just before sending |
| Redirects | Never followed (`redirect: manual`); any 3xx answer is refused, so the authorization header cannot reach another host |
| Hive budget | One unit is reserved in the app's Redis daily Hive budget (same code and limits as the app, demo pool, identity `m7-live-check`) before the Hive request. Denied, or Redis unreachable: nothing is sent. Never refunded, even after a timeout. A Hive 429 marks the day exhausted, as in the app |
| Groq | One HTTP attempt and one model |
| Repeat runs | Refused while any earlier `live-report-*.json` or `live-run-*.marker` exists in `out/`, unless `--allow-repeat` is given. The marker is written before the first request; reports are created new each time and never overwritten or deleted |

A live run needs Redis (default `redis://localhost:6379`, or `REDIS_URL`).

## Files

| File | Purpose | Committed |
|---|---|---|
| `m7.env.example` | Settings template with empty keys | Yes |
| `synthetic-result.json` | Synthetic analysis result for the Groq check | Yes |
| `../scripts/liveCheck.ts` | The check: dry run by default; live only with `--live` and `M7_LIVE=1` | Yes |
| `.env.m7` | Your local settings, copied from the template | **No** (gitignored) |
| `out/` | `dry-run-report.json`, and per live run `live-run-<time>.marker` and `live-report-<time>.json`, scrubbed of secrets | **No** (gitignored) |

## Dry run (no network)

```bash
cd server
npx tsx scripts/liveCheck.ts
```

It records exactly what would be sent (URLs, header names, form fields, JSON body), answers
with simulated documented responses, runs the parser and validation, and writes
`out/dry-run-report.json`. It cannot open a network connection.

## Live run (after approval)

1. Add the API hosts to the environment's network allowlist: `api.thehive.ai` and
   `api.groq.com`.
2. Provide the keys:
   - **In the Claude cloud sandbox:** through the secure credential form, where the proxy
     injects them into requests. Put a placeholder value in `.env.m7`.
   - **On your own machine:** in `.env.m7`, never committed.
3. `cp live-check/m7.env.example live-check/.env.m7`, set `M7_LIVE=1`, then:

   ```bash
   NODE_USE_ENV_PROXY=1 M7_LIVE=1 npx tsx --env-file=live-check/.env.m7 scripts/liveCheck.ts --live
   ```

   `NODE_USE_ENV_PROXY=1` is only needed in the sandbox, where traffic goes through a proxy.
4. Review `out/live-report-<time>.json`:
   - Hive: status, scores and the scrubbed raw response;
   - the budget reservation and usage after the request;
   - the destination guard's log;
   - Groq: status, model, token usage, and whether the explanation passed validation.

   If Hive's real response differs from the documented shape, the parser reports
   `invalid_response:<why>` and keeps the raw response, so it can be adapted deliberately.
