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

At most **one billed request per service**: no retries and a single Groq model. The live
run calls the providers directly, outside the app's Redis budget counter, so this one Hive
request also counts against Hive's own 100-a-day limit.

## Files

| File | Purpose | Committed |
|---|---|---|
| `m7.env.example` | Settings template with empty keys | Yes |
| `synthetic-result.json` | Synthetic analysis result for the Groq check | Yes |
| `../scripts/liveCheck.ts` | The check: dry run by default; live only with `--live` and `M7_LIVE=1` | Yes |
| `.env.m7` | Your local settings, copied from the template | **No** (gitignored) |
| `out/` | Reports, scrubbed of secrets | **No** (gitignored) |

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
4. Review `out/live-report.json`:
   - Hive: status, scores and the scrubbed raw response;
   - Groq: status, model, token usage, and whether the explanation passed validation.

   If Hive's real response differs from the documented shape, the parser reports
   `invalid_response:<why>` and keeps the raw response, so it can be adapted deliberately.
