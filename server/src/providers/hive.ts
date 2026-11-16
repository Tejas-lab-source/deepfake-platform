/**
 * Real Hive v3 client (AI-Generated and Deepfake Content Detection).
 *
 *   POST {HIVE_API_URL}   authorization: Bearer <key>   multipart field "media"
 *
 * Every attempt is a billable request: `reserveAttempt` is called before each one, and
 * retries default to zero. Results never contain the key or request headers.
 */
import { openAsBlob } from 'node:fs';
import { fetchWithTimeout, HttpError, readJson, withRetries, type FetchLike } from '../http.js';
import { scrubDeep } from '../secrets.js';
import { parseHiveResponse } from './hiveSchema.js';
import type { Verifier, VerifierResult, VerifyOptions } from './verifier.js';

const RAW_LIMIT_BYTES = 256 * 1024;

export interface HiveOptions {
  apiKey: string;
  url: string;
  timeoutMs: number;
  retries: number;
  fetchImpl?: FetchLike;
}

const empty = { deepfakeScore: null, aiGeneratedScore: null } as const;

/** Keep the stored raw response small and free of secrets. */
function storableRaw(json: unknown): unknown {
  const text = JSON.stringify(json);
  if (text.length <= RAW_LIMIT_BYTES) return scrubDeep(json);
  return { truncated: true, bytes: text.length };
}

function failure(err: HttpError): VerifierResult {
  if (err.kind === 'timeout') return { status: 'timeout', ...empty, raw: null, reason: 'timeout' };
  if (err.kind === 'network') return { status: 'error', ...empty, raw: null, reason: 'network' };
  if (err.kind === 'parse') return { status: 'error', ...empty, raw: null, reason: 'invalid_json' };
  const status = err.status ?? 0;
  if (status === 429) {
    return { status: 'quota', ...empty, raw: null, reason: 'hive_rate_limited', rateLimited: true };
  }
  const reason =
    status === 401 || status === 403
      ? 'auth'
      : status === 413
        ? 'too_large'
        : status >= 500
          ? 'upstream'
          : `http_${status}`;
  return { status: 'error', ...empty, raw: { httpStatus: status }, reason };
}

export class HiveVerifier implements Verifier {
  readonly provider = 'hive';
  private readonly fetchImpl: FetchLike;

  constructor(private readonly opts: HiveOptions) {
    this.fetchImpl = opts.fetchImpl ?? ((url, init) => fetch(url, init));
  }

  async verify(filePath: string, options: VerifyOptions = {}): Promise<VerifierResult> {
    if (!this.opts.apiKey) {
      return { status: 'error', ...empty, raw: null, reason: 'not_configured' };
    }
    let json: unknown;
    try {
      json = await withRetries(
        async () => {
          if (options.reserveAttempt && !(await options.reserveAttempt())) {
            throw new BudgetDenied();
          }
          const form = new FormData();
          // A neutral file name: the user's original file name is never sent.
          const blob = await openAsBlob(filePath, options.mime ? { type: options.mime } : {});
          form.append('media', blob, 'upload');
          const res = await fetchWithTimeout(
            this.fetchImpl,
            'hive',
            this.opts.url,
            {
              method: 'POST',
              headers: { authorization: `Bearer ${this.opts.apiKey}`, accept: 'application/json' },
              body: form,
            },
            this.opts.timeoutMs,
          );
          return readJson(res, 'hive');
        },
        { retries: this.opts.retries },
      );
    } catch (err) {
      if (err instanceof BudgetDenied) {
        return { status: 'quota', ...empty, raw: null, reason: 'budget' };
      }
      if (err instanceof HttpError) return failure(err);
      return { status: 'error', ...empty, raw: null, reason: 'client_error' };
    }
    const parsed = parseHiveResponse(json);
    if (!parsed.ok) {
      return {
        status: 'error',
        ...empty,
        raw: { invalid: storableRaw(json), detail: parsed.detail },
        reason: `invalid_response:${parsed.reason}`,
      };
    }
    return {
      status: 'ok',
      deepfakeScore: parsed.deepfakeScore,
      aiGeneratedScore: parsed.aiGeneratedScore,
      frames: parsed.frames,
      raw: storableRaw(json),
    };
  }
}

class BudgetDenied extends Error {}
