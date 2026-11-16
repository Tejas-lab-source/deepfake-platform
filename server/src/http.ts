/**
 * HTTP helpers for external APIs: a fetch with a hard timeout, a retry loop that honours
 * `retry-after`, and one error type that never carries request headers (so a key cannot leak
 * through an error object).
 */
import { scrub } from './secrets.js';

export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;
export type ErrorKind = 'timeout' | 'network' | 'http' | 'parse';

export class HttpError extends Error {
  constructor(
    readonly provider: string,
    readonly kind: ErrorKind,
    message: string,
    readonly status: number | null = null,
    readonly retryAfterMs: number | null = null,
  ) {
    super(scrub(`${provider}: ${message}`));
    this.name = 'HttpError';
  }

  /** Worth retrying: timeouts, network failures, 429 and 5xx. Never other 4xx. */
  get retryable(): boolean {
    if (this.kind === 'timeout' || this.kind === 'network') return true;
    if (this.kind === 'http' && this.status !== null) {
      return this.status === 429 || this.status >= 500;
    }
    return false;
  }
}

/** Parse `retry-after` (seconds or an HTTP date) into milliseconds. */
export function parseRetryAfter(value: string | null, now = Date.now()): number | null {
  if (!value) return null;
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) return Math.round(seconds * 1000);
  const date = Date.parse(value);
  return Number.isNaN(date) ? null : Math.max(0, date - now);
}

/** fetch with a hard timeout. Throws HttpError for timeouts and network failures; returns
 * the Response for any HTTP status (the caller decides what a status means). */
export async function fetchWithTimeout(
  fetchImpl: FetchLike,
  provider: string,
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetchImpl(url, { ...init, signal: controller.signal });
  } catch (err) {
    if (controller.signal.aborted) {
      throw new HttpError(provider, 'timeout', `no response within ${timeoutMs} ms`);
    }
    const reason = err instanceof Error ? err.message : String(err);
    throw new HttpError(provider, 'network', `request failed: ${reason}`);
  } finally {
    clearTimeout(timer);
  }
}

const MAX_BODY_BYTES = 2 * 1024 * 1024;

/** Read a JSON body (at most 2 MB). Non-2xx statuses become HttpError with the status and,
 * for 429, the retry-after delay. The error message carries the status only, never the body,
 * which can echo request details. */
export async function readJson(response: Response, provider: string): Promise<unknown> {
  if (!response.ok) {
    const retryAfter = parseRetryAfter(response.headers.get('retry-after'));
    await response.body?.cancel().catch(() => undefined);
    throw new HttpError(provider, 'http', `HTTP ${response.status}`, response.status, retryAfter);
  }
  const text = await response.text();
  if (text.length > MAX_BODY_BYTES) {
    throw new HttpError(provider, 'parse', `response larger than ${MAX_BODY_BYTES} bytes`);
  }
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new HttpError(provider, 'parse', 'response is not valid JSON');
  }
}

export interface RetryOptions {
  retries: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
  /** Give up instead of waiting when retry-after asks for longer than this. */
  maxRetryAfterMs?: number;
  sleep?: (ms: number) => Promise<void>;
  random?: () => number;
  onRetry?: (attempt: number, err: HttpError, delayMs: number) => void;
}

const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/** Run `fn`, retrying retryable HttpErrors up to `retries` times with jittered exponential
 * backoff, or the server's retry-after when it is short enough. Other errors are rethrown. */
export async function withRetries<T>(
  fn: (attempt: number) => Promise<T>,
  opts: RetryOptions,
): Promise<T> {
  const base = opts.baseDelayMs ?? 500;
  const cap = opts.maxDelayMs ?? 5_000;
  const maxRetryAfter = opts.maxRetryAfterMs ?? 5_000;
  const sleep = opts.sleep ?? defaultSleep;
  const random = opts.random ?? Math.random;
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn(attempt);
    } catch (err) {
      if (!(err instanceof HttpError) || !err.retryable || attempt >= opts.retries) throw err;
      if (err.retryAfterMs !== null && err.retryAfterMs > maxRetryAfter) throw err;
      const backoff = Math.min(cap, base * 2 ** attempt) * (0.5 + random() / 2);
      const delay = err.retryAfterMs ?? Math.round(backoff);
      opts.onRetry?.(attempt + 1, err, delay);
      await sleep(delay);
    }
  }
}
