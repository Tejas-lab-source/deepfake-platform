import { describe, expect, it, vi } from 'vitest';
import {
  fetchWithTimeout,
  HttpError,
  parseRetryAfter,
  readJson,
  withRetries,
  type FetchLike,
} from '../src/http.js';
import { registerSecrets } from '../src/secrets.js';

const KEY = 'hive-http-test-key-0123456789';
registerSecrets([KEY]);

const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers });

describe('fetchWithTimeout', () => {
  it('aborts a slow request and reports a timeout', async () => {
    const slow: FetchLike = (_url, init) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new Error('aborted')));
      });
    const err = await fetchWithTimeout(slow, 'hive', 'https://x', {}, 20).catch((e) => e);
    expect(err).toBeInstanceOf(HttpError);
    expect(err).toMatchObject({ kind: 'timeout', retryable: true });
  });

  it('reports network failures without leaking the key', async () => {
    const failing: FetchLike = () => Promise.reject(new Error(`ECONNRESET while sending ${KEY}`));
    const err = (await fetchWithTimeout(failing, 'hive', 'https://x', {}, 1000).catch(
      (e) => e,
    )) as HttpError;
    expect(err.kind).toBe('network');
    expect(err.message).not.toContain(KEY);
  });

  it('passes any HTTP status through to the caller', async () => {
    const res = await fetchWithTimeout(async () => json(503, {}), 'hive', 'https://x', {}, 1000);
    expect(res.status).toBe(503);
  });
});

describe('readJson', () => {
  it('parses JSON bodies', async () => {
    expect(await readJson(json(200, { a: 1 }), 'groq')).toEqual({ a: 1 });
  });

  it('turns error statuses into HttpError without the body, keeping retry-after', async () => {
    const err = (await readJson(
      json(429, { error: { message: `echo ${KEY}` } }, { 'retry-after': '2' }),
      'groq',
    ).catch((e) => e)) as HttpError;
    expect(err).toMatchObject({ kind: 'http', status: 429, retryAfterMs: 2000, retryable: true });
    expect(err.message).toBe('groq: HTTP 429');
    const auth = (await readJson(json(401, {}), 'groq').catch((e) => e)) as HttpError;
    expect(auth.retryable).toBe(false);
  });

  it('rejects malformed JSON', async () => {
    const err = await readJson(new Response('<html>', { status: 200 }), 'hive').catch((e) => e);
    expect(err).toMatchObject({ kind: 'parse', retryable: false });
  });
});

describe('parseRetryAfter', () => {
  it('reads seconds and HTTP dates', () => {
    expect(parseRetryAfter('1.5')).toBe(1500);
    expect(parseRetryAfter(new Date(10_000).toUTCString(), 4_000)).toBe(6_000);
    expect(parseRetryAfter('soon')).toBeNull();
    expect(parseRetryAfter(null)).toBeNull();
  });
});

describe('withRetries', () => {
  const sleep = vi.fn(async () => undefined);
  const opts = { sleep, random: () => 0.5, baseDelayMs: 100 };

  it('retries retryable errors, then succeeds', async () => {
    let calls = 0;
    const result = await withRetries(
      async () => {
        calls++;
        if (calls < 3) throw new HttpError('groq', 'http', 'HTTP 503', 503);
        return 'ok';
      },
      { ...opts, retries: 2 },
    );
    expect(result).toBe('ok');
    expect(calls).toBe(3);
  });

  it('never retries with zero retries (the Hive default)', async () => {
    let calls = 0;
    await expect(
      withRetries(
        async () => {
          calls++;
          throw new HttpError('hive', 'timeout', 'slow');
        },
        { ...opts, retries: 0 },
      ),
    ).rejects.toBeInstanceOf(HttpError);
    expect(calls).toBe(1);
  });

  it('does not retry client errors or non-HTTP errors', async () => {
    for (const err of [new HttpError('groq', 'http', 'HTTP 400', 400), new Error('bug')]) {
      let calls = 0;
      await expect(
        withRetries(
          async () => {
            calls++;
            throw err;
          },
          { ...opts, retries: 3 },
        ),
      ).rejects.toBe(err);
      expect(calls).toBe(1);
    }
  });

  it('honours a short retry-after and gives up on a long one', async () => {
    sleep.mockClear();
    let calls = 0;
    await withRetries(
      async () => {
        calls++;
        if (calls === 1) throw new HttpError('groq', 'http', 'HTTP 429', 429, 1200);
        return 'ok';
      },
      { ...opts, retries: 2 },
    );
    expect(sleep).toHaveBeenCalledWith(1200);
    const long = new HttpError('groq', 'http', 'HTTP 429', 429, 60_000);
    await expect(
      withRetries(
        async () => {
          throw long;
        },
        { ...opts, retries: 2 },
      ),
    ).rejects.toBe(long);
  });
});
