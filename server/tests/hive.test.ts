import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import type { FetchLike } from '../src/http.js';
import { HiveVerifier } from '../src/providers/hive.js';
import { parseHiveResponse } from '../src/providers/hiveSchema.js';

const KEY = 'hive-unit-test-key-0123456789';
const URL_ = 'https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection';
const doc = JSON.parse(
  readFileSync(new URL('./fixtures/hive-v3-image.json', import.meta.url), 'utf8'),
);

function video(frames: Array<[number, number, number]>) {
  return {
    task_id: 't',
    model: 'm',
    output: frames.map(([ts, deepfake, ai], i) => ({
      extra: [
        { name: 'frame_index', value: i },
        { name: 'timestamp', value: ts },
      ],
      classes: [
        { class: 'not_ai_generated', value: 1 - ai },
        { class: 'ai_generated', value: ai },
        { class: 'deepfake', value: deepfake },
      ],
    })),
  };
}

const dir = mkdtempSync(path.join(tmpdir(), 'hive-test-'));
const file = path.join(dir, 'abc123');
writeFileSync(file, Buffer.from([0xff, 0xd8, 0xff, 0xe0, 1, 2, 3]));

function client(responses: Array<Response | Error>, extra: Partial<{ retries: number }> = {}) {
  const calls: Array<{ url: string; init: RequestInit }> = [];
  const fetchImpl: FetchLike = async (url, init) => {
    calls.push({ url, init: init ?? {} });
    const next = responses.shift();
    if (!next) throw new Error('no more responses');
    if (next instanceof Error) throw next;
    return next;
  };
  const hive = new HiveVerifier({
    apiKey: KEY,
    url: URL_,
    timeoutMs: 1000,
    retries: 0,
    fetchImpl,
    ...extra,
  });
  return { hive, calls };
}

const ok = (body: unknown) => new Response(JSON.stringify(body), { status: 200 });

describe('parseHiveResponse', () => {
  it('reads the documented v3 example', () => {
    const r = parseHiveResponse(doc);
    expect(r).toMatchObject({ ok: true, deepfakeScore: 0.03, aiGeneratedScore: 0.02 });
    if (r.ok)
      expect(r.frames).toEqual([
        { frameIndex: 0, timestamp: 0, deepfake: 0.03, aiGenerated: 0.02 },
      ]);
  });

  it('takes the maximum over video frames and keeps every frame with its timestamp', () => {
    const r = parseHiveResponse(
      video([
        [0, 0.1, 0.2],
        [0.5, 0.95, 0.1],
        [1.0, 0.3, 0.6],
      ]),
    );
    expect(r).toMatchObject({ ok: true, deepfakeScore: 0.95, aiGeneratedScore: 0.6 });
    if (r.ok) expect(r.frames.map((f) => f.timestamp)).toEqual([0, 0.5, 1.0]);
  });

  it('rejects the legacy v2 "score" field instead of accepting it silently', () => {
    const legacy = {
      output: [
        {
          classes: [
            { class: 'deepfake', score: 0.9 },
            { class: 'ai_generated', score: 0.1 },
          ],
        },
      ],
    };
    expect(parseHiveResponse(legacy)).toMatchObject({ ok: false, reason: 'legacy_score_field' });
  });

  it.each([
    ['no output', { task_id: 'x' }, 'schema'],
    ['empty output', { output: [] }, 'schema'],
    ['value out of range', video([[0, 1.5, 0.1]]), 'schema'],
    [
      'missing deepfake',
      { output: [{ classes: [{ class: 'ai_generated', value: 0.1 }] }] },
      'missing_class',
    ],
    [
      'missing ai_generated',
      { output: [{ classes: [{ class: 'deepfake', value: 0.1 }] }] },
      'missing_class',
    ],
    [
      'duplicate class',
      {
        output: [
          {
            classes: [
              { class: 'deepfake', value: 0.1 },
              { class: 'deepfake', value: 0.2 },
              { class: 'ai_generated', value: 0 },
            ],
          },
        ],
      },
      'duplicate_class',
    ],
    ['not an object', 'hello', 'schema'],
  ])('rejects %s', (_name, body, reason) => {
    expect(parseHiveResponse(body)).toMatchObject({ ok: false, reason });
  });
});

describe('HiveVerifier', () => {
  it('sends one multipart request with the bearer key and a neutral file name', async () => {
    const { hive, calls } = client([ok(doc)]);
    const r = await hive.verify(file, { mime: 'image/jpeg' });
    expect(r).toMatchObject({ status: 'ok', deepfakeScore: 0.03, aiGeneratedScore: 0.02 });
    expect(calls).toHaveLength(1);
    const { url, init } = calls[0]!;
    expect(url).toBe(URL_);
    expect(init.method).toBe('POST');
    expect((init.headers as Record<string, string>).authorization).toBe(`Bearer ${KEY}`);
    const body = init.body as FormData;
    const media = body.get('media') as File;
    expect([...body.keys()]).toEqual(['media']);
    expect(media.name).toBe('upload');
    expect(media.type).toBe('image/jpeg');
    expect(media.size).toBe(7);
  });

  it('never calls Hive without a key', async () => {
    const { calls } = client([]);
    const hive = new HiveVerifier({
      apiKey: '',
      url: URL_,
      timeoutMs: 1000,
      retries: 0,
      fetchImpl: vi.fn(),
    });
    expect(await hive.verify(file)).toMatchObject({ status: 'error', reason: 'not_configured' });
    expect(calls).toHaveLength(0);
  });

  it.each([
    [429, 'quota', 'hive_rate_limited'],
    [401, 'error', 'auth'],
    [403, 'error', 'auth'],
    [413, 'error', 'too_large'],
    [400, 'error', 'http_400'],
    [500, 'error', 'upstream'],
  ])('maps HTTP %i to %s (%s) and makes no retry by default', async (code, status, reason) => {
    const { hive, calls } = client([new Response('{}', { status: code }), ok(doc)]);
    const r = await hive.verify(file);
    expect(r).toMatchObject({ status, reason, deepfakeScore: null });
    expect(calls).toHaveLength(1);
    if (code === 429) expect(r.rateLimited).toBe(true);
  });

  it('reports timeouts, network failures and bad JSON', async () => {
    const slow: FetchLike = (_u, init) =>
      new Promise((_r, reject) =>
        init?.signal?.addEventListener('abort', () => reject(new Error('x'))),
      );
    const timeout = new HiveVerifier({
      apiKey: KEY,
      url: URL_,
      timeoutMs: 20,
      retries: 0,
      fetchImpl: slow,
    });
    expect(await timeout.verify(file)).toMatchObject({ status: 'timeout' });
    expect(await client([new Error('ECONNRESET')]).hive.verify(file)).toMatchObject({
      status: 'error',
      reason: 'network',
    });
    expect(await client([new Response('<html>', { status: 200 })]).hive.verify(file)).toMatchObject(
      {
        status: 'error',
        reason: 'invalid_json',
      },
    );
  });

  it('keeps an unexpected response for inspection but returns no scores', async () => {
    const legacy = { output: [{ classes: [{ class: 'deepfake', score: 0.9 }] }] };
    const r = await client([ok(legacy)]).hive.verify(file);
    expect(r).toMatchObject({
      status: 'error',
      reason: 'invalid_response:legacy_score_field',
      deepfakeScore: null,
    });
    expect(JSON.stringify(r.raw)).toContain('"score":0.9');
  });

  it('reserves budget before every attempt and stops when denied', async () => {
    const reserve = vi.fn().mockResolvedValueOnce(true).mockResolvedValueOnce(false);
    const { hive, calls } = client([new Response('{}', { status: 503 }), ok(doc)], { retries: 1 });
    const r = await hive.verify(file, { reserveAttempt: reserve });
    expect(reserve).toHaveBeenCalledTimes(2);
    expect(calls).toHaveLength(1); // the retry was not sent: no budget left
    expect(r).toMatchObject({ status: 'quota', reason: 'budget' });
    const denied = client([ok(doc)]);
    expect(await denied.hive.verify(file, { reserveAttempt: async () => false })).toMatchObject({
      status: 'quota',
    });
    expect(denied.calls).toHaveLength(0);
  });

  it('never puts the key into the result', async () => {
    for (const res of [ok(doc), new Response('{}', { status: 401 }), new Error(`boom ${KEY}`)]) {
      const r = await client([res]).hive.verify(file);
      expect(JSON.stringify(r)).not.toContain(KEY);
    }
  });
});
