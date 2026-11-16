import { describe, expect, it, vi } from 'vitest';
import type { FetchLike } from '../src/http.js';
import { GroqClient, type GroqOptions } from '../src/providers/groq.js';

const KEY = 'gsk_unitTestKey0123456789abcdef';
const schema = {
  type: 'object',
  properties: { a: { type: 'string' } },
  required: ['a'],
  additionalProperties: false,
};

const completion = (content: string | null, extra: Record<string, unknown> = {}) =>
  new Response(
    JSON.stringify({
      model: 'openai/gpt-oss-120b',
      choices: [
        { index: 0, finish_reason: 'stop', message: { role: 'assistant', content }, ...extra },
      ],
      usage: { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
    }),
    { status: 200 },
  );

function client(responses: Array<Response | Error>, over: Partial<GroqOptions> = {}) {
  const calls: Array<{
    url: string;
    body: Record<string, unknown>;
    headers: Record<string, string>;
  }> = [];
  const fetchImpl: FetchLike = async (url, init) => {
    calls.push({
      url,
      body: JSON.parse(String(init?.body)),
      headers: init?.headers as Record<string, string>,
    });
    const next = responses.shift();
    if (!next) throw new Error('no more responses');
    if (next instanceof Error) throw next;
    return next;
  };
  const groq = new GroqClient({
    apiKey: KEY,
    baseUrl: 'https://api.groq.com/openai/v1',
    models: ['openai/gpt-oss-120b', 'openai/gpt-oss-20b'],
    timeoutMs: 1000,
    retries: 2,
    maxCompletionTokens: 1024,
    reasoningEffort: 'low',
    temperature: 0.2,
    fetchImpl,
    sleep: async () => undefined,
    ...over,
  });
  return { groq, calls };
}

const req = { system: 'sys', user: 'Result JSON:\n{"x":1}', schemaName: 'explanation', schema };

describe('GroqClient', () => {
  it('sends the approved strict structured-output request, text only', async () => {
    const { groq, calls } = client([completion('{"a":"hi"}')]);
    const r = await groq.completeJson(req);
    expect(r).toMatchObject({
      ok: true,
      content: { a: 'hi' },
      model: 'openai/gpt-oss-120b',
      attempts: 1,
    });
    const { url, body, headers } = calls[0]!;
    expect(url).toBe('https://api.groq.com/openai/v1/chat/completions');
    expect(headers.authorization).toBe(`Bearer ${KEY}`);
    expect(body).toEqual({
      model: 'openai/gpt-oss-120b',
      messages: [
        { role: 'system', content: 'sys' },
        { role: 'user', content: 'Result JSON:\n{"x":1}' },
      ],
      response_format: {
        type: 'json_schema',
        json_schema: { name: 'explanation', strict: true, schema },
      },
      temperature: 0.2,
      max_completion_tokens: 1024,
      reasoning_effort: 'low',
    });
  });

  it('falls back to the next model when one is missing or retired', async () => {
    const { groq, calls } = client([new Response('{}', { status: 404 }), completion('{"a":"b"}')]);
    const r = await groq.completeJson(req);
    expect(r).toMatchObject({ ok: true, attempts: 2 });
    expect(calls.map((c) => c.body.model)).toEqual(['openai/gpt-oss-120b', 'openai/gpt-oss-20b']);
  });

  it('stops at once on authentication failures', async () => {
    const { groq, calls } = client([new Response('{}', { status: 401 }), completion('{"a":"b"}')]);
    expect(await groq.completeJson(req)).toMatchObject({ ok: false, reason: 'http_401' });
    expect(calls).toHaveLength(1);
  });

  it('retries 429 and 5xx, then gives up without trying other models', async () => {
    const r429 = () => new Response('{}', { status: 429, headers: { 'retry-after': '1' } });
    const ok = client([r429(), completion('{"a":"b"}')]);
    expect(await ok.groq.completeJson(req)).toMatchObject({ ok: true, attempts: 2 });
    const bad = client([
      new Response('{}', { status: 503 }),
      new Response('{}', { status: 503 }),
      new Response('{}', { status: 503 }),
    ]);
    expect(await bad.groq.completeJson(req)).toMatchObject({
      ok: false,
      reason: 'http_503',
      attempts: 3,
    });
    expect(new Set(bad.calls.map((c) => c.body.model))).toEqual(new Set(['openai/gpt-oss-120b']));
  });

  it.each([
    ['truncated output', completion('{"a":', {}), 'invalid_json'],
    [
      'length stop',
      new Response(
        JSON.stringify({
          choices: [{ finish_reason: 'length', message: { content: '{"a":"b"}' } }],
        }),
      ),
      'truncated',
    ],
    [
      'refusal',
      new Response(
        JSON.stringify({
          choices: [{ finish_reason: 'stop', message: { content: null, refusal: 'no' } }],
        }),
      ),
      'refusal',
    ],
    ['empty content', completion(null), 'empty'],
    ['wrong envelope', new Response(JSON.stringify({ result: 'x' })), 'invalid_response'],
  ])('rejects %s', async (_n, res, reason) => {
    expect(await client([res]).groq.completeJson(req)).toMatchObject({ ok: false, reason });
  });

  it('reserves budget before each attempt and sends nothing when denied', async () => {
    const reserve = vi.fn(async () => false);
    const { groq, calls } = client([completion('{"a":"b"}')]);
    expect(await groq.completeJson({ ...req, reserveAttempt: reserve })).toMatchObject({
      ok: false,
      reason: 'budget',
    });
    expect(calls).toHaveLength(0);
  });

  it('is not configured without a key and never leaks it', async () => {
    const none = client([], { apiKey: '' });
    expect(await none.groq.completeJson(req)).toMatchObject({
      ok: false,
      reason: 'not_configured',
    });
    expect(none.calls).toHaveLength(0);
    for (const res of [
      completion('{"a":"b"}'),
      new Response('{}', { status: 401 }),
      new Error(`boom ${KEY}`),
    ]) {
      expect(JSON.stringify(await client([res]).groq.completeJson(req))).not.toContain(KEY);
    }
  });
});
