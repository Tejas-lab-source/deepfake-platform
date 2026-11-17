import { randomUUID } from 'node:crypto';
import { mkdtempSync, readdirSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { inflateSync } from 'node:zlib';
import { Redis } from 'ioredis';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import {
  APPROVED_URLS,
  assertApprovedUrl,
  DestinationRefused,
  guardedFetch,
  GROQ_BASE_URL,
  HIVE_URL,
  liveAllowed,
  liveDestinations,
  LiveRunRefused,
  M7_BUDGET_USER,
  missingCredentials,
  RepeatRunRefused,
  runCheck,
  syntheticPng,
  type GuardEvent,
  type RecordedRequest,
} from '../scripts/liveCheck.js';
import { DailyBudget } from '../src/budget.js';
import type { FetchLike } from '../src/http.js';

const GROQ_URL = `${GROQ_BASE_URL}/chat/completions`;
const HIVE_KEY = 'hive-m7-test-key-0123456789';
const GROQ_KEY = 'gsk_m7TestKey0123456789abcdef';
const hiveDoc = readFileSync(new URL('./fixtures/hive-v3-image.json', import.meta.url), 'utf8');
const groqOk = JSON.stringify({
  model: 'openai/gpt-oss-120b',
  choices: [
    {
      finish_reason: 'stop',
      message: {
        content: JSON.stringify({
          summary: 'The automated check found signs of face manipulation.',
          why: 'Our pipeline scored 87% against a 50% threshold; the second check scored 12% against 90%, so the two checks disagree.',
          caveats: ['This is an automated check and is not proof.'],
        }),
      },
    },
  ],
});

interface Call {
  url: string;
  authorization: string | undefined;
  redirect: RequestRedirect | undefined;
}

/** A fake network. `hive` and `groq` decide each service's answer. */
function fakeNetwork(
  answers: {
    hive?: () => Promise<Response> | Response;
    groq?: () => Promise<Response> | Response;
  } = {},
) {
  const calls: Call[] = [];
  const fetchImpl: FetchLike = async (url, init) => {
    const headers = (init?.headers ?? {}) as Record<string, string>;
    calls.push({ url, authorization: headers.authorization, redirect: init?.redirect });
    if (url === HIVE_URL) return (answers.hive ?? (() => new Response(hiveDoc)))();
    if (url === GROQ_URL) return (answers.groq ?? (() => new Response(groqOk)))();
    throw new Error(`unexpected URL ${url}`);
  };
  return { fetchImpl, calls };
}

const liveEnv = (over: Record<string, string> = {}): NodeJS.ProcessEnv => ({
  HIVE_MODE: 'real',
  GROQ_MODE: 'real',
  HIVE_API_KEY: HIVE_KEY,
  GROQ_API_KEY: GROQ_KEY,
  M7_LIVE: '1',
  ...over,
});
const outDir = () => mkdtempSync(path.join(tmpdir(), 'm7-out-'));

// ---- real Redis for the budget (fails loudly when missing, like the other budget tests) --
const url = process.env.TEST_REDIS_URL ?? 'redis://127.0.0.1:6379/15';
let redis: Redis;
let available = false;
beforeAll(async () => {
  redis = new Redis(url, { lazyConnect: true, maxRetriesPerRequest: 1, connectTimeout: 1000 });
  try {
    await redis.connect();
    available = (await redis.ping()) === 'PONG';
  } catch {
    available = false;
  }
  if (!available && process.env.SKIP_REDIS_TESTS !== '1') {
    throw new Error(
      `Redis is not reachable at ${url}. Start Redis, or set SKIP_REDIS_TESTS=1 to skip on purpose.`,
    );
  }
});
afterAll(async () => {
  if (available) {
    const keys = await redis.keys('m7t-*');
    if (keys.length) await redis.del(...keys);
  }
  redis.disconnect();
});
const appBudget = (limits = { daily: 100, reserve: 20, perUser: 10 }) =>
  new DailyBudget(
    redis,
    'hive',
    limits,
    () => new Date('2026-11-17T12:00:00Z'),
    `m7t-${randomUUID()}`,
  );

// ---- synthetic data and the live gate ----------------------------------------------------

describe('synthetic data and gate', () => {
  it('generates a valid, deterministic, abstract PNG', () => {
    const a = syntheticPng(64);
    expect(a.subarray(0, 8).toString('hex')).toBe('89504e470d0a1a0a');
    expect(a.readUInt32BE(16)).toBe(64);
    expect(a.readUInt32BE(20)).toBe(64);
    expect(a.equals(syntheticPng(64))).toBe(true);
    const idatStart = a.indexOf('IDAT') + 4;
    const idatLen = a.readUInt32BE(idatStart - 8);
    expect(inflateSync(a.subarray(idatStart, idatStart + idatLen)).length).toBe((64 * 3 + 1) * 64);
  });

  it('refuses live calls without both --live and M7_LIVE=1', () => {
    const real = { HIVE_MODE: 'real', GROQ_MODE: 'real' };
    expect(liveAllowed([], { ...real, M7_LIVE: '1' }).ok).toBe(false);
    expect(liveAllowed(['--live'], { ...real, M7_LIVE: '0' }).ok).toBe(false);
    expect(liveAllowed(['--live'], { M7_LIVE: '1', HIVE_MODE: 'mock', GROQ_MODE: 'off' }).ok).toBe(
      false,
    );
    expect(liveAllowed(['--live'], { ...real, M7_LIVE: '1' }).ok).toBe(true);
  });
});

// ---- 1. destinations ------------------------------------------------------------------

describe('destination allowlist', () => {
  it('accepts exactly the two official endpoints', () => {
    expect([...APPROVED_URLS].sort()).toEqual([HIVE_URL, GROQ_URL].sort());
    expect(assertApprovedUrl(HIVE_URL).hostname).toBe('api.thehive.ai');
    expect(assertApprovedUrl(GROQ_URL).hostname).toBe('api.groq.com');
  });

  it.each([
    ['plain http', HIVE_URL.replace('https:', 'http:')],
    [
      'another port',
      'https://api.thehive.ai:8443/api/v3/hive/ai-generated-and-deepfake-content-detection',
    ],
    ['localhost', 'https://localhost/api/v3/hive/ai-generated-and-deepfake-content-detection'],
    ['loopback IP', 'https://127.0.0.1/api/v3/hive/ai-generated-and-deepfake-content-detection'],
    ['IPv6 loopback', 'https://[::1]/openai/v1/chat/completions'],
    ['private IP', 'https://10.0.0.5/openai/v1/chat/completions'],
    ['private IP 192.168', 'https://192.168.1.20/openai/v1/chat/completions'],
    [
      'lookalike host',
      'https://api.thehive.ai.evil.example/api/v3/hive/ai-generated-and-deepfake-content-detection',
    ],
    ['subdomain lookalike', 'https://xapi.groq.com/openai/v1/chat/completions'],
    ['another path', 'https://api.groq.com/openai/v1/models'],
    ['credentials in URL', 'https://user:pass@api.groq.com/openai/v1/chat/completions'],
    ['query string', `${GROQ_URL}?redirect=https://evil.example`],
    ['not a URL', 'api.groq.com'],
  ])('refuses %s', (_name, raw) => {
    expect(() => assertApprovedUrl(raw)).toThrow(DestinationRefused);
  });

  it('refuses custom endpoint settings in live mode, before anything starts', async () => {
    expect(liveDestinations({}).hiveUrl).toBe(HIVE_URL);
    expect(liveDestinations({ GROQ_BASE_URL: `${GROQ_BASE_URL}/` }).groqBase).toBe(GROQ_BASE_URL);
    expect(() => liveDestinations({ HIVE_API_URL: 'https://127.0.0.1/x' })).toThrow(
      DestinationRefused,
    );
    expect(() => liveDestinations({ GROQ_BASE_URL: 'http://api.groq.com/openai/v1' })).toThrow(
      DestinationRefused,
    );

    const { fetchImpl, calls } = fakeNetwork();
    const out = outDir();
    await expect(
      runCheck({
        live: true,
        env: liveEnv({ HIVE_API_URL: 'https://evil.example/collect' }),
        fetchImpl,
        outDir: out,
      }),
    ).rejects.toThrow(DestinationRefused);
    expect(calls).toHaveLength(0);
    expect(readdirSync(out)).toEqual([]); // not even a run marker
  });

  it('never follows a redirect, so the key never reaches another host', async () => {
    const events: GuardEvent[] = [];
    const inner = vi.fn<FetchLike>(
      async () =>
        new Response(null, {
          status: 302,
          headers: { location: 'https://evil.example/steal?token=abc' },
        }),
    );
    const guarded = guardedFetch(inner, events);
    await expect(
      guarded(HIVE_URL, { method: 'POST', headers: { authorization: `Bearer ${HIVE_KEY}` } }),
    ).rejects.toThrow(/redirect to https:\/\/evil\.example\/steal refused/);
    expect(inner).toHaveBeenCalledTimes(1);
    expect(inner.mock.calls[0]![0]).toBe(HIVE_URL);
    expect(inner.mock.calls[0]![1]?.redirect).toBe('manual');
    expect(events).toEqual([
      {
        url: HIVE_URL,
        outcome: 'refused_redirect',
        detail: 'redirect to https://evil.example/steal',
      },
    ]);
    // A refused destination never reaches the network at all.
    await expect(
      guarded('https://evil.example/x', { headers: { authorization: 'Bearer x' } }),
    ).rejects.toThrow(DestinationRefused);
    expect(inner).toHaveBeenCalledTimes(1);
  });

  it('a live run with redirecting providers sends credentials only to approved URLs', async () => {
    if (!available) return;
    const redirect = () =>
      new Response(null, { status: 307, headers: { location: 'https://evil.example/' } });
    const { fetchImpl, calls } = fakeNetwork({ hive: redirect, groq: redirect });
    const report = await runCheck({
      live: true,
      env: liveEnv(),
      fetchImpl,
      outDir: outDir(),
      budget: appBudget(),
    });
    expect(calls.map((c) => c.url)).toEqual([HIVE_URL, GROQ_URL]);
    expect(calls.every((c) => c.redirect === 'manual')).toBe(true);
    expect(report.hive).toMatchObject({ status: 'error', reason: 'network' });
    expect(report.groq).toMatchObject({ status: 'error', attempts: 1 });
    const guard = report.destinationGuard as GuardEvent[];
    expect(guard.map((g) => g.outcome)).toEqual(['refused_redirect', 'refused_redirect']);
    expect(JSON.stringify(report)).not.toContain(HIVE_KEY);
    expect(JSON.stringify(report)).not.toContain(GROQ_KEY);
  });
});

// ---- live gate and credentials inside runCheck (no Redis needed) ---------------------------

/** A budget that records reservations; it must never be touched by a refused run. */
const spyBudget = () => {
  const reserve = vi.fn(async () => ({ allowed: true }));
  return { budget: { reserve } as unknown as DailyBudget, reserve };
};

/** Replaces the real fetch for the test, so even a regression can never reach the network. */
const blockRealFetch = () =>
  vi
    .spyOn(globalThis, 'fetch')
    .mockImplementation(() => Promise.reject(new Error('real network is blocked in tests')));

describe('live gate inside runCheck (direct calls cannot bypass it)', () => {
  it.each([
    ['M7_LIVE missing', { M7_LIVE: undefined }],
    ['M7_LIVE=0', { M7_LIVE: '0' }],
    ['M7_LIVE=true', { M7_LIVE: 'true' }],
    ['no real mode', { HIVE_MODE: 'mock', GROQ_MODE: 'off' }],
  ])('refuses runCheck({live:true}) with %s: no request, marker or budget', async (_n, over) => {
    const env = { ...liveEnv(), ...over } as NodeJS.ProcessEnv;
    const realFetch = blockRealFetch();
    const { budget, reserve } = spyBudget();
    const out = outDir();
    try {
      // No fetchImpl: this is the path that would use the real network.
      await expect(runCheck({ live: true, env, outDir: out, budget })).rejects.toThrow(
        LiveRunRefused,
      );
      // An injected network is not reached either.
      const { fetchImpl, calls } = fakeNetwork();
      await expect(runCheck({ live: true, env, fetchImpl, outDir: out, budget })).rejects.toThrow(
        LiveRunRefused,
      );
      expect(calls).toHaveLength(0);
      expect(realFetch).not.toHaveBeenCalled();
    } finally {
      realFetch.mockRestore();
    }
    expect(reserve).not.toHaveBeenCalled();
    expect(readdirSync(out)).toEqual([]);
  });

  it('keeps the dry run working without M7_LIVE', async () => {
    const realFetch = blockRealFetch();
    try {
      const report = await runCheck({ live: false, env: {}, outDir: outDir() });
      expect(report.mode).toMatch(/^DRY RUN/);
      expect(report.groq).toMatchObject({ status: 'ok' });
      expect(realFetch).not.toHaveBeenCalled();
    } finally {
      realFetch.mockRestore();
    }
  });
});

describe('credentials are checked before anything is consumed', () => {
  it('names missing keys only for the services in real mode', () => {
    expect(missingCredentials(liveEnv())).toEqual([]);
    expect(missingCredentials(liveEnv({ HIVE_API_KEY: '' }))).toEqual(['HIVE_API_KEY']);
    expect(missingCredentials(liveEnv({ GROQ_API_KEY: '   ' }))).toEqual(['GROQ_API_KEY']);
    expect(missingCredentials(liveEnv({ HIVE_MODE: 'off', HIVE_API_KEY: '' }))).toEqual([]);
    expect(missingCredentials(liveEnv({ HIVE_API_KEY: 'placeholder' }))).toEqual([]);
  });

  it.each([
    ['Hive', { HIVE_API_KEY: '' }, 'HIVE_API_KEY', GROQ_KEY],
    ['Hive (whitespace)', { HIVE_API_KEY: '  ' }, 'HIVE_API_KEY', GROQ_KEY],
    ['Groq', { GROQ_API_KEY: '' }, 'GROQ_API_KEY', HIVE_KEY],
    ['Groq (Groq-only run)', { HIVE_MODE: 'off', GROQ_API_KEY: '' }, 'GROQ_API_KEY', HIVE_KEY],
  ])(
    'a missing %s key: no request, no marker, no budget, no key value in the error',
    async (_n, over, name, otherKey) => {
      const env = liveEnv(over as Record<string, string>);
      const { fetchImpl, calls } = fakeNetwork();
      const { budget, reserve } = spyBudget();
      const out = outDir();
      const err = await runCheck({ live: true, env, fetchImpl, outDir: out, budget }).catch(
        (e: unknown) => e,
      );
      expect(err).toBeInstanceOf(LiveRunRefused);
      expect((err as Error).message).toContain(name);
      expect((err as Error).message).not.toContain(otherKey);
      expect(calls).toHaveLength(0);
      expect(reserve).not.toHaveBeenCalled();
      expect(readdirSync(out)).toEqual([]);
    },
  );
});

// ---- 2. Hive budget ----------------------------------------------------------------------

describe('Redis Hive budget (real Redis)', () => {
  const hiveOnly = (over: Record<string, string> = {}) => liveEnv({ GROQ_MODE: 'off', ...over });

  it('reserves exactly one unit before the single Hive request and reports usage', async () => {
    if (!available) return;
    const budget = appBudget();
    const reserve = vi.spyOn(budget, 'reserve');
    const { fetchImpl, calls } = fakeNetwork();
    const report = await runCheck({
      live: true,
      env: hiveOnly(),
      fetchImpl,
      outDir: outDir(),
      budget,
    });
    expect(reserve).toHaveBeenCalledTimes(1);
    expect(reserve).toHaveBeenCalledWith({ userId: M7_BUDGET_USER, demo: true });
    expect(calls).toHaveLength(1);
    expect(report.hive).toMatchObject({
      status: 'ok',
      deepfakeScore: 0.03,
      reservation: { allowed: true, used: 1 },
      budgetAfter: {
        used: 1,
        demoUsed: 1,
        exhausted: false,
        limits: { daily: 100, reserve: 20, perUser: 10 },
      },
    });
  });

  it('sends nothing when the reservation is denied', async () => {
    if (!available) return;
    const { fetchImpl, calls } = fakeNetwork();
    const report = await runCheck({
      live: true,
      env: hiveOnly(),
      fetchImpl,
      outDir: outDir(),
      budget: appBudget({ daily: 0, reserve: 0, perUser: -1 }),
    });
    expect(calls).toHaveLength(0);
    expect(report.hive).toMatchObject({
      status: 'not_sent',
      reason: 'budget:daily_limit',
      requestsSent: 0,
    });
  });

  it('sends nothing when Redis is unavailable', async () => {
    const { fetchImpl, calls } = fakeNetwork();
    const report = await runCheck({
      live: true,
      env: hiveOnly({ REDIS_URL: 'redis://127.0.0.1:1' }),
      fetchImpl,
      outDir: outDir(),
    });
    expect(calls).toHaveLength(0);
    expect(report.hive).toMatchObject({ status: 'not_sent', requestsSent: 0 });
    expect(String((report.hive as { reason: string }).reason)).toMatch(/^budget_unavailable/);
  });

  it('does not refund the reservation after a timeout', async () => {
    if (!available) return;
    const budget = appBudget();
    const hang: FetchLike = (_u, init) =>
      new Promise((_r, reject) =>
        init?.signal?.addEventListener('abort', () => reject(new Error('aborted'))),
      );
    const report = await runCheck({
      live: true,
      env: hiveOnly({ HIVE_TIMEOUT_MS: '50' }),
      fetchImpl: hang,
      outDir: outDir(),
      budget,
    });
    expect(report.hive).toMatchObject({ status: 'timeout', budgetAfter: { used: 1 } });
    expect((await budget.usage()).used).toBe(1);
  });

  it('marks the day exhausted when Hive answers 429, like the app', async () => {
    if (!available) return;
    const budget = appBudget();
    const { fetchImpl } = fakeNetwork({ hive: () => new Response('{}', { status: 429 }) });
    const report = await runCheck({
      live: true,
      env: hiveOnly(),
      fetchImpl,
      outDir: outDir(),
      budget,
    });
    expect(report.hive).toMatchObject({
      status: 'quota',
      budgetAfter: { used: 1, exhausted: true },
    });
  });

  it('keeps the Groq test to one HTTP attempt and one model', async () => {
    for (const status of [503, 404, 429]) {
      const { fetchImpl, calls } = fakeNetwork({ groq: () => new Response('{}', { status }) });
      const report = await runCheck({
        live: true,
        env: liveEnv({ HIVE_MODE: 'off' }),
        fetchImpl,
        outDir: outDir(),
      });
      expect(calls.map((c) => c.url)).toEqual([GROQ_URL]);
      expect(report.groq).toMatchObject({ status: 'error', attempts: 1 });
    }
  });
});

// ---- 3. repeat runs ------------------------------------------------------------------------

describe('repeat-run protection', () => {
  it('refuses a second live run unless --allow-repeat is given, and never overwrites reports', async () => {
    if (!available) return;
    const out = outDir();
    const first = fakeNetwork();
    const day1 = () => new Date('2026-11-17T10:00:00Z');
    await runCheck({
      live: true,
      env: liveEnv(),
      fetchImpl: first.fetchImpl,
      outDir: out,
      budget: appBudget(),
      now: day1,
    });
    const afterFirst = readdirSync(out).sort();
    expect(afterFirst).toEqual([
      'live-report-2026-11-17T10-00-00-000Z.json',
      'live-run-2026-11-17T10-00-00-000Z.marker',
    ]);
    const firstReport = readFileSync(path.join(out, afterFirst[0]!), 'utf8');

    const second = fakeNetwork();
    await expect(
      runCheck({
        live: true,
        env: liveEnv(),
        fetchImpl: second.fetchImpl,
        outDir: out,
        budget: appBudget(),
        now: day1,
      }),
    ).rejects.toThrow(RepeatRunRefused);
    expect(second.calls).toHaveLength(0);
    expect(readdirSync(out).sort()).toEqual(afterFirst);

    const third = fakeNetwork();
    await runCheck({
      live: true,
      env: liveEnv(),
      fetchImpl: third.fetchImpl,
      outDir: out,
      budget: appBudget(),
      allowRepeat: true,
      now: () => new Date('2026-11-17T11:00:00Z'),
    });
    expect(third.calls).toHaveLength(2);
    expect(readdirSync(out)).toHaveLength(4);
    expect(readFileSync(path.join(out, afterFirst[0]!), 'utf8')).toBe(firstReport);
  });

  it('a failed live run still blocks an unplanned repeat (its marker is written before any request)', async () => {
    const out = outDir();
    const boom: FetchLike = () => Promise.reject(new Error('process killed'));
    // Groq-only with a failing network: the run ends with an error report; the next run is
    // still refused.
    await runCheck({
      live: true,
      env: liveEnv({ HIVE_MODE: 'off' }),
      fetchImpl: boom,
      outDir: out,
    });
    await expect(
      runCheck({ live: true, env: liveEnv({ HIVE_MODE: 'off' }), fetchImpl: boom, outDir: out }),
    ).rejects.toThrow(RepeatRunRefused);
  });
});

// ---- dry run unchanged -----------------------------------------------------------------------

describe('dry run', () => {
  it('makes no network call, sends only synthetic minimal data, and ignores earlier live runs', async () => {
    const out = outDir();
    if (available) {
      // An earlier live run must not block a dry run.
      await runCheck({
        live: true,
        env: liveEnv({ GROQ_MODE: 'off' }),
        fetchImpl: fakeNetwork().fetchImpl,
        outDir: out,
        budget: appBudget(),
      });
    }
    const networkFetch = vi.spyOn(globalThis, 'fetch');
    const report = await runCheck({
      live: false,
      env: { GROQ_API_KEY: 'gsk_dryRunPlaceholder0123456789' },
      outDir: out,
    });
    expect(networkFetch).not.toHaveBeenCalled();
    networkFetch.mockRestore();

    const requests = report.requestsThatWouldBeSent as RecordedRequest[];
    expect(requests.map((r) => r.service)).toEqual(['hive', 'groq']);
    const [hive, groq] = requests as [RecordedRequest, RecordedRequest];
    expect(hive.url).toBe(HIVE_URL);
    expect(hive.formFields).toEqual([
      { name: 'media', fileName: 'upload', type: 'image/png', bytes: expect.any(Number) },
    ]);
    expect(hive.headerNames).toEqual(['accept', 'authorization']);
    expect(groq.url).toBe(GROQ_URL);
    const messages = (groq.jsonBody as { messages: Array<{ role: string; content: string }> })
      .messages;
    expect(messages.map((m) => m.role)).toEqual(['system', 'user']);
    const facts = JSON.parse(messages[1]!.content.replace(/^Result JSON:\n/, ''));
    expect(Object.keys(facts).sort()).toEqual([
      'checksDisagree',
      'finalVerdict',
      'mockMode',
      'modelCardVersion',
      'ourPipeline',
      'ruleApplied',
      'secondCheck',
      'verificationBadge',
    ]);
    const sent = JSON.stringify(requests);
    for (const forbidden of [tmpdir(), 'synthetic.png', '/data/uploads', '@', 'sha256', 'base64']) {
      expect(sent).not.toContain(forbidden);
    }
    expect(report.hive).toMatchObject({ status: 'ok', deepfakeScore: 0.03 });
    expect(report.hive).not.toHaveProperty('reservation'); // no Redis in a dry run
    expect(report.groq).toMatchObject({ status: 'ok', validation: { ok: true } });
    expect(JSON.stringify(report)).not.toContain('gsk_dryRunPlaceholder0123456789');
    expect(readdirSync(out)).toContain('dry-run-report.json');
  });
});
