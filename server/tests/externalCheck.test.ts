import { randomUUID } from 'node:crypto';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { Redis } from 'ioredis';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { DailyBudget, InflightLock } from '../src/budget.js';
import type { FetchLike } from '../src/http.js';
import {
  runExternalCheck,
  safeExternalCheck,
  type CheckStore,
  type ExternalCheckDeps,
  type NewCheck,
  type StoredCheck,
} from '../src/providers/externalCheck.js';
import { HiveVerifier } from '../src/providers/hive.js';
import { MockHiveVerifier } from '../src/providers/hiveMock.js';
import type { Verifier, VerifierResult } from '../src/providers/verifier.js';
import { reconcile } from '../src/reconcile.js';

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
});
afterAll(async () => {
  if (available) {
    const keys = await redis.keys('x-*');
    if (keys.length) await redis.del(...keys);
  }
  redis.disconnect();
});

const doc = JSON.parse(
  readFileSync(new URL('./fixtures/hive-v3-image.json', import.meta.url), 'utf8'),
);
const dir = mkdtempSync(path.join(tmpdir(), 'ext-test-'));
const file = path.join(dir, 'f');
writeFileSync(file, Buffer.from([0xff, 0xd8, 0xff, 0xe0, 9, 9]));

class MemoryStore implements CheckStore {
  docs: Array<NewCheck & { _id: string }> = [];
  async findCachedOk(sha256: string, provider: string) {
    return (
      [...this.docs]
        .reverse()
        .find((d) => d.mediaSha256 === sha256 && d.provider === provider && d.status === 'ok') ??
      null
    );
  }
  async create(doc: NewCheck): Promise<StoredCheck> {
    const stored = { ...doc, _id: randomUUID() };
    this.docs.push(stored);
    return stored;
  }
}

function fakeVerifier(result: Partial<VerifierResult> = {}, delayMs = 0) {
  const calls: string[] = [];
  const verifier: Verifier = {
    provider: 'hive',
    async verify(filePath, options) {
      calls.push(filePath);
      if (options?.reserveAttempt && !(await options.reserveAttempt())) {
        return {
          status: 'quota',
          deepfakeScore: null,
          aiGeneratedScore: null,
          raw: null,
          reason: 'budget',
        };
      }
      await new Promise((r) => setTimeout(r, delayMs));
      return {
        status: 'ok',
        deepfakeScore: 0.95,
        aiGeneratedScore: 0.1,
        raw: { ok: true },
        ...result,
      };
    },
  };
  return { verifier, calls };
}

function deps(over: Partial<ExternalCheckDeps> = {}): ExternalCheckDeps & { store: MemoryStore } {
  return {
    verifier: fakeVerifier().verifier,
    provider: 'hive',
    budget: null,
    lock: new InflightLock(redis, `x-${randomUUID()}`),
    store: new MemoryStore(),
    threshold: 0.9,
    timeoutMs: 2000,
    pollMs: 10,
    ...over,
  } as ExternalCheckDeps & { store: MemoryStore };
}

const req = (over: Partial<Parameters<typeof runExternalCheck>[1]> = {}) => ({
  sha256: 'a'.repeat(64),
  filePath: file,
  mime: 'image/jpeg',
  userId: 'u1',
  demo: false,
  consent: true,
  ...over,
});

const budget = (limits = { daily: 100, reserve: 20, perUser: 10 }) =>
  new DailyBudget(
    redis,
    'hive',
    limits,
    () => new Date('2026-11-16T12:00:00Z'),
    `x-${randomUUID()}`,
  );

describe('external check orchestration', () => {
  it('sends nothing and records nothing without explicit consent', async ({ skip }) => {
    if (!available) skip();
    const { verifier, calls } = fakeVerifier();
    const d = deps({ verifier });
    for (const consent of [false, undefined as unknown as boolean, 'true' as unknown as boolean]) {
      expect(await runExternalCheck(d, req({ consent }))).toBeNull();
    }
    expect(calls).toHaveLength(0);
    expect(d.store.docs).toHaveLength(0);
  });

  it('reuses a cached ok result without calling the provider or spending budget', async ({
    skip,
  }) => {
    if (!available) skip();
    const { verifier, calls } = fakeVerifier();
    const b = budget();
    const d = deps({ verifier, budget: b });
    const first = await runExternalCheck(d, req());
    const second = await runExternalCheck(d, req({ userId: 'someone-else' }));
    expect(first).toMatchObject({ status: 'ok', verdict: 'FAKE' });
    expect(second?._id).toBe(first?._id);
    expect(calls).toHaveLength(1);
    expect((await b.usage()).used).toBe(1);
  });

  it('does not cache failures: the next upload tries again', async ({ skip }) => {
    if (!available) skip();
    const { verifier, calls } = fakeVerifier({
      status: 'error',
      deepfakeScore: null,
      reason: 'upstream',
    });
    const d = deps({ verifier });
    expect(await runExternalCheck(d, req())).toMatchObject({
      status: 'error',
      reason: 'upstream',
      verdict: null,
    });
    await runExternalCheck(d, req());
    expect(calls).toHaveLength(2);
  });

  it('sends a file once even when it is uploaded many times at once', async ({ skip }) => {
    if (!available) skip();
    const { verifier, calls } = fakeVerifier({}, 150);
    const d = deps({ verifier });
    const results = await Promise.all(
      Array.from({ length: 6 }, (_, i) => runExternalCheck(d, req({ userId: `u${i}` }))),
    );
    expect(calls).toHaveLength(1);
    expect(new Set(results.map((r) => r?._id)).size).toBe(1);
    expect(results.every((r) => r?.status === 'ok')).toBe(true);
  });

  it('enforces the budget through the real Hive client: denied requests are never sent', async ({
    skip,
  }) => {
    if (!available) skip();
    const fetchImpl = vi.fn<FetchLike>(
      async () => new Response(JSON.stringify(doc), { status: 200 }),
    );
    const hive = new HiveVerifier({
      apiKey: 'hive-key-0123456789abcd',
      url: 'https://hive.test',
      timeoutMs: 1000,
      retries: 0,
      fetchImpl,
    });
    const b = budget({ daily: 100, reserve: 20, perUser: 2 });
    const d = deps({ verifier: hive, budget: b });
    for (let i = 0; i < 2; i++) {
      expect(await runExternalCheck(d, req({ sha256: `${i}`.repeat(64) }))).toMatchObject({
        status: 'ok',
      });
    }
    const denied = await runExternalCheck(d, req({ sha256: 'f'.repeat(64) }));
    expect(denied).toMatchObject({ status: 'quota', reason: 'budget:user_limit' });
    expect(fetchImpl).toHaveBeenCalledTimes(2);
    // The demo pool is not subject to the per-user cap.
    expect(
      await runExternalCheck(d, req({ sha256: 'e'.repeat(64), userId: 'admin', demo: true })),
    ).toMatchObject({ status: 'ok' });
  });

  it('stops for the day after Hive answers 429', async ({ skip }) => {
    if (!available) skip();
    const fetchImpl = vi.fn<FetchLike>(async () => new Response('{}', { status: 429 }));
    const hive = new HiveVerifier({
      apiKey: 'hive-key-0123456789abcd',
      url: 'https://hive.test',
      timeoutMs: 1000,
      retries: 0,
      fetchImpl,
    });
    const b = budget();
    const d = deps({ verifier: hive, budget: b });
    expect(await runExternalCheck(d, req())).toMatchObject({
      status: 'quota',
      reason: 'hive_rate_limited',
    });
    expect(await runExternalCheck(d, req({ sha256: 'b'.repeat(64) }))).toMatchObject({
      status: 'quota',
      reason: 'budget:exhausted',
    });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect((await b.usage()).exhausted).toBe(true);
  });

  it('records "disabled" when the second check is off, without any network use', async ({
    skip,
  }) => {
    if (!available) skip();
    const d = deps({ verifier: null });
    expect(await runExternalCheck(d, req())).toMatchObject({
      status: 'error',
      reason: 'disabled',
      provider: 'hive',
    });
  });

  it('works in mock mode with no key and no network', async ({ skip }) => {
    if (!available) skip();
    const d = deps({ verifier: new MockHiveVerifier() });
    const r = await runExternalCheck(d, req());
    expect(r).toMatchObject({ status: 'ok', provider: 'hive-mock' });
  });

  it('releases the lock when the provider throws', async ({ skip }) => {
    if (!available) skip();
    const lock = new InflightLock(redis, `x-${randomUUID()}`);
    const broken: Verifier = {
      provider: 'hive',
      verify: async () => {
        throw new Error('boom');
      },
    };
    const d = deps({ verifier: broken, lock });
    expect(await runExternalCheck(d, req())).toMatchObject({
      status: 'error',
      reason: 'client_error',
    });
    expect(await lock.isHeld(`hive:${'a'.repeat(64)}`)).toBe(false);
  });

  it('never throws to the pipeline: storage or Redis failures become null', async ({ skip }) => {
    if (!available) skip();
    const onError = vi.fn();
    const store: CheckStore = {
      findCachedOk: async () => {
        throw new Error('mongo down');
      },
      create: async () => {
        throw new Error('x');
      },
    };
    expect(await safeExternalCheck(deps({ store }), req(), onError)).toBeNull();
    expect(onError).toHaveBeenCalledOnce();
    const badRedis = new Redis('redis://127.0.0.1:1', {
      lazyConnect: true,
      maxRetriesPerRequest: 0,
      connectTimeout: 200,
      retryStrategy: () => null,
    });
    badRedis.on('error', () => undefined);
    const d = deps({ lock: new InflightLock(badRedis, 'x') });
    expect(await safeExternalCheck(d, req(), () => undefined)).toBeNull();
    badRedis.disconnect();
  });

  it('feeds the reconciler unchanged: an unavailable check gives single_check', async ({
    skip,
  }) => {
    if (!available) skip();
    const d = deps({ verifier: null });
    const ext = await runExternalCheck(d, req());
    const out = reconcile({
      ours: { verdict: 'FAKE', confident: true },
      external: { status: ext!.status, deepfakeScore: ext!.deepfakeScore },
      hiveThreshold: 0.9,
    });
    expect(out.badge).toBe('single_check');
  });
});
