import { randomUUID } from 'node:crypto';
import { Redis } from 'ioredis';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { DailyBudget, InflightLock, utcDay } from '../src/budget.js';

// These tests need a real Redis (atomicity is the point). They use database 15 and a random
// key prefix per test, and are skipped when no Redis is reachable.
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
    const keys = await redis.keys('t-*');
    if (keys.length) await redis.del(...keys);
  }
  redis.disconnect();
});

const day = new Date('2026-11-16T12:00:00Z');
const hive = (now = () => day) =>
  new DailyBudget(
    redis,
    'hive',
    { daily: 100, reserve: 20, perUser: 10 },
    now,
    `t-${randomUUID()}`,
  );

describe.runIf(process.env.SKIP_REDIS_TESTS !== '1')('DailyBudget (real Redis)', () => {
  it('keeps the reserve for the demo pool and caps each user', async ({ skip }) => {
    if (!available) skip();
    const b = hive();
    // Eight users x 10 = 80 regular requests: the regular share (100 - 20).
    for (let u = 0; u < 8; u++) {
      for (let i = 0; i < 10; i++)
        expect((await b.reserve({ userId: `u${u}` })).allowed).toBe(true);
      expect(await b.reserve({ userId: `u${u}` })).toMatchObject({ allowed: false });
    }
    expect(await b.reserve({ userId: 'fresh' })).toEqual({
      allowed: false,
      reason: 'daily_limit',
      used: 80,
    });
    // The demo pool can still use the remaining 20, and no more.
    for (let i = 0; i < 20; i++)
      expect((await b.reserve({ userId: 'admin', demo: true })).allowed).toBe(true);
    expect(await b.reserve({ userId: 'admin', demo: true })).toEqual({
      allowed: false,
      reason: 'daily_limit',
      used: 100,
    });
    expect(await b.usage()).toMatchObject({ used: 100, demoUsed: 20, exhausted: false });
  });

  it('a user over the cap is refused without spending budget', async ({ skip }) => {
    if (!available) skip();
    const b = hive();
    for (let i = 0; i < 10; i++) await b.reserve({ userId: 'u' });
    expect(await b.reserve({ userId: 'u' })).toEqual({
      allowed: false,
      reason: 'user_limit',
      used: 10,
    });
    expect(await b.reserve({ userId: 'u' })).toMatchObject({ reason: 'user_limit' });
    expect((await b.usage('u')).used).toBe(10);
    expect((await b.usage('u')).userUsed).toBe(10);
    expect((await b.reserve({ userId: 'v' })).allowed).toBe(true);
  });

  it('never overspends under concurrency', async ({ skip }) => {
    if (!available) skip();
    const b = new DailyBudget(
      redis,
      'hive',
      { daily: 100, reserve: 20, perUser: -1 },
      () => day,
      `t-${randomUUID()}`,
    );
    const regular = Array.from({ length: 150 }, (_, i) => b.reserve({ userId: `u${i % 7}` }));
    const demo = Array.from({ length: 50 }, () => b.reserve({ userId: 'admin', demo: true }));
    const results = await Promise.all([...regular, ...demo]);
    const regularAllowed = results.slice(0, 150).filter((r) => r.allowed).length;
    const demoAllowed = results.slice(150).filter((r) => r.allowed).length;
    expect(regularAllowed).toBeLessThanOrEqual(80);
    expect(regularAllowed + demoAllowed).toBe(100);
    expect((await b.usage()).used).toBe(100);
  });

  it('stops everything once the provider says it is exhausted', async ({ skip }) => {
    if (!available) skip();
    const b = hive();
    expect((await b.reserve({ userId: 'u' })).allowed).toBe(true);
    await b.markExhausted();
    expect(await b.reserve({ userId: 'u' })).toMatchObject({ allowed: false, reason: 'exhausted' });
    expect(await b.reserve({ userId: 'admin', demo: true })).toMatchObject({ reason: 'exhausted' });
    expect((await b.usage()).exhausted).toBe(true);
  });

  it('starts afresh each UTC day and lets keys expire', async ({ skip }) => {
    if (!available) skip();
    let now = new Date('2026-11-16T23:59:59Z');
    const b = new DailyBudget(
      redis,
      'hive',
      { daily: 1, reserve: 0, perUser: -1 },
      () => now,
      `t-${randomUUID()}`,
    );
    expect((await b.reserve({})).allowed).toBe(true);
    expect((await b.reserve({})).allowed).toBe(false);
    now = new Date('2026-11-17T00:00:01Z');
    expect((await b.reserve({})).allowed).toBe(true);
    expect(utcDay(now)).toBe('2026-11-17');
    const keys = await redis.keys('t-*:hive:2026-11-17:used');
    const ttl = await redis.ttl(keys[0]!);
    expect(ttl).toBeGreaterThan(47 * 3600);
  });

  it('works without a reserve or per-user cap (Groq)', async ({ skip }) => {
    if (!available) skip();
    const g = new DailyBudget(
      redis,
      'groq',
      { daily: 3, reserve: 0, perUser: -1 },
      () => day,
      `t-${randomUUID()}`,
    );
    const r = await Promise.all([1, 2, 3, 4].map(() => g.reserve({ userId: 'u' })));
    expect(r.filter((x) => x.allowed)).toHaveLength(3);
  });
});

describe.runIf(process.env.SKIP_REDIS_TESTS !== '1')('InflightLock (real Redis)', () => {
  it('lets one holder in, releases only with its token, and expires', async ({ skip }) => {
    if (!available) skip();
    const lock = new InflightLock(redis, `t-${randomUUID()}`);
    const token = await lock.acquire('sha', 200);
    expect(token).toBeTruthy();
    expect(await lock.acquire('sha', 200)).toBeNull();
    expect(await lock.release('sha', 'wrong-token')).toBe(false);
    expect(await lock.isHeld('sha')).toBe(true);
    expect(await lock.release('sha', token!)).toBe(true);
    expect(await lock.acquire('sha', 50)).toBeTruthy();
    await new Promise((r) => setTimeout(r, 120));
    expect(await lock.isHeld('sha')).toBe(false);
  });

  it('only one of many concurrent callers gets the lock', async ({ skip }) => {
    if (!available) skip();
    const lock = new InflightLock(redis, `t-${randomUUID()}`);
    const tokens = await Promise.all(
      Array.from({ length: 30 }, () => lock.acquire('same-file', 1000)),
    );
    expect(tokens.filter(Boolean)).toHaveLength(1);
  });
});
