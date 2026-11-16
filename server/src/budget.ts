/**
 * Daily API budgets and the in-flight lock, in Redis.
 *
 * Reservations are atomic (one Lua script), so concurrent requests can never overspend.
 * A reservation is counted when a request is about to be sent and is never refunded: the
 * providers' billing for failed requests is not documented, so the safe assumption is that
 * every attempt costs. Cache hits never reach this code.
 *
 * Keys (day = UTC date, all expire after 48 h):
 *   budget:{name}:{day}:used        requests sent today
 *   budget:{name}:{day}:demo        of which from the demo reserve
 *   budget:{name}:{day}:user:{id}   per user (non-demo only)
 *   budget:{name}:{day}:exhausted   set when the provider itself answered 429
 */
import { randomUUID } from 'node:crypto';
import type { Redis } from 'ioredis';

export interface BudgetLimits {
  /** Requests per UTC day in total. */
  daily: number;
  /** Part of `daily` only the demo pool may use. */
  reserve: number;
  /** Requests per user per day outside the demo pool; -1 for no per-user cap. */
  perUser: number;
}

export type DenyReason = 'exhausted' | 'daily_limit' | 'user_limit';
export type Reservation =
  { allowed: true; used: number } | { allowed: false; reason: DenyReason; used: number };

const TTL_SECONDS = 48 * 3600;

const RESERVE = `
local used = tonumber(redis.call('GET', KEYS[1]) or '0')
if redis.call('EXISTS', KEYS[4]) == 1 then return {0, 'exhausted', used} end
local daily, reserve, perUser = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
local ttl = tonumber(ARGV[6])
if ARGV[4] == '1' then
  if used >= daily then return {0, 'daily_limit', used} end
  redis.call('INCR', KEYS[3])
  redis.call('EXPIRE', KEYS[3], ttl)
else
  if used >= daily - reserve then return {0, 'daily_limit', used} end
  if ARGV[5] == '1' and perUser >= 0 then
    local u = tonumber(redis.call('GET', KEYS[2]) or '0')
    if u >= perUser then return {0, 'user_limit', used} end
    redis.call('INCR', KEYS[2])
    redis.call('EXPIRE', KEYS[2], ttl)
  end
end
used = redis.call('INCR', KEYS[1])
redis.call('EXPIRE', KEYS[1], ttl)
return {1, 'ok', used}
`;

export function utcDay(date: Date): string {
  return date.toISOString().slice(0, 10);
}

export class DailyBudget {
  constructor(
    private readonly redis: Redis,
    readonly name: string,
    readonly limits: BudgetLimits,
    private readonly now: () => Date = () => new Date(),
    private readonly prefix = 'budget',
  ) {}

  private keys(userId: string | undefined) {
    const base = `${this.prefix}:${this.name}:${utcDay(this.now())}`;
    return {
      used: `${base}:used`,
      user: `${base}:user:${userId ?? '-'}`,
      demo: `${base}:demo`,
      exhausted: `${base}:exhausted`,
    };
  }

  /** Atomically take one request from today's budget. */
  async reserve(who: { userId?: string; demo?: boolean }): Promise<Reservation> {
    const k = this.keys(who.userId);
    const result = (await this.redis.eval(
      RESERVE,
      4,
      k.used,
      k.user,
      k.demo,
      k.exhausted,
      String(this.limits.daily),
      String(this.limits.reserve),
      String(this.limits.perUser),
      who.demo ? '1' : '0',
      who.userId ? '1' : '0',
      String(TTL_SECONDS),
    )) as [number, string, number];
    const [allowed, reason, used] = result;
    return allowed === 1
      ? { allowed: true, used: Number(used) }
      : { allowed: false, reason: reason as DenyReason, used: Number(used) };
  }

  /** The provider refused (HTTP 429): stop sending until the next UTC day. */
  async markExhausted(): Promise<void> {
    await this.redis.set(this.keys(undefined).exhausted, '1', 'EX', TTL_SECONDS);
  }

  async usage(userId?: string): Promise<{
    day: string;
    used: number;
    demoUsed: number;
    userUsed: number | null;
    exhausted: boolean;
    limits: BudgetLimits;
  }> {
    const k = this.keys(userId);
    const [used, demo, user, exhausted] = await this.redis.mget(
      k.used,
      k.demo,
      k.user,
      k.exhausted,
    );
    return {
      day: utcDay(this.now()),
      used: Number(used ?? 0),
      demoUsed: Number(demo ?? 0),
      userUsed: userId ? Number(user ?? 0) : null,
      exhausted: exhausted === '1',
      limits: this.limits,
    };
  }
}

const RELEASE = `
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0
`;

/** "Only one request per file at a time": a token lock with an expiry, so a crashed worker
 * cannot hold it forever and only the holder can release it. */
export class InflightLock {
  constructor(
    private readonly redis: Redis,
    private readonly prefix = 'lock',
  ) {}

  async acquire(key: string, ttlMs: number): Promise<string | null> {
    const token = randomUUID();
    const ok = await this.redis.set(`${this.prefix}:${key}`, token, 'PX', ttlMs, 'NX');
    return ok === 'OK' ? token : null;
  }

  async release(key: string, token: string): Promise<boolean> {
    return (await this.redis.eval(RELEASE, 1, `${this.prefix}:${key}`, token)) === 1;
  }

  async isHeld(key: string): Promise<boolean> {
    return (await this.redis.exists(`${this.prefix}:${key}`)) === 1;
  }
}
