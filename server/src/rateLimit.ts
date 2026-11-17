/**
 * Fixed-window rate limiting. The Redis limiter is shared by every server process; the memory
 * limiter is for a single process (and tests). Both answer "allowed?" plus when to retry.
 */
import type { RequestHandler } from 'express';
import type { Redis } from 'ioredis';
import { logger } from './logger.js';
import { HttpProblem } from './middleware/problem.js';

export interface RateDecision {
  allowed: boolean;
  remaining: number;
  retryAfterSec: number;
}

export interface RateLimiter {
  readonly limit: number;
  readonly windowSec: number;
  hit(key: string): Promise<RateDecision>;
}

// INCR and set the expiry on the first hit, atomically; returns [count, ttl].
const HIT = `
local n = redis.call('INCR', KEYS[1])
if n == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
local ttl = redis.call('TTL', KEYS[1])
if ttl < 0 then redis.call('EXPIRE', KEYS[1], ARGV[1]); ttl = tonumber(ARGV[1]) end
return {n, ttl}
`;

export function redisRateLimiter(
  redis: Redis,
  name: string,
  limit: number,
  windowSec: number,
): RateLimiter {
  return {
    limit,
    windowSec,
    async hit(key) {
      const [count, ttl] = (await redis.eval(HIT, 1, `ratelimit:${name}:${key}`, windowSec)) as [
        number,
        number,
      ];
      return {
        allowed: count <= limit,
        remaining: Math.max(0, limit - count),
        retryAfterSec: Math.max(1, ttl),
      };
    },
  };
}

export function memoryRateLimiter(
  limit: number,
  windowSec: number,
  now: () => number = Date.now,
): RateLimiter {
  const windows = new Map<string, { count: number; resetAt: number }>();
  return {
    limit,
    windowSec,
    async hit(key) {
      const t = now();
      let w = windows.get(key);
      if (!w || w.resetAt <= t) {
        w = { count: 0, resetAt: t + windowSec * 1000 };
        windows.set(key, w);
      }
      w.count++;
      return {
        allowed: w.count <= limit,
        remaining: Math.max(0, limit - w.count),
        retryAfterSec: Math.max(1, Math.ceil((w.resetAt - t) / 1000)),
      };
    },
  };
}

/**
 * Express middleware: one counter per user (or per client IP before login). If the limiter's
 * store is down, the request is allowed and the outage is logged: rate limiting must not take
 * the API offline.
 */
export function rateLimit(
  limiter: RateLimiter,
  keyOf: (req: Parameters<RequestHandler>[0]) => string,
): RequestHandler {
  return (req, res, next) => {
    limiter
      .hit(keyOf(req))
      .then((d) => {
        res.setHeader('RateLimit-Limit', String(limiter.limit));
        res.setHeader('RateLimit-Remaining', String(d.remaining));
        if (d.allowed) return next();
        res.setHeader('Retry-After', String(d.retryAfterSec));
        next(
          new HttpProblem(
            429,
            'Too Many Requests',
            `Rate limit reached; try again in ${d.retryAfterSec} s.`,
          ),
        );
      })
      .catch((err: unknown) => {
        logger.warn({ err }, 'rate limiter unavailable; request allowed');
        next();
      });
  };
}

/** The authenticated user's id, or the client address before login. */
export const userOrIp = (req: Parameters<RequestHandler>[0]): string =>
  req.user?.id ?? `ip:${req.ip ?? 'unknown'}`;
