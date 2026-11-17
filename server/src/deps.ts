/**
 * Everything the HTTP app needs from the outside world, injected so routes can be tested
 * without MongoDB, Redis or the ML service. Production wiring lives in productionDeps().
 */
import type { Redis } from 'ioredis';
import { config } from './config.js';
import { mongoRepos } from './data/mongo.js';
import type { Repos } from './data/types.js';
import { configLimits, type UploadLimits } from './middleware/upload.js';
import { configuredMlClient, type MlApi } from './providers/mlClient.js';
import { getAnalysisQueue } from './queue/queues.js';
import { redisRateLimiter, type RateLimiter } from './rateLimit.js';
import { diskFileStore, type FileStore } from './storage.js';

export interface AnalysisQueue {
  /** Queue an analysis; returns the job id. Idempotent per analysis. */
  enqueue(analysisId: string): Promise<string>;
}

export interface RateLimiters {
  /** Login and registration attempts, per client address. */
  auth: RateLimiter;
  /** Uploads, per user. */
  upload: RateLimiter;
  /** Questions to /ask, per user. */
  ask: RateLimiter;
  /** Filter Lab comparisons (expensive ML calls), per user. */
  compare: RateLimiter;
}

export interface AppDeps {
  repos: Repos;
  files: FileStore;
  ml: MlApi;
  queue: AnalysisQueue;
  limits: UploadLimits;
  rateLimiters: RateLimiters;
  /** Allowed browser origins for CORS; empty means same-origin only. */
  corsOrigins: string[];
  mockMode: boolean;
}

export function productionRateLimiters(redis: Redis): RateLimiters {
  return {
    auth: redisRateLimiter(redis, 'auth', config.RATE_LIMIT_AUTH_PER_15MIN, 15 * 60),
    upload: redisRateLimiter(redis, 'upload', config.RATE_LIMIT_UPLOADS_PER_HOUR, 3600),
    ask: redisRateLimiter(redis, 'ask', config.RATE_LIMIT_ASK_PER_HOUR, 3600),
    compare: redisRateLimiter(redis, 'compare', config.RATE_LIMIT_COMPARE_PER_HOUR, 3600),
  };
}

export function productionDeps(redis: Redis): AppDeps {
  const queue = getAnalysisQueue();
  return {
    repos: mongoRepos(),
    files: diskFileStore(config.UPLOAD_DIR),
    ml: configuredMlClient(),
    queue: {
      async enqueue(analysisId) {
        const job = await queue.add('analyze', { analysisId }, { jobId: `analysis-${analysisId}` });
        return String(job.id);
      },
    },
    limits: configLimits(),
    rateLimiters: productionRateLimiters(redis),
    corsOrigins: config.CORS_ORIGINS,
    mockMode: config.MOCK_MODE,
  };
}
