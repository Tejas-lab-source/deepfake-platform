import { Queue } from 'bullmq';
import { Redis } from 'ioredis';
import { config } from '../config.js';

export const ANALYSIS_QUEUE = 'analysis';
export const EXPLAIN_QUEUE = 'explain';

export interface AnalysisJob {
  analysisId: string;
}

/** BullMQ needs maxRetriesPerRequest = null on its connections. */
export function newRedis(): Redis {
  return new Redis(config.REDIS_URL, { maxRetriesPerRequest: null });
}

// Bounded retries with backoff for transient failures (ML service restarting, Redis blips).
const jobDefaults = {
  removeOnComplete: 1000,
  removeOnFail: 1000,
  attempts: config.JOB_ATTEMPTS,
  backoff: { type: 'exponential' as const, delay: 15_000 },
};

let analysisQueue: Queue<AnalysisJob> | null = null;
let explainQueue: Queue<AnalysisJob> | null = null;

export function getAnalysisQueue(): Queue<AnalysisJob> {
  analysisQueue ??= new Queue<AnalysisJob>(ANALYSIS_QUEUE, {
    connection: newRedis(),
    defaultJobOptions: jobDefaults,
  });
  return analysisQueue;
}

export function getExplainQueue(): Queue<AnalysisJob> {
  explainQueue ??= new Queue<AnalysisJob>(EXPLAIN_QUEUE, {
    connection: newRedis(),
    defaultJobOptions: jobDefaults,
  });
  return explainQueue;
}

export async function closeQueues(): Promise<void> {
  await Promise.all([analysisQueue?.close(), explainQueue?.close()]);
}
