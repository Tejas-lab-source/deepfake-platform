/**
 * BullMQ wiring for the analysis and explanation jobs. The job logic is in process.ts; this
 * file builds its production dependencies (MongoDB, Redis budgets, Hive/Groq clients chosen by
 * HIVE_MODE/GROQ_MODE) and maps failures onto bounded retries:
 *   - PermanentJobError (bad media, expired upload, unsupported) fails at once;
 *   - anything else is retried up to JOB_ATTEMPTS times with backoff, then fails.
 * A failure is only recorded on the analysis when no retry is left.
 */
import { UnrecoverableError, Worker, type Job } from 'bullmq';
import { Redis } from 'ioredis';
import { DailyBudget, InflightLock } from '../budget.js';
import { config } from '../config.js';
import type { Repos } from '../data/types.js';
import { explainResult, type ExplainerDeps, type StoredExplanation } from '../explain/explainer.js';
import { GROQ_PROMPT_VERSION } from '../explain/prompt.js';
import { logger } from '../logger.js';
import { ExternalCheck } from '../models/index.js';
import {
  safeExternalCheck,
  type CheckStore,
  type ExternalCheckDeps,
} from '../providers/externalCheck.js';
import { GroqClient } from '../providers/groq.js';
import { HiveVerifier } from '../providers/hive.js';
import { MockHiveVerifier } from '../providers/hiveMock.js';
import type { MlApi } from '../providers/mlClient.js';
import type { FileStore } from '../storage.js';
import {
  failAnalysis,
  PermanentJobError,
  processAnalysis,
  processExplanation,
  setStage,
  type ProcessDeps,
  type ProgressEvent,
} from './process.js';
import {
  ANALYSIS_QUEUE,
  EXPLAIN_QUEUE,
  getExplainQueue,
  newRedis,
  type AnalysisJob,
} from './queues.js';

const checkStore: CheckStore = {
  findCachedOk: (sha256, provider) =>
    ExternalCheck.findOne({ mediaSha256: sha256, provider, status: 'ok' })
      .sort({ createdAt: -1 })
      .lean() as never,
  create: async (doc) => (await ExternalCheck.create(doc)).toObject() as never,
};

let budgetRedis: Redis | null = null;

/** One fail-fast Redis client for budgets and locks: a Redis outage cannot hang a job. */
function sharedBudgetRedis(): Redis {
  if (!budgetRedis) {
    budgetRedis = new Redis(config.REDIS_URL, { maxRetriesPerRequest: 1, connectTimeout: 2000 });
    budgetRedis.on('error', (err) => logger.warn({ err }, 'budget redis error'));
  }
  return budgetRedis;
}

let externalDeps: ExternalCheckDeps | null = null;

/** Built on first use from HIVE_MODE: mock (free, no network), real (Hive v3, budgeted) or off. */
function externalCheckDeps(): ExternalCheckDeps {
  if (externalDeps) return externalDeps;
  const redis = sharedBudgetRedis();
  const real = config.HIVE_MODE === 'real';
  externalDeps = {
    verifier:
      config.HIVE_MODE === 'mock'
        ? new MockHiveVerifier()
        : real
          ? new HiveVerifier({
              apiKey: config.HIVE_API_KEY,
              url: config.HIVE_API_URL,
              timeoutMs: config.HIVE_TIMEOUT_MS,
              retries: config.HIVE_RETRIES,
            })
          : null,
    provider: 'hive',
    budget: real
      ? new DailyBudget(redis, 'hive', {
          daily: config.HIVE_DAILY_LIMIT,
          reserve: config.HIVE_DEMO_RESERVE,
          perUser: config.HIVE_USER_DAILY_LIMIT,
        })
      : null,
    lock: new InflightLock(redis),
    store: checkStore,
    threshold: config.HIVE_THRESHOLD,
    timeoutMs: config.HIVE_TIMEOUT_MS * (config.HIVE_RETRIES + 1) + 5_000,
  };
  return externalDeps;
}

/** Built from GROQ_MODE: real (Groq, budgeted, validated), or mock/off: the template. */
function explanationDeps(repos: Repos): ExplainerDeps {
  const models = [config.GROQ_MODEL_EXPLAIN, ...config.GROQ_MODEL_FALLBACKS].filter(
    (m, i, all) => all.indexOf(m) === i,
  );
  const real = config.GROQ_MODE === 'real';
  return {
    mode: config.GROQ_MODE,
    models,
    groq: real
      ? new GroqClient({
          apiKey: config.GROQ_API_KEY,
          baseUrl: config.GROQ_BASE_URL,
          models,
          timeoutMs: config.GROQ_TIMEOUT_MS,
          retries: config.GROQ_RETRIES,
          maxCompletionTokens: config.GROQ_MAX_COMPLETION_TOKENS,
          reasoningEffort: config.GROQ_REASONING_EFFORT,
          temperature: config.GROQ_TEMPERATURE,
        })
      : null,
    budget: real
      ? new DailyBudget(sharedBudgetRedis(), 'groq', {
          daily: config.GROQ_DAILY_LIMIT,
          reserve: 0,
          perUser: -1,
        })
      : null,
    cache: {
      find: async (sha) =>
        (await repos.explanations.findCachedGroq(
          sha,
          GROQ_PROMPT_VERSION,
        )) as StoredExplanation | null,
    },
  };
}

export interface WorkerWiring {
  repos: Repos;
  files: FileStore;
  ml: MlApi;
  emit: (event: ProgressEvent) => void;
}

export function productionProcessDeps(w: WorkerWiring): ProcessDeps {
  const explainer = explanationDeps(w.repos);
  return {
    repos: w.repos,
    files: w.files,
    ml: w.ml,
    emit: w.emit,
    hiveThreshold: config.HIVE_THRESHOLD,
    progressPollMs: 750,
    externalCheck: (req) =>
      safeExternalCheck(externalCheckDeps(), { ...req, consent: true }, (err) =>
        logger.warn({ err }, 'external check failed'),
      ),
    explain: (input, extras) =>
      explainResult(explainer, input, extras, (err) =>
        logger.warn({ err }, 'explanation failed; template used'),
      ),
    enqueueExplain: async (analysisId) => {
      await getExplainQueue().add(
        'explain',
        { analysisId },
        { jobId: `explain-${analysisId}-${Date.now()}`, attempts: config.JOB_ATTEMPTS },
      );
    },
  };
}

/** Run a step; permanent errors skip BullMQ's retries. */
const step =
  (d: ProcessDeps, fn: (d: ProcessDeps, id: string) => Promise<void>) =>
  async (job: Job<AnalysisJob>): Promise<void> => {
    try {
      await fn(d, job.data.analysisId);
    } catch (err) {
      if (err instanceof PermanentJobError) throw new UnrecoverableError(err.message);
      throw err;
    }
  };

/** Called for every failed attempt: record a failure only when no retry is left. */
async function onFailed(
  d: ProcessDeps,
  job: Job<AnalysisJob> | undefined,
  err: Error,
): Promise<void> {
  if (!job) return;
  const { analysisId } = job.data;
  const attempts = job.opts.attempts ?? 1;
  const final = err.name === 'UnrecoverableError' || job.attemptsMade >= attempts;
  logger.warn(
    { analysisId, queue: job.queueName, attempt: job.attemptsMade, final, reason: err.message },
    final ? 'job failed' : 'job attempt failed; retrying',
  );
  if (final) {
    await failAnalysis(d, analysisId, err.message).catch((e: unknown) =>
      logger.error({ err: e, analysisId }, 'could not record the failure'),
    );
  } else {
    await setStage(d, analysisId, 'queued').catch(() => undefined);
  }
}

export function startWorkers(d: ProcessDeps): { close: () => Promise<void> } {
  const opts = { concurrency: config.WORKER_CONCURRENCY };
  const workers = [
    new Worker<AnalysisJob>(ANALYSIS_QUEUE, step(d, processAnalysis), {
      ...opts,
      connection: newRedis(),
    }),
    new Worker<AnalysisJob>(EXPLAIN_QUEUE, step(d, processExplanation), {
      ...opts,
      connection: newRedis(),
    }),
  ];
  for (const worker of workers) {
    worker.on('failed', (job, err) => void onFailed(d, job, err));
    worker.on('error', (err) => logger.error({ err }, 'worker error'));
  }
  return { close: async () => void (await Promise.all(workers.map((w) => w.close()))) };
}
