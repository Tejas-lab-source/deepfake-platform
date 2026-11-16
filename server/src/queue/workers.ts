import { Worker, type Job } from 'bullmq';
import { config } from '../config.js';
import { explainResult, type ExplainerDeps, type StoredExplanation } from '../explain/explainer.js';
import { GROQ_PROMPT_VERSION } from '../explain/prompt.js';
import { logger } from '../logger.js';
import { storagePath } from '../middleware/upload.js';
import { Analysis, Explanation, ExternalCheck, Media, User } from '../models/index.js';
import { Redis } from 'ioredis';
import { DailyBudget, InflightLock } from '../budget.js';
import { GroqClient } from '../providers/groq.js';
import { HiveVerifier } from '../providers/hive.js';
import { MockHiveVerifier } from '../providers/hiveMock.js';
import {
  safeExternalCheck,
  type CheckStore,
  type ExternalCheckDeps,
} from '../providers/externalCheck.js';
import { analyzeMedia } from '../providers/mlClient.js';
import { reconcile, type ExternalStatus } from '../reconcile.js';
import { emitProgress } from '../sockets.js';
import type { Stage } from '../types.js';
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

/** Built on first use from HIVE_MODE: mock (free, no network), real (Hive v3, budgeted) or
 * off. The budget's Redis client fails fast, so a Redis outage cannot hang a job. */
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

let explainerDeps: ExplainerDeps | null = null;

/** Built on first use from GROQ_MODE: real (Groq, budgeted, validated), or mock/off, which
 * both use the template explanation. */
function explanationDeps(): ExplainerDeps {
  if (explainerDeps) return explainerDeps;
  const models = [config.GROQ_MODEL_EXPLAIN, ...config.GROQ_MODEL_FALLBACKS].filter(
    (m, i, all) => all.indexOf(m) === i,
  );
  const real = config.GROQ_MODE === 'real';
  explainerDeps = {
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
        (await Explanation.findOne({
          inputSha256: sha,
          fallbackUsed: false,
          promptVersion: GROQ_PROMPT_VERSION,
        })
          .sort({ createdAt: -1 })
          .select('-_id -analysisId -createdAt -updatedAt -__v')
          .lean()) as StoredExplanation | null,
    },
  };
  return explainerDeps;
}

async function setStage(analysisId: string, stage: Stage): Promise<void> {
  const status = stage === 'done' ? 'done' : stage === 'failed' ? 'failed' : 'processing';
  await Analysis.updateOne({ _id: analysisId }, { stage, status });
  emitProgress({ analysisId, stage });
}

async function runAnalysis(job: Job<AnalysisJob>): Promise<void> {
  const { analysisId } = job.data;
  const analysis = await Analysis.findById(analysisId);
  if (!analysis) return; // deleted while queued
  const media = await Media.findById(analysis.mediaId);
  if (!media) throw new Error('The uploaded file has expired or was deleted.');

  await setStage(analysisId, 'detecting_faces');
  // Our pipeline and the second opinion run in parallel.
  const mlPromise = analyzeMedia(media.kind, media.storageKey);
  // The second check only runs with the uploader's explicit consent for this upload.
  const consent = analysis.externalCheckRequested === true;
  const externalPromise = consent
    ? (async () => {
        const owner = await User.findById(analysis.ownerId).select('role').lean();
        return safeExternalCheck(
          externalCheckDeps(),
          {
            sha256: media.sha256,
            filePath: storagePath(media.storageKey),
            mime: media.mime,
            userId: String(analysis.ownerId),
            demo: owner?.role === 'admin',
            consent,
          },
          (err) => logger.warn({ err, analysisId }, 'external check failed'),
        );
      })().catch((err: unknown) => {
        logger.warn({ err, analysisId }, 'external check failed');
        return null;
      })
    : Promise.resolve(null);
  await setStage(analysisId, 'scoring');
  const ml = await mlPromise;

  await setStage(analysisId, 'verifying');
  const external = await externalPromise;
  const externalStatus: ExternalStatus = !analysis.externalCheckRequested
    ? 'opted_out'
    : (external?.status ?? 'error');
  const final = reconcile({
    ours: { verdict: ml.verdict, confident: ml.confident },
    external: { status: externalStatus, deepfakeScore: external?.deepfakeScore ?? null },
    hiveThreshold: config.HIVE_THRESHOLD,
  });

  const face = ml.faces?.[0];
  await Analysis.updateOne(
    { _id: analysisId },
    {
      final: { verdict: final.verdict, badge: final.badge, ruleApplied: final.ruleApplied },
      ours: {
        verdict: ml.verdict,
        probability: ml.probability,
        threshold: ml.threshold,
        confident: ml.confident,
        uncertainReason: ml.uncertainReason,
        detectorScores: face?.scores ?? null,
        spread: face?.spread ?? null,
      },
      externalCheckId: external?._id ?? null,
      faces: ml.faces ?? [],
      frames: ml.frames ?? [],
      modelCardVersion: ml.modelCardVersion,
      ml,
    },
  );
  await setStage(analysisId, 'explaining');
  await getExplainQueue().add('explain', { analysisId }, { jobId: `explain-${analysisId}` });
}

async function runExplain(job: Job<AnalysisJob>): Promise<void> {
  const { analysisId } = job.data;
  const analysis = await Analysis.findById(analysisId).lean();
  if (!analysis?.final || !analysis.ours) return;
  const external = analysis.externalCheckId
    ? await ExternalCheck.findById(analysis.externalCheckId).lean()
    : null;

  const ml = analysis.ml as {
    mock?: boolean;
    faces?: Array<{ scores?: { gendHead?: number; sbi?: number } }>;
  } | null;
  const input = {
    final: analysis.final as never,
    ours: {
      verdict: analysis.ours.verdict as never,
      probability: analysis.ours.probability ?? null,
      threshold: analysis.ours.threshold ?? 0.5,
      confident: analysis.ours.confident ?? false,
      uncertainReason: analysis.ours.uncertainReason ?? null,
      spread: analysis.ours.spread ?? null,
    },
    external: external
      ? {
          status: external.status,
          deepfakeScore: external.deepfakeScore ?? null,
          threshold: config.HIVE_THRESHOLD,
          verdict:
            external.status === 'ok' && typeof external.deepfakeScore === 'number'
              ? external.deepfakeScore >= config.HIVE_THRESHOLD
                ? ('FAKE' as const)
                : ('REAL' as const)
              : null,
        }
      : null,
    mock: Boolean(ml?.mock),
  };
  // Groq sees only the whitelisted result JSON built from these values (explain/prompt.ts).
  const explanation = await explainResult(
    explanationDeps(),
    input,
    {
      ruleApplied: analysis.final.ruleApplied ?? null,
      detectorScores: ml?.faces?.[0]?.scores ?? null,
      aiGeneratedScore: external?.aiGeneratedScore ?? null,
      provider: external?.provider ?? null,
      modelCardVersion: analysis.modelCardVersion ?? null,
    },
    (err) => logger.warn({ err, analysisId }, 'explanation failed; template used'),
  );
  await Explanation.findOneAndUpdate({ analysisId }, explanation, { upsert: true });
  await setStage(analysisId, 'done');
}

async function markFailed(job: Job<AnalysisJob> | undefined, err: Error): Promise<void> {
  if (!job) return;
  const { analysisId } = job.data;
  logger.error({ err, analysisId, queue: job.queueName }, 'job failed');
  await Analysis.updateOne(
    { _id: analysisId },
    { status: 'failed', stage: 'failed', error: err.message },
  );
  emitProgress({ analysisId, stage: 'failed', detail: err.message });
}

export function startWorkers(): { close: () => Promise<void> } {
  const opts = { concurrency: config.WORKER_CONCURRENCY };
  const workers = [
    new Worker<AnalysisJob>(ANALYSIS_QUEUE, runAnalysis, { ...opts, connection: newRedis() }),
    new Worker<AnalysisJob>(EXPLAIN_QUEUE, runExplain, { ...opts, connection: newRedis() }),
  ];
  for (const worker of workers) {
    worker.on('failed', (job, err) => void markFailed(job, err));
    worker.on('error', (err) => logger.error({ err }, 'worker error'));
  }
  return { close: async () => void (await Promise.all(workers.map((w) => w.close()))) };
}
