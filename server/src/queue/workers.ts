import { Worker, type Job } from 'bullmq';
import { config } from '../config.js';
import { buildTemplateExplanation, TEMPLATE_PROMPT_VERSION } from '../explainTemplate.js';
import { logger } from '../logger.js';
import { storagePath } from '../middleware/upload.js';
import { Analysis, Explanation, ExternalCheck, Media } from '../models/index.js';
import { MockHiveVerifier, UnavailableVerifier } from '../providers/hiveMock.js';
import { analyzeMedia } from '../providers/mlClient.js';
import { verifyWithTimeout, type Verifier } from '../providers/verifier.js';
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

const verifier: Verifier = config.MOCK_MODE ? new MockHiveVerifier() : new UnavailableVerifier();

async function setStage(analysisId: string, stage: Stage): Promise<void> {
  const status = stage === 'done' ? 'done' : stage === 'failed' ? 'failed' : 'processing';
  await Analysis.updateOne({ _id: analysisId }, { stage, status });
  emitProgress({ analysisId, stage });
}

/** Ask the second-opinion provider, reusing a stored answer for the same file if there is one. */
async function externalCheck(sha256: string, filePath: string) {
  const cached = await ExternalCheck.findOne({
    mediaSha256: sha256,
    provider: verifier.provider,
    status: 'ok',
  });
  if (cached) return cached;
  const started = Date.now();
  const result = await verifyWithTimeout(verifier, filePath, config.HIVE_TIMEOUT_MS);
  const verdict =
    result.status === 'ok' && result.deepfakeScore !== null
      ? result.deepfakeScore >= config.HIVE_THRESHOLD
        ? 'FAKE'
        : 'REAL'
      : null;
  return ExternalCheck.create({
    mediaSha256: sha256,
    provider: verifier.provider,
    status: result.status,
    deepfakeScore: result.deepfakeScore,
    aiGeneratedScore: result.aiGeneratedScore,
    threshold: config.HIVE_THRESHOLD,
    verdict,
    latencyMs: Date.now() - started,
    rawResponse: result.raw,
  });
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
  const externalPromise = analysis.externalCheckRequested
    ? externalCheck(media.sha256, storagePath(media.storageKey)).catch((err: unknown) => {
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

  // Phase 1 always uses the template. The Groq call (with this as its fallback) is Phase 4.
  const explanation = buildTemplateExplanation({
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
                ? 'FAKE'
                : 'REAL'
              : null,
        }
      : null,
    mock: Boolean((analysis.ml as { mock?: boolean } | null)?.mock),
  });
  await Explanation.findOneAndUpdate(
    { analysisId },
    { ...explanation, groqModel: null, promptVersion: TEMPLATE_PROMPT_VERSION, fallbackUsed: true },
    { upsert: true },
  );
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
