/**
 * What an analysis job does, independent of BullMQ and MongoDB (both are injected), so every
 * step is testable. The BullMQ wiring lives in workers.ts.
 *
 *   1. ML service (our verdict) and, with consent, the second check run in parallel.
 *      While the ML service works, its measured progress (faces or frames done / total) is
 *      polled and forwarded to the browser.
 *   2. The reconciler merges the two; the result is saved.
 *   3. The explanation job writes the explanation (template unless Groq is enabled).
 *
 * Re-running a job is safe: a finished analysis is left alone, and a saved result is never
 * recomputed just to write its explanation.
 */
import { explanationInput, resultExtras } from '../analysisFacts.js';
import type { StoredExplanation } from '../explain/explainer.js';
import type { ExplanationInput } from '../explainTemplate.js';
import type { ResultExtras } from '../explain/prompt.js';
import type { AnalysisRecord, Progress, Repos } from '../data/types.js';
import type { StoredCheck } from '../providers/externalCheck.js';
import { MlServiceError, type MlApi } from '../providers/mlClient.js';
import { reconcile, type ExternalStatus } from '../reconcile.js';
import type { FileStore } from '../storage.js';
import { collectDerivedKeys } from '../storage.js';
import type { MlResult, Stage } from '../types.js';

export interface ProgressEvent {
  analysisId: string;
  stage: Stage;
  status: AnalysisRecord['status'];
  progress?: Progress | null;
  detail?: string;
}

export interface ExternalRequest {
  sha256: string;
  filePath: string;
  mime: string;
  userId: string;
  demo: boolean;
}

export interface ProcessDeps {
  repos: Repos;
  files: FileStore;
  ml: MlApi;
  /** The consented second check (Phase 4 safeExternalCheck); null result = not available. */
  externalCheck(req: ExternalRequest): Promise<StoredCheck | null>;
  explain(input: ExplanationInput, extras: ResultExtras): Promise<StoredExplanation>;
  enqueueExplain(analysisId: string): Promise<void>;
  emit(event: ProgressEvent): void;
  hiveThreshold: number;
  /** How often to ask the ML service how far it is. */
  progressPollMs: number;
}

/** A failure that retrying cannot fix (bad media, expired upload, unsupported request). */
export class PermanentJobError extends Error {}

const statusOf = (stage: Stage): AnalysisRecord['status'] =>
  stage === 'done'
    ? 'done'
    : stage === 'failed'
      ? 'failed'
      : stage === 'queued'
        ? 'queued'
        : 'processing';

export async function setStage(
  d: ProcessDeps,
  analysisId: string,
  stage: Stage,
  progress: Progress | null = null,
): Promise<void> {
  const status = statusOf(stage);
  await d.repos.analyses.update(analysisId, {
    stage,
    status,
    progress,
    ...(stage === 'done' ? { completedAt: new Date() } : {}),
  });
  d.emit({ analysisId, stage, status, progress });
}

/** Map an ML failure to a message a user can act on, and decide whether a retry could help. */
export function classifyMlError(err: unknown): { permanent: boolean; message: string } {
  if (err instanceof MlServiceError) {
    if (err.status === 422 || err.status === 413 || err.status === 415 || err.status === 400) {
      // Invalid, empty, corrupt or oversized media: the service explains why.
      return { permanent: true, message: err.message.replace(/^ML service \d+: /, '') };
    }
    if (err.status === 501) {
      return {
        permanent: true,
        message: 'This analysis is not supported by the ML service in its current mode.',
      };
    }
    return {
      permanent: false,
      message: 'The ML service is unavailable; the analysis will be retried.',
    };
  }
  return { permanent: false, message: 'Unexpected error while analysing the upload.' };
}

const ML_STAGES = new Set<Stage>([
  'detecting_faces',
  'extracting_frames',
  'scoring',
  'aggregating',
]);

/** Poll the ML service's progress for one request and forward real changes. */
function watchMlProgress(d: ProcessDeps, analysisId: string, progressId: string): () => void {
  let last = '';
  let stopped = false;
  const timer = setInterval(() => {
    void d.ml.progress(progressId).then(async (p) => {
      if (stopped || !p) return;
      const stage = (ML_STAGES.has(p.stage as Stage) ? p.stage : 'scoring') as Stage;
      const key = `${stage}:${p.done}/${p.total}`;
      if (key === last) return;
      last = key;
      await setStage(d, analysisId, stage, { done: p.done, total: p.total, unit: p.unit }).catch(
        () => undefined,
      );
    });
  }, d.progressPollMs);
  return () => {
    stopped = true;
    clearInterval(timer);
  };
}

export async function processAnalysis(d: ProcessDeps, analysisId: string): Promise<void> {
  const analysis = await d.repos.analyses.findById(analysisId);
  if (!analysis) return; // deleted while queued
  if (analysis.status === 'done') return; // already finished (duplicate or replayed job)
  if (analysis.final && analysis.ours) {
    // The result was saved but the explanation step did not run: only queue that.
    await setStage(d, analysisId, 'explaining');
    await d.enqueueExplain(analysisId);
    return;
  }

  const media = await d.repos.media.findById(analysis.mediaId);
  const filePath = media ? d.files.uploadPath(media.storageKey) : null;
  if (!media || !filePath) {
    throw new PermanentJobError('The uploaded file has expired or was deleted.');
  }
  await d.repos.analyses.update(analysisId, { attempts: analysis.attempts + 1, error: null });
  await setStage(d, analysisId, 'detecting_faces');

  const progressId = `${analysisId}-${analysis.attempts + 1}`;
  const stopWatching = watchMlProgress(d, analysisId, progressId);
  const mlPromise = d.ml.analyze(media.kind, media.storageKey, { progressId });

  // The second check only runs with the uploader's explicit consent for this upload.
  const consent = analysis.externalCheckRequested;
  const externalPromise: Promise<StoredCheck | null> = consent
    ? (async () => {
        const owner = await d.repos.users.findById(analysis.ownerId);
        return d.externalCheck({
          sha256: media.sha256,
          filePath,
          mime: media.mime,
          userId: analysis.ownerId,
          demo: owner?.role === 'admin',
        });
      })().catch(() => null)
    : Promise.resolve(null);

  let ml: MlResult;
  try {
    ml = await mlPromise;
  } catch (err) {
    await externalPromise; // never leave the second check running unobserved
    const { permanent, message } = classifyMlError(err);
    throw permanent ? new PermanentJobError(message) : new Error(message);
  } finally {
    stopWatching();
  }

  await setStage(d, analysisId, 'verifying');
  const external = await externalPromise;
  const externalStatus: ExternalStatus = !consent ? 'opted_out' : (external?.status ?? 'error');
  const final = reconcile({
    ours: { verdict: ml.verdict, confident: ml.confident },
    external: { status: externalStatus, deepfakeScore: external?.deepfakeScore ?? null },
    hiveThreshold: d.hiveThreshold,
  });

  const face = ml.faces?.[0];
  await d.repos.analyses.update(analysisId, {
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
    externalCheckId: external ? String(external._id) : null,
    faces: ml.faces ?? [],
    frames: ml.frames ?? [],
    heatmapKeys: collectDerivedKeys(ml),
    modelCardVersion: ml.modelCardVersion,
    ml,
  });
  await setStage(d, analysisId, 'explaining');
  await d.enqueueExplain(analysisId);
}

export async function processExplanation(d: ProcessDeps, analysisId: string): Promise<void> {
  const analysis = await d.repos.analyses.findById(analysisId);
  if (!analysis || analysis.status === 'done') return;
  const external = analysis.externalCheckId
    ? await d.repos.externalChecks.findById(analysis.externalCheckId)
    : null;
  const input = explanationInput(analysis, external, d.hiveThreshold);
  if (!input) return;
  // Groq (when enabled) sees only the whitelisted result JSON built from these values.
  const explanation = await d.explain(input, resultExtras(analysis, external));
  await d.repos.explanations.upsert(analysisId, explanation);
  await setStage(d, analysisId, 'done');
}

/** Record a job's final failure with a message safe to show the user. */
export async function failAnalysis(
  d: ProcessDeps,
  analysisId: string,
  message: string,
): Promise<void> {
  await d.repos.analyses.update(analysisId, {
    status: 'failed',
    stage: 'failed',
    progress: null,
    error: message,
  });
  d.emit({ analysisId, stage: 'failed', status: 'failed', detail: message });
}
