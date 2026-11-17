/**
 * Periodic housekeeping, run by the server process.
 *
 * Stale jobs: an analysis still "queued" or "processing" long after its last update has lost
 * its job (worker crash, Redis flush, process killed mid-video). Each one is either requeued
 * (attempts left), sent straight to the explanation step (its result is already saved), or
 * marked failed with a clear message, so nothing stays stuck forever.
 *
 * Uploads: MongoDB's TTL index removes expired media records, but not the files. Files in the
 * upload directory older than the retention period are deleted. Derived images (heatmaps)
 * belong to analyses and are deleted with them.
 */
import { readdir, stat, unlink } from 'node:fs/promises';
import path from 'node:path';
import type { AnalysisRecord, Repos } from './data/types.js';
import { logger } from './logger.js';

const UPLOAD_NAME = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

export interface StaleJobDeps {
  repos: Repos;
  /** Whether BullMQ still holds a live job (waiting, delayed or active) for this analysis. */
  hasLiveJob(analysis: AnalysisRecord): Promise<boolean>;
  requeueAnalysis(analysisId: string): Promise<void>;
  requeueExplanation(analysisId: string): Promise<void>;
  fail(analysisId: string, message: string): Promise<void>;
  maxAttempts: number;
  staleAfterMs: number;
  now?: () => number;
}

export interface StaleJobReport {
  checked: number;
  requeued: number;
  explained: number;
  failed: number;
}

export async function reconcileStaleJobs(d: StaleJobDeps): Promise<StaleJobReport> {
  const now = (d.now ?? Date.now)();
  const stale = await d.repos.analyses.findStale(new Date(now - d.staleAfterMs));
  const report: StaleJobReport = { checked: stale.length, requeued: 0, explained: 0, failed: 0 };
  for (const a of stale) {
    try {
      if (await d.hasLiveJob(a)) continue;
      if (a.final && a.ours) {
        await d.requeueExplanation(a.id);
        report.explained++;
      } else if (a.attempts < d.maxAttempts) {
        await d.repos.analyses.update(a.id, { status: 'queued', stage: 'queued', progress: null });
        await d.requeueAnalysis(a.id);
        report.requeued++;
      } else {
        await d.fail(
          a.id,
          'Processing was interrupted and could not be completed. Please upload again.',
        );
        report.failed++;
      }
    } catch (err) {
      logger.warn({ err, analysisId: a.id }, 'stale job reconciliation failed');
    }
  }
  return report;
}

/** Delete uploads older than `maxAgeMs`. Only server-made names (UUIDs) are touched. */
export async function sweepUploads(
  uploadDir: string,
  maxAgeMs: number,
  now: number = Date.now(),
): Promise<number> {
  let removed = 0;
  let names: string[];
  try {
    names = await readdir(uploadDir);
  } catch {
    return 0;
  }
  for (const name of names) {
    if (!UPLOAD_NAME.test(name)) continue;
    const file = path.join(uploadDir, name);
    try {
      const info = await stat(file);
      if (info.isFile() && now - info.mtimeMs > maxAgeMs) {
        await unlink(file);
        removed++;
      }
    } catch {
      // vanished meanwhile
    }
  }
  return removed;
}

/** Run both sweeps every `intervalMs`; returns a stop function. Errors are logged, not thrown. */
export function startMaintenance(run: () => Promise<void>, intervalMs: number): () => void {
  const tick = (): void => {
    run().catch((err: unknown) => logger.warn({ err }, 'maintenance run failed'));
  };
  tick();
  const timer = setInterval(tick, intervalMs);
  timer.unref();
  return () => clearInterval(timer);
}
