import { mkdtempSync, utimesSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import type { StoredExplanation } from '../src/explain/explainer.js';
import { reconcileStaleJobs, sweepUploads } from '../src/maintenance.js';
import { MlServiceError, type MlApi, type MlProgress } from '../src/providers/mlClient.js';
import {
  classifyMlError,
  failAnalysis,
  PermanentJobError,
  processAnalysis,
  processExplanation,
  type ProcessDeps,
  type ProgressEvent,
} from '../src/queue/process.js';
import { diskFileStore } from '../src/storage.js';
import type { MlResult } from '../src/types.js';
import { memoryRepos } from './helpers/memoryRepos.js';

const STORAGE_KEY = '6f1c1a2e-0000-4000-8000-0000000000aa';

const mlImage: MlResult = {
  verdict: 'FAKE',
  probability: 0.93,
  threshold: 0.5,
  confident: true,
  uncertainReason: null,
  faces: [
    {
      scores: { gendHead: 2.4, sbi: 1.1 },
      spread: 0.3,
      heatmaps: { gend: `derived/${'1'.repeat(32)}.png` },
    },
  ],
  modelCardVersion: 'v-test',
};

const template: StoredExplanation = {
  summary: 'The automated check found signs of manipulation.',
  why: 'w',
  caveats: ['automated'],
  groqModel: null,
  promptVersion: 'template-v1',
  fallbackUsed: true,
  fallbackReason: 'groq_off',
  inputSha256: null,
  latencyMs: null,
  usage: null,
};

async function setup(opts: { kind?: 'image' | 'video'; consent?: boolean } = {}) {
  const repos = memoryRepos();
  const uploadDir = mkdtempSync(path.join(tmpdir(), 'df-proc-'));
  writeFileSync(path.join(uploadDir, STORAGE_KEY), 'x');
  const owner = await repos.users.create({ email: 'o@example.test', passwordHash: 'x' });
  const media = await repos.media.create({
    ownerId: owner.id,
    kind: opts.kind ?? 'image',
    storageKey: STORAGE_KEY,
    mime: 'image/png',
    sha256: 'b'.repeat(64),
    sizeBytes: 1,
    expiresAt: new Date(Date.now() + 3600_000),
  });
  const analysis = await repos.analyses.create({
    ownerId: owner.id,
    mediaId: media.id,
    mediaSha256: media.sha256,
    kind: media.kind,
    externalCheckRequested: opts.consent ?? false,
  });
  const events: ProgressEvent[] = [];
  const explained: string[] = [];
  const ml = {
    analyze: vi.fn(async () => mlImage),
    modelInfo: vi.fn(),
    compare: vi.fn(),
    progress: vi.fn(async (): Promise<MlProgress | null> => null),
  };
  const deps: ProcessDeps = {
    repos,
    files: diskFileStore(uploadDir),
    ml: ml as unknown as MlApi,
    externalCheck: vi.fn(async () => null),
    explain: vi.fn(async () => template),
    enqueueExplain: async (id) => {
      explained.push(id);
    },
    emit: (e) => events.push(e),
    hiveThreshold: 0.9,
    progressPollMs: 5,
  };
  return { repos, deps, ml, events, explained, analysis, media, uploadDir };
}

describe('processAnalysis', () => {
  it('runs the stages in order, saves the reconciled result and queues the explanation', async () => {
    const s = await setup();
    await processAnalysis(s.deps, s.analysis.id);
    expect(s.events.map((e) => e.stage)).toEqual(['detecting_faces', 'verifying', 'explaining']);
    expect(s.ml.analyze).toHaveBeenCalledWith('image', STORAGE_KEY, {
      progressId: `${s.analysis.id}-1`,
    });
    const saved = (await s.repos.analyses.findById(s.analysis.id))!;
    expect(saved).toMatchObject({
      status: 'processing',
      stage: 'explaining',
      attempts: 1,
      final: { verdict: 'FAKE', badge: 'single_check' },
      ours: { verdict: 'FAKE', probability: 0.93, detectorScores: { gendHead: 2.4, sbi: 1.1 } },
      heatmapKeys: [`derived/${'1'.repeat(32)}.png`],
      modelCardVersion: 'v-test',
    });
    expect(s.explained).toEqual([s.analysis.id]);
    expect(s.deps.externalCheck).not.toHaveBeenCalled(); // no consent, no second check
  });

  it("forwards the ML service's measured progress while it works", async () => {
    const s = await setup({ kind: 'video' });
    const steps: MlProgress[] = [
      { stage: 'extracting_frames', done: 0, total: 8, unit: 'frames' },
      { stage: 'scoring', done: 3, total: 8, unit: 'frames' },
      { stage: 'scoring', done: 3, total: 8, unit: 'frames' }, // unchanged: not re-sent
      { stage: 'scoring', done: 8, total: 8, unit: 'frames' },
    ];
    let i = 0;
    s.ml.progress.mockImplementation(async () => steps[Math.min(i++, steps.length - 1)]!);
    s.ml.analyze.mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve({ ...mlImage, faces: [] }), 80)),
    );
    await processAnalysis(s.deps, s.analysis.id);
    const progress = s.events
      .filter((e) => e.progress)
      .map((e) => `${e.stage} ${e.progress!.done}/${e.progress!.total}`);
    expect(progress).toEqual(['extracting_frames 0/8', 'scoring 3/8', 'scoring 8/8']);
    // Progress stops once the ML call returns: the last events are the later stages.
    expect(s.events.slice(-2).map((e) => e.stage)).toEqual(['verifying', 'explaining']);
  });

  it('runs the second check only with consent, and reconciles with it', async () => {
    const s = await setup({ consent: true });
    (s.deps.externalCheck as ReturnType<typeof vi.fn>).mockResolvedValue({
      _id: 'c'.repeat(24),
      status: 'ok',
      deepfakeScore: 0.95,
      aiGeneratedScore: 0.1,
      verdict: 'FAKE',
    });
    await processAnalysis(s.deps, s.analysis.id);
    expect(s.deps.externalCheck).toHaveBeenCalledWith(
      expect.objectContaining({ sha256: 'b'.repeat(64), userId: s.analysis.ownerId, demo: false }),
    );
    const saved = (await s.repos.analyses.findById(s.analysis.id))!;
    expect(saved.final).toMatchObject({ verdict: 'FAKE', badge: 'verified_by_2' });
    expect(saved.externalCheckId).toBe('c'.repeat(24));
  });

  it('is idempotent: a finished analysis is left alone; a saved result only needs its explanation', async () => {
    const s = await setup();
    await s.repos.analyses.update(s.analysis.id, { status: 'done', stage: 'done' });
    await processAnalysis(s.deps, s.analysis.id);
    expect(s.ml.analyze).not.toHaveBeenCalled();

    const t = await setup();
    await t.repos.analyses.update(t.analysis.id, {
      final: { verdict: 'REAL', badge: 'single_check', ruleApplied: 'r' },
      ours: {
        verdict: 'REAL',
        probability: 0.1,
        threshold: 0.5,
        confident: true,
        uncertainReason: null,
        detectorScores: null,
        spread: null,
      },
    });
    await processAnalysis(t.deps, t.analysis.id);
    expect(t.ml.analyze).not.toHaveBeenCalled();
    expect(t.explained).toEqual([t.analysis.id]);
  });

  it('fails permanently for bad media or an expired upload, and transiently otherwise', async () => {
    const s = await setup();
    s.ml.analyze.mockRejectedValue(
      new MlServiceError(422, 'ML service 422: video has no decodable frames'),
    );
    await expect(processAnalysis(s.deps, s.analysis.id)).rejects.toThrow(
      new PermanentJobError('video has no decodable frames'),
    );

    const t = await setup();
    t.ml.analyze.mockRejectedValue(new MlServiceError(503, 'ML service unreachable'));
    const err = await processAnalysis(t.deps, t.analysis.id).catch((e: unknown) => e);
    expect(err).not.toBeInstanceOf(PermanentJobError);

    const u = await setup();
    await u.repos.media.delete(u.media.id);
    await expect(processAnalysis(u.deps, u.analysis.id)).rejects.toBeInstanceOf(PermanentJobError);

    expect(classifyMlError(new MlServiceError(501, 'x')).permanent).toBe(true);
    expect(classifyMlError(new Error('boom')).permanent).toBe(false);
  });

  it('does nothing for an analysis deleted while queued', async () => {
    const s = await setup();
    await s.repos.analyses.delete(s.analysis.id);
    await processAnalysis(s.deps, s.analysis.id);
    expect(s.ml.analyze).not.toHaveBeenCalled();
  });
});

describe('processExplanation and failures', () => {
  it('stores the explanation and finishes the analysis', async () => {
    const s = await setup();
    await processAnalysis(s.deps, s.analysis.id);
    await processExplanation(s.deps, s.analysis.id);
    const saved = (await s.repos.analyses.findById(s.analysis.id))!;
    expect(saved).toMatchObject({ status: 'done', stage: 'done', progress: null });
    expect(saved.completedAt).toBeInstanceOf(Date);
    expect((await s.repos.explanations.findByAnalysis(s.analysis.id))?.fallbackReason).toBe(
      'groq_off',
    );
    expect(s.events.at(-1)).toMatchObject({ stage: 'done', status: 'done' });
  });

  it('records a failure with a user-facing message', async () => {
    const s = await setup();
    await failAnalysis(s.deps, s.analysis.id, 'The uploaded file has expired or was deleted.');
    expect(await s.repos.analyses.findById(s.analysis.id)).toMatchObject({
      status: 'failed',
      stage: 'failed',
      error: 'The uploaded file has expired or was deleted.',
    });
    expect(s.events.at(-1)).toMatchObject({
      stage: 'failed',
      detail: expect.stringMatching(/expired/),
    });
  });
});

describe('maintenance', () => {
  it('requeues, re-explains or fails stale jobs that lost their worker', async () => {
    const s = await setup();
    const repos = s.repos;
    const mk = async (patch: Parameters<typeof repos.analyses.update>[1]) => {
      const a = await repos.analyses.create({
        ownerId: s.analysis.ownerId,
        mediaId: s.media.id,
        mediaSha256: 'x',
        kind: 'image',
        externalCheckRequested: false,
      });
      await repos.analyses.update(a.id, { status: 'processing', ...patch });
      return a.id;
    };
    const retry = await mk({ attempts: 1 });
    const explain = await mk({
      attempts: 1,
      final: { verdict: 'REAL', badge: 'single_check', ruleApplied: 'r' },
      ours: {
        verdict: 'REAL',
        probability: 0.2,
        threshold: 0.5,
        confident: true,
        uncertainReason: null,
        detectorScores: null,
        spread: null,
      },
    });
    const exhausted = await mk({ attempts: 2 });
    const alive = await mk({ attempts: 1 });
    const requeued: string[] = [];
    const explained: string[] = [];
    const failed: string[] = [];
    const report = await reconcileStaleJobs({
      repos,
      maxAttempts: 2,
      staleAfterMs: 0,
      now: () => Date.now() + 60_000,
      hasLiveJob: async (a) => a.id === alive,
      requeueAnalysis: async (id) => void requeued.push(id),
      requeueExplanation: async (id) => void explained.push(id),
      fail: async (id) => void failed.push(id),
    });
    expect(requeued).toContain(retry);
    expect(explained).toEqual([explain]);
    expect(failed).toEqual([exhausted]);
    expect(requeued).not.toContain(alive);
    expect(report.failed).toBe(1);
    expect((await repos.analyses.findById(retry))!.status).toBe('queued');
  });

  it('deletes only old, server-named uploads', async () => {
    const dir = mkdtempSync(path.join(tmpdir(), 'df-sweep-'));
    const old = path.join(dir, '6f1c1a2e-0000-4000-8000-000000000001');
    const fresh = path.join(dir, '6f1c1a2e-0000-4000-8000-000000000002');
    const other = path.join(dir, 'keep-me.txt');
    for (const f of [old, fresh, other]) writeFileSync(f, 'x');
    const past = new Date(Date.now() - 48 * 3600_000);
    utimesSync(old, past, past);
    utimesSync(other, past, past);
    expect(await sweepUploads(dir, 24 * 3600_000)).toBe(1);
    expect([existsSync(old), existsSync(fresh), existsSync(other)]).toEqual([false, true, true]);
    expect(await sweepUploads(path.join(dir, 'missing'), 1)).toBe(0);
  });
});
