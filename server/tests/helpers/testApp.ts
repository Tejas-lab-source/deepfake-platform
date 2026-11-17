/**
 * A fully wired HTTP app for route tests: in-memory repositories, a temporary upload
 * directory, a scripted ML service and a recording queue. No MongoDB, Redis, ML service or
 * network is involved.
 */
import { randomUUID } from 'node:crypto';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { vi } from 'vitest';
import { createApp } from '../../src/app.js';
import type { AppDeps } from '../../src/deps.js';
import { signTokens } from '../../src/middleware/auth.js';
import type { MlApi } from '../../src/providers/mlClient.js';
import { memoryRateLimiter } from '../../src/rateLimit.js';
import { diskFileStore } from '../../src/storage.js';
import { memoryRepos, type MemoryRepos } from './memoryRepos.js';

/** A real 1x1 PNG. */
export const PNG_1PX = Buffer.from(
  '89504e470d0a1a0a0000000d4948445200000001000000010806000000' +
    '1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082',
  'hex',
);

export interface TestContext {
  app: ReturnType<typeof createApp>;
  deps: AppDeps;
  repos: MemoryRepos;
  uploadDir: string;
  ml: { [K in keyof MlApi]: ReturnType<typeof vi.fn> };
  enqueued: string[];
  /** A registered user and a bearer header for them. */
  user(role?: 'user' | 'admin'): Promise<{ id: string; auth: { Authorization: string } }>;
  /** Write a derived image (heatmap) and return its key, e.g. "derived/<hex>.png". */
  derived(bytes?: Buffer): string;
}

export function testApp(overrides: Partial<AppDeps> = {}): TestContext {
  const repos = memoryRepos();
  const uploadDir = mkdtempSync(path.join(tmpdir(), 'df-uploads-'));
  mkdirSync(path.join(uploadDir, 'derived'), { recursive: true });
  const enqueued: string[] = [];
  const ml = {
    analyze: vi.fn(),
    modelInfo: vi.fn(async () => ({
      mockMode: true,
      k: 4,
      maxFrames: 32,
      videoTopK: 5,
      calibrationStatus: 'defaults',
      fixtureModelCard: false,
      calibration: {
        threshold: 0.5,
        temperature: 1,
        version: 'defaults-uncalibrated',
        sources: {},
      },
    })),
    compare: vi.fn(),
    progress: vi.fn(async () => null),
  };
  const generous = () => memoryRateLimiter(1000, 60);
  const deps: AppDeps = {
    repos,
    files: diskFileStore(uploadDir),
    ml: ml as unknown as MlApi,
    queue: {
      async enqueue(analysisId) {
        enqueued.push(analysisId);
        return `analysis-${analysisId}`;
      },
    },
    limits: { maxImageMb: 1, maxVideoMb: 2 },
    rateLimiters: { auth: generous(), upload: generous(), ask: generous(), compare: generous() },
    corsOrigins: [],
    mockMode: true,
    hiveThreshold: 0.9,
    ask: { mode: 'off', groq: null },
    maxQuestionChars: 500,
    ...overrides,
  };
  return {
    app: createApp(deps),
    deps,
    repos,
    uploadDir,
    ml,
    enqueued,
    async user(role = 'user') {
      const u = await repos.users.create({
        email: `${randomUUID()}@example.test`,
        passwordHash: 'x',
        role,
      });
      return {
        id: u.id,
        auth: { Authorization: `Bearer ${signTokens({ id: u.id, role }).accessToken}` },
      };
    },
    derived(bytes = PNG_1PX) {
      const name = randomUUID().replace(/-/g, '');
      writeFileSync(path.join(uploadDir, 'derived', `${name}.png`), bytes);
      return `derived/${name}.png`;
    },
  };
}

export const SECRET_SHA = 'a'.repeat(64);

/** A finished analysis (image or video) owned by `ownerId`, with its upload on disk. */
export async function seedFinished(
  ctx: TestContext,
  ownerId: string,
  opts: {
    kind?: 'image' | 'video';
    verdict?: 'FAKE' | 'REAL' | 'UNCERTAIN';
    ml?: Record<string, unknown>;
    externalCheckRequested?: boolean;
    status?: 'done' | 'processing';
  } = {},
) {
  const kind = opts.kind ?? 'image';
  const verdict = opts.verdict ?? 'FAKE';
  const media = await ctx.repos.media.create({
    ownerId,
    kind,
    storageKey: randomUUID(),
    mime: kind === 'image' ? 'image/png' : 'video/mp4',
    sha256: SECRET_SHA,
    sizeBytes: 10,
    expiresAt: new Date(Date.now() + 3600_000),
  });
  writeFileSync(path.join(ctx.uploadDir, media.storageKey), PNG_1PX);
  const a = await ctx.repos.analyses.create({
    ownerId,
    mediaId: media.id,
    mediaSha256: media.sha256,
    kind,
    externalCheckRequested: opts.externalCheckRequested ?? false,
  });
  const probability = verdict === 'FAKE' ? 0.91 : verdict === 'REAL' ? 0.12 : 0.53;
  await ctx.repos.analyses.update(a.id, {
    status: opts.status ?? 'done',
    stage: opts.status === 'processing' ? 'scoring' : 'done',
    final: { verdict, badge: 'single_check', ruleApplied: 'external_unavailable:opted_out' },
    ours: {
      verdict,
      probability,
      threshold: 0.5,
      confident: verdict !== 'UNCERTAIN',
      uncertainReason: verdict === 'UNCERTAIN' ? 'near_threshold' : null,
      detectorScores: { gendHead: 2.1, sbi: 1.3 },
      spread: 0.2,
    },
    modelCardVersion: 'mc-test',
    completedAt: new Date(),
    ml: {
      verdict,
      probability,
      threshold: 0.5,
      modelCardVersion: 'mc-test',
      faces: [
        {
          scores: { gendHead: 2.1, sbi: 1.3 },
          variantLogits: { clean: 2.0, F1: 2.2 },
          spread: 0.2,
          beautifiedLikelihood: 0.31,
          heatmapIoU: 0.42,
          heatmaps: null,
        },
      ],
      ...opts.ml,
    },
  });
  return { analysis: (await ctx.repos.analyses.findById(a.id))!, media };
}

/** A video ML result: 4 sampled frames, one without a face. */
export const VIDEO_ML = {
  kind: 'video',
  video: { fps: 10, frameCount: 40, durationSec: 4, width: 64, height: 48, sampledFrames: 4 },
  framesSampled: 4,
  framesScored: 3,
  aggregate: { method: 'top_k_mean', k: 3, logit: 1.6, sbi: null },
  mostSuspicious: { index: 25, timestamp: 2.5, logit: 2.4 },
  leastSuspicious: { index: 5, timestamp: 0.5, logit: 0.7 },
  frames: [
    {
      index: 5,
      timestamp: 0.5,
      decoded: true,
      hasFace: true,
      logit: 0.7,
      probability: 0.67,
      verdict: 'FAKE',
      explained: true,
    },
    {
      index: 15,
      timestamp: 1.5,
      decoded: true,
      hasFace: false,
      logit: null,
      probability: null,
      verdict: 'UNCERTAIN',
      explained: false,
    },
    {
      index: 25,
      timestamp: 2.5,
      decoded: true,
      hasFace: true,
      logit: 2.4,
      probability: 0.92,
      verdict: 'FAKE',
      explained: true,
    },
    {
      index: 35,
      timestamp: 3.5,
      decoded: true,
      hasFace: true,
      logit: 1.7,
      probability: 0.85,
      verdict: 'FAKE',
      explained: false,
    },
  ],
};
