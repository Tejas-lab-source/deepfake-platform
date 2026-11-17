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
