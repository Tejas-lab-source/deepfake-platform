/**
 * HTTP route tests with injected in-memory repositories (no MongoDB). They test the routes'
 * behaviour: auth, ownership, validation, files on disk. They are not MongoDB integration tests.
 */
import { existsSync, readdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import request from 'supertest';
import { describe, expect, it } from 'vitest';
import { memoryRateLimiter } from '../../src/rateLimit.js';
import { PNG_1PX, testApp, type TestContext } from '../helpers/testApp.js';

async function seedDone(ctx: TestContext, ownerId: string, ml: Record<string, unknown> = {}) {
  const media = await ctx.repos.media.create({
    ownerId,
    kind: 'image',
    storageKey: '6f1c1a2e-0000-4000-8000-000000000001',
    mime: 'image/png',
    sha256: 'a'.repeat(64),
    sizeBytes: 10,
    expiresAt: new Date(Date.now() + 3600_000),
  });
  writeFileSync(path.join(ctx.uploadDir, media.storageKey), PNG_1PX);
  const a = await ctx.repos.analyses.create({
    ownerId,
    mediaId: media.id,
    mediaSha256: media.sha256,
    kind: 'image',
    externalCheckRequested: false,
  });
  await ctx.repos.analyses.update(a.id, {
    status: 'done',
    stage: 'done',
    final: { verdict: 'FAKE', badge: 'single_check', ruleApplied: 'ours_only' },
    ours: {
      verdict: 'FAKE',
      probability: 0.91,
      threshold: 0.5,
      confident: true,
      uncertainReason: null,
      detectorScores: { gendHead: 2.1, sbi: 1.3 },
      spread: 0.2,
    },
    ml: { verdict: 'FAKE', probability: 0.91, threshold: 0.5, modelCardVersion: 'v1', ...ml },
  });
  return { analysis: (await ctx.repos.analyses.findById(a.id))!, media };
}

describe('authentication', () => {
  it('registers, logs in and refreshes; rejects wrong passwords and duplicates', async () => {
    const { app } = testApp();
    const creds = { email: 'Person@Example.test', password: 'correct horse' };
    const reg = await request(app).post('/api/v1/auth/register').send(creds).expect(201);
    expect(reg.body.user).toMatchObject({ email: 'person@example.test', role: 'user' });
    expect(reg.body.user).not.toHaveProperty('passwordHash');
    await request(app).post('/api/v1/auth/register').send(creds).expect(409);
    await request(app)
      .post('/api/v1/auth/login')
      .send({ ...creds, password: 'wrong password' })
      .expect(401);
    await request(app)
      .post('/api/v1/auth/login')
      .send({ email: 'nobody@example.test', password: 'whatever1' })
      .expect(401);
    const login = await request(app).post('/api/v1/auth/login').send(creds).expect(200);
    const refreshed = await request(app)
      .post('/api/v1/auth/refresh')
      .send({ refreshToken: login.body.refreshToken })
      .expect(200);
    expect(refreshed.body.accessToken).toEqual(expect.any(String));
    // An access token is not a refresh token.
    await request(app)
      .post('/api/v1/auth/refresh')
      .send({ refreshToken: login.body.accessToken })
      .expect(401);
  });

  it('answers invalid input and missing tokens with problem+json', async () => {
    const { app } = testApp();
    const bad = await request(app)
      .post('/api/v1/auth/register')
      .send({ email: 'x', password: '1' });
    expect(bad.status).toBe(400);
    expect(bad.headers['content-type']).toMatch(/application\/problem\+json/);
    expect(bad.body).toMatchObject({ status: 400, title: 'Invalid request' });
    const anon = await request(app).get('/api/v1/analyses').expect(401);
    expect(anon.body.detail).toBe('Missing or invalid access token');
    await request(app).get('/api/v1/analyses').set('Authorization', 'Bearer not-a-jwt').expect(401);
  });

  it('rate-limits login attempts per client', async () => {
    const { app } = testApp({
      rateLimiters: {
        auth: memoryRateLimiter(2, 60),
        upload: memoryRateLimiter(100, 60),
        ask: memoryRateLimiter(100, 60),
        compare: memoryRateLimiter(100, 60),
      },
    });
    const creds = { email: 'a@example.test', password: 'whatever1' };
    await request(app).post('/api/v1/auth/login').send(creds).expect(401);
    await request(app).post('/api/v1/auth/login').send(creds).expect(401);
    const limited = await request(app).post('/api/v1/auth/login').send(creds).expect(429);
    expect(limited.headers['retry-after']).toEqual(expect.any(String));
  });
});

describe('uploads', () => {
  it('accepts an image by magic bytes, stores it under a random name and queues it', async () => {
    const ctx = testApp();
    const { auth } = await ctx.user();
    const res = await request(ctx.app)
      .post('/api/v1/analyses')
      .set(auth)
      .field('externalCheck', 'false')
      .attach('file', PNG_1PX, { filename: '../../secret name.jpg', contentType: 'text/plain' })
      .expect(202);
    expect(res.body).toEqual({ analysisId: expect.any(String), jobId: expect.any(String) });
    expect(ctx.enqueued).toEqual([res.body.analysisId]);
    const stored = readdirSync(ctx.uploadDir).filter((f) => f !== 'derived');
    expect(stored).toHaveLength(1);
    expect(stored[0]).toMatch(/^[0-9a-f-]{36}$/); // never the client's name
    const media = [...ctx.repos.store.media.values()][0]!;
    expect(media).toMatchObject({ kind: 'image', mime: 'image/png', sizeBytes: PNG_1PX.length });
  });

  it('rejects unsupported, empty and oversized files and deletes them', async () => {
    const ctx = testApp();
    const { auth } = await ctx.user();
    const post = (buf: Buffer) =>
      request(ctx.app).post('/api/v1/analyses').set(auth).attach('file', buf, 'x.png');
    expect((await post(Buffer.from('just text, not an image'))).status).toBe(415);
    expect((await post(Buffer.alloc(0))).status).toBe(415);
    // 1 MB image limit in the test app: a PNG header followed by 1.5 MB.
    const big = Buffer.concat([PNG_1PX, Buffer.alloc(1.5 * 1024 * 1024)]);
    expect((await post(big)).status).toBe(413);
    // Over the multer ceiling (the 2 MB video limit) the upload is cut off.
    expect((await post(Buffer.concat([PNG_1PX, Buffer.alloc(3 * 1024 * 1024)]))).status).toBe(413);
    expect(readdirSync(ctx.uploadDir).filter((f) => f !== 'derived')).toEqual([]);
    expect(ctx.enqueued).toEqual([]);
  });

  it('requires the file field and validates externalCheck', async () => {
    const ctx = testApp();
    const { auth } = await ctx.user();
    await request(ctx.app).post('/api/v1/analyses').set(auth).field('x', '1').expect(400);
    await request(ctx.app)
      .post('/api/v1/analyses')
      .set(auth)
      .field('externalCheck', 'yes please')
      .attach('file', PNG_1PX, 'a.png')
      .expect(400);
    expect(readdirSync(ctx.uploadDir).filter((f) => f !== 'derived')).toEqual([]);
  });

  it('reuses the running analysis when the same file is uploaded again', async () => {
    const ctx = testApp();
    const { auth } = await ctx.user();
    const first = await request(ctx.app)
      .post('/api/v1/analyses')
      .set(auth)
      .attach('file', PNG_1PX, 'a.png');
    const second = await request(ctx.app)
      .post('/api/v1/analyses')
      .set(auth)
      .attach('file', PNG_1PX, 'b.png');
    expect(second.status).toBe(202);
    expect(second.body).toMatchObject({ analysisId: first.body.analysisId, duplicate: true });
    expect(ctx.enqueued).toHaveLength(1);
    expect(readdirSync(ctx.uploadDir).filter((f) => f !== 'derived')).toHaveLength(1);
  });
});

describe('reading analyses', () => {
  it("lists only the caller's analyses, newest first, paged", async () => {
    const ctx = testApp();
    const me = await ctx.user();
    const other = await ctx.user();
    for (let i = 0; i < 3; i++) await seedDone(ctx, me.id);
    await seedDone(ctx, other.id);
    const page = await request(ctx.app)
      .get('/api/v1/analyses?page=1&limit=2')
      .set(me.auth)
      .expect(200);
    expect(page.body).toMatchObject({ page: 1, limit: 2, total: 3 });
    expect(page.body.items).toHaveLength(2);
    expect(page.body.items[0]).toEqual({
      id: expect.any(String),
      kind: 'image',
      status: 'done',
      stage: 'done',
      final: { verdict: 'FAKE', badge: 'single_check', ruleApplied: 'ours_only' },
      ours: { probability: 0.91 },
      createdAt: expect.any(String),
    });
    await request(ctx.app).get('/api/v1/analyses?limit=500').set(me.auth).expect(400);
    await request(ctx.app).get('/api/v1/analyses?page=0').set(me.auth).expect(400);
  });

  it('returns one analysis to its owner only, with no internal fields', async () => {
    const ctx = testApp();
    const me = await ctx.user();
    const other = await ctx.user();
    const { analysis } = await seedDone(ctx, me.id);
    const res = await request(ctx.app)
      .get(`/api/v1/analyses/${analysis.id}`)
      .set(me.auth)
      .expect(200);
    expect(res.body).toMatchObject({ id: analysis.id, status: 'done', final: { verdict: 'FAKE' } });
    const text = JSON.stringify(res.body);
    for (const internal of ['mediaId', 'mediaSha256', 'storageKey', 'a'.repeat(64)]) {
      expect(text).not.toContain(internal);
    }
    await request(ctx.app).get(`/api/v1/analyses/${analysis.id}`).set(other.auth).expect(404);
    await request(ctx.app).get('/api/v1/analyses/not-an-id').set(me.auth).expect(404);
    await request(ctx.app)
      .get(`/api/v1/analyses/${'0'.repeat(24)}`)
      .set(me.auth)
      .expect(404);
  });
});

describe('heatmaps', () => {
  it("serves the owner's derived images from the derived directory only", async () => {
    const ctx = testApp();
    const me = await ctx.user();
    const other = await ctx.user();
    const key = ctx.derived();
    const notMine = ctx.derived();
    const { analysis } = await seedDone(ctx, me.id, {
      faces: [{ heatmaps: { gend: key, sbi: null } }],
    });
    const name = key.slice('derived/'.length);
    const url = `/api/v1/analyses/${analysis.id}/heatmaps/${name}`;

    const ok = await request(ctx.app).get(url).set(me.auth).expect(200);
    expect(ok.headers['content-type']).toBe('image/png');
    expect(ok.headers['x-content-type-options']).toBe('nosniff');
    expect(Buffer.compare(ok.body as Buffer, PNG_1PX)).toBe(0);

    await request(ctx.app).get(url).expect(401);
    await request(ctx.app).get(url).set(other.auth).expect(404);
    // An existing derived file that this analysis never produced.
    await request(ctx.app)
      .get(`/api/v1/analyses/${analysis.id}/heatmaps/${notMine.slice(8)}`)
      .set(me.auth)
      .expect(404);
    // Traversal and odd names never reach the filesystem.
    for (const evil of [
      '..%2F..%2Fetc%2Fpasswd',
      '..%2Fderived%2Fx.png',
      'x.png',
      `${name}.png`,
      '%2e%2e',
    ]) {
      const res = await request(ctx.app)
        .get(`/api/v1/analyses/${analysis.id}/heatmaps/${evil}`)
        .set(me.auth);
      expect([400, 404]).toContain(res.status);
      expect(res.headers['content-type']).toMatch(/problem\+json/);
    }
  });

  it('reports missing and damaged image files clearly', async () => {
    const ctx = testApp();
    const me = await ctx.user();
    const damaged = ctx.derived(Buffer.from('not a png at all'));
    const missing = 'derived/' + 'e'.repeat(32) + '.png';
    const { analysis } = await seedDone(ctx, me.id, {
      faces: [{ heatmaps: { gend: damaged, sbi: missing } }],
    });
    const d = await request(ctx.app)
      .get(`/api/v1/analyses/${analysis.id}/heatmaps/${damaged.slice(8)}`)
      .set(me.auth)
      .expect(404);
    expect(d.body.detail).toMatch(/damaged/);
    const m = await request(ctx.app)
      .get(`/api/v1/analyses/${analysis.id}/heatmaps/${missing.slice(8)}`)
      .set(me.auth)
      .expect(404);
    expect(m.body.detail).toMatch(/no longer available/);
  });
});

describe('deleting an analysis', () => {
  it('removes the upload, its derived images, explanation, comparisons and unshared second check', async () => {
    const ctx = testApp();
    const me = await ctx.user();
    const other = await ctx.user();
    const heat = ctx.derived();
    const cmpHeat = ctx.derived();
    const { analysis, media } = await seedDone(ctx, me.id, {
      faces: [{ heatmaps: { gend: heat } }],
    });
    ctx.repos.store.externalChecks.set('c'.repeat(24), {
      id: 'c'.repeat(24),
      mediaSha256: media.sha256,
      provider: 'hive',
      status: 'ok',
      deepfakeScore: 0.1,
      aiGeneratedScore: 0.1,
      threshold: 0.9,
      verdict: 'REAL',
      latencyMs: 5,
      reason: null,
      frames: [],
    });
    await ctx.repos.analyses.update(analysis.id, { externalCheckId: 'c'.repeat(24) });
    await ctx.repos.explanations.upsert(analysis.id, {
      groqModel: null,
      promptVersion: 'template-v1',
      summary: 's',
      why: 'w',
      caveats: [],
      fallbackUsed: true,
      fallbackReason: 'groq_off',
      inputSha256: null,
      latencyMs: null,
      usage: null,
    });
    await ctx.repos.comparisons.create({
      analysisId: analysis.id,
      ownerId: me.id,
      family: 'F1',
      strength: 0.5,
      threshold: 0.5,
      clean: {},
      filtered: {},
      flipped: false,
      heatmapIoU: 0.5,
      heatmapKeys: [cmpHeat],
      modelCardVersion: 'v1',
    });

    await request(ctx.app).delete(`/api/v1/analyses/${analysis.id}`).set(other.auth).expect(404);
    await request(ctx.app).delete(`/api/v1/analyses/${analysis.id}`).set(me.auth).expect(204);

    expect(existsSync(path.join(ctx.uploadDir, media.storageKey))).toBe(false);
    expect(existsSync(path.join(ctx.uploadDir, heat))).toBe(false);
    expect(existsSync(path.join(ctx.uploadDir, cmpHeat))).toBe(false);
    expect(ctx.repos.store.analyses.size).toBe(0);
    expect(ctx.repos.store.media.size).toBe(0);
    expect(ctx.repos.store.explanations.size).toBe(0);
    expect(ctx.repos.store.comparisons.size).toBe(0);
    expect(ctx.repos.store.externalChecks.size).toBe(0);
    await request(ctx.app).get(`/api/v1/analyses/${analysis.id}`).set(me.auth).expect(404);
  });
});
