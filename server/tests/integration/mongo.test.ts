/**
 * MongoDB integration tests: the production repositories (src/data/mongo.ts) against a real
 * MongoDB. Each run uses its own database, dropped afterwards. Like the Redis tests, they fail
 * loudly when MongoDB is unreachable unless SKIP_MONGO_TESTS=1 is set on purpose.
 */
import mongoose from 'mongoose';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { mongoRepos } from '../../src/data/mongo.js';
import { evalReportFromLofo } from '../../src/routes/meta.js';
import { Media } from '../../src/models/index.js';
import { readFileSync } from 'node:fs';

const base = process.env.TEST_MONGO_URL ?? 'mongodb://127.0.0.1:27017';
const dbName = `deepfake_test_${process.pid}_${Date.now()}`;
let available = false;

beforeAll(async () => {
  try {
    await mongoose.connect(`${base}/${dbName}`, { serverSelectionTimeoutMS: 2000 });
    available = true;
    await mongoose.connection.syncIndexes();
  } catch {
    available = false;
  }
  if (!available && process.env.SKIP_MONGO_TESTS !== '1') {
    throw new Error(
      `MongoDB is not reachable at ${base}. Start MongoDB, or set SKIP_MONGO_TESTS=1 to skip on purpose.`,
    );
  }
});

afterAll(async () => {
  if (available) await mongoose.connection.dropDatabase();
  await mongoose.disconnect();
});

const repos = mongoRepos();

async function owner() {
  return repos.users.create({ email: `${Math.random()}@example.test`, passwordHash: 'hash' });
}

async function analysisFor(ownerId: string, sha = 'a'.repeat(64)) {
  const media = await repos.media.create({
    ownerId,
    kind: 'image',
    storageKey: '6f1c1a2e-0000-4000-8000-000000000001',
    mime: 'image/png',
    sha256: sha,
    sizeBytes: 10,
    expiresAt: new Date(Date.now() + 3600_000),
  });
  const analysis = await repos.analyses.create({
    ownerId,
    mediaId: media.id,
    mediaSha256: sha,
    kind: 'image',
    externalCheckRequested: false,
  });
  return { media, analysis };
}

describe.runIf(process.env.SKIP_MONGO_TESTS !== '1')('MongoDB repositories', () => {
  it('stores users with a unique, lower-cased email', async () => {
    const u = await repos.users.create({ email: 'Mixed@Example.TEST', passwordHash: 'h' });
    expect(u).toMatchObject({ email: 'mixed@example.test', role: 'user' });
    expect(await repos.users.findByEmail('mixed@example.test')).toMatchObject({ id: u.id });
    expect(await repos.users.findById(u.id)).toMatchObject({ email: 'mixed@example.test' });
    expect(await repos.users.findById('not-an-id')).toBeNull();
    await expect(
      repos.users.create({ email: 'mixed@example.test', passwordHash: 'h' }),
    ).rejects.toThrow(/duplicate key/);
  });

  it('creates, reads, updates and deletes analyses with ownership', async () => {
    const me = await owner();
    const other = await owner();
    const { analysis, media } = await analysisFor(me.id);
    expect(analysis).toMatchObject({
      status: 'queued',
      stage: 'queued',
      attempts: 0,
      progress: null,
    });
    expect(await repos.analyses.findOwned(analysis.id, me.id)).toMatchObject({
      id: analysis.id,
      mediaId: media.id,
    });
    expect(await repos.analyses.findOwned(analysis.id, other.id)).toBeNull();
    expect(await repos.analyses.findOwned('nope', me.id)).toBeNull();

    await repos.analyses.update(analysis.id, {
      status: 'processing',
      stage: 'scoring',
      progress: { done: 3, total: 8, unit: 'frames' },
      attempts: 1,
      heatmapKeys: [`derived/${'1'.repeat(32)}.png`],
      ml: { verdict: 'FAKE', frames: [{ index: 0, timestamp: 0, hasFace: true, logit: 1 }] },
    });
    const updated = (await repos.analyses.findById(analysis.id))!;
    expect(updated).toMatchObject({
      status: 'processing',
      progress: { done: 3, total: 8, unit: 'frames' },
      attempts: 1,
      heatmapKeys: [`derived/${'1'.repeat(32)}.png`],
      ml: { verdict: 'FAKE' },
    });
    expect(updated.updatedAt.getTime()).toBeGreaterThanOrEqual(analysis.updatedAt.getTime());

    await repos.analyses.delete(analysis.id);
    expect(await repos.analyses.findById(analysis.id)).toBeNull();
    expect(await repos.media.delete(media.id)).toMatchObject({ id: media.id });
    expect(await repos.media.findById(media.id)).toBeNull();
  });

  it("pages a user's analyses newest first without the heavy ML result", async () => {
    const me = await owner();
    const ids: string[] = [];
    for (let i = 0; i < 3; i++) {
      const { analysis } = await analysisFor(me.id, String(i).repeat(64));
      await repos.analyses.update(analysis.id, { ml: { big: 'x'.repeat(1000) } });
      ids.push(analysis.id);
    }
    const page1 = await repos.analyses.listOwned(me.id, 1, 2);
    expect(page1.total).toBe(3);
    expect(page1.items.map((a) => a.id)).toEqual([ids[2], ids[1]]);
    expect(page1.items[0]!.ml).toBeNull(); // projected out
    expect((await repos.analyses.listOwned(me.id, 2, 2)).items.map((a) => a.id)).toEqual([ids[0]]);
  });

  it('finds running duplicates, stale jobs and second-check references', async () => {
    const me = await owner();
    const { analysis } = await analysisFor(me.id, 'd'.repeat(64));
    expect((await repos.analyses.findActiveDuplicate(me.id, 'd'.repeat(64)))?.id).toBe(analysis.id);
    await repos.analyses.update(analysis.id, { status: 'done' });
    expect(await repos.analyses.findActiveDuplicate(me.id, 'd'.repeat(64))).toBeNull();

    const stale = (await analysisFor(me.id, 'e'.repeat(64))).analysis;
    const later = new Date(Date.now() + 60_000);
    expect((await repos.analyses.findStale(later)).map((a) => a.id)).toContain(stale.id);
    expect((await repos.analyses.findStale(new Date(0))).map((a) => a.id)).not.toContain(stale.id);

    const checkId = new mongoose.Types.ObjectId().toHexString();
    await repos.analyses.update(stale.id, { externalCheckId: checkId });
    expect(await repos.analyses.countByExternalCheck(checkId)).toBe(1);
  });

  it('upserts explanations and finds cached Groq ones only', async () => {
    const me = await owner();
    const { analysis } = await analysisFor(me.id);
    const doc = {
      groqModel: null,
      promptVersion: 'template-v1',
      summary: 's',
      why: 'w',
      caveats: ['c'],
      fallbackUsed: true,
      fallbackReason: 'groq_off',
      inputSha256: 'f'.repeat(64),
      latencyMs: null,
      usage: null,
    };
    await repos.explanations.upsert(analysis.id, doc);
    await repos.explanations.upsert(analysis.id, { ...doc, summary: 's2' });
    expect(await repos.explanations.findByAnalysis(analysis.id)).toMatchObject({
      summary: 's2',
      fallbackUsed: true,
    });
    expect(await repos.explanations.findCachedGroq('f'.repeat(64), 'template-v1')).toBeNull();
    await repos.explanations.upsert(analysis.id, {
      ...doc,
      fallbackUsed: false,
      promptVersion: 'groq-explain-v1',
      groqModel: 'm',
    });
    expect(
      await repos.explanations.findCachedGroq('f'.repeat(64), 'groq-explain-v1'),
    ).toMatchObject({ groqModel: 'm' });
    await repos.explanations.deleteByAnalysis(analysis.id);
    expect(await repos.explanations.findByAnalysis(analysis.id)).toBeNull();
  });

  it('stores comparisons and evaluation reports', async () => {
    const me = await owner();
    const { analysis } = await analysisFor(me.id);
    const c = await repos.comparisons.create({
      analysisId: analysis.id,
      ownerId: me.id,
      family: 'F1',
      strength: 0.5,
      threshold: 0.5,
      clean: { verdict: 'FAKE' },
      filtered: { verdict: 'REAL' },
      flipped: true,
      heatmapIoU: 0.4,
      heatmapKeys: [],
      modelCardVersion: 'v',
    });
    expect(await repos.comparisons.listByAnalysis(analysis.id)).toEqual([c]);
    await repos.comparisons.deleteByAnalysis(analysis.id);
    expect(await repos.comparisons.listByAnalysis(analysis.id)).toEqual([]);

    const lofo = JSON.parse(
      readFileSync(new URL('../fixtures/lofo-sample.json', import.meta.url), 'utf8'),
    );
    const report = await repos.evalReports.create(evalReportFromLofo(lofo));
    const latest = await repos.evalReports.latest();
    expect(latest).toMatchObject({ id: report.id, fixture: true });
    expect(latest!.runs[0]).toMatchObject({ scheme: 'H-a', ci95: { clean_auc: [0.7, 0.9] } });
  });

  it('has the TTL index that expires uploads and the query indexes', async () => {
    const mediaIndexes = await Media.collection.indexes();
    expect(mediaIndexes.find((i) => i.key.expiresAt === 1)?.expireAfterSeconds).toBe(0);
    const analysisIndexes = await mongoose.connection.collection('analyses').indexes();
    const keys = analysisIndexes.map((i) => Object.keys(i.key).join(','));
    expect(keys).toEqual(
      expect.arrayContaining([
        'ownerId,createdAt',
        'ownerId,mediaSha256,status',
        'status,updatedAt',
      ]),
    );
  });
});
