import { readFileSync } from 'node:fs';
import request from 'supertest';
import { describe, expect, it } from 'vitest';
import { MlServiceError } from '../../src/providers/mlClient.js';
import { testApp } from '../helpers/testApp.js';

const lofo = JSON.parse(
  readFileSync(new URL('../fixtures/lofo-sample.json', import.meta.url), 'utf8'),
) as Record<string, unknown>;

describe('GET /model-card', () => {
  it("presents the ML service's model information with licences and decision rules", async () => {
    const ctx = testApp();
    const { auth } = await ctx.user();
    await request(ctx.app).get('/api/v1/model-card').expect(401);
    const res = await request(ctx.app).get('/api/v1/model-card').set(auth).expect(200);
    expect(res.body).toMatchObject({
      mockMode: true,
      calibration: { threshold: 0.5, temperature: 1, version: 'defaults-uncalibrated' },
      pipeline: { beautifiedCopiesPerImage: 4, maxVideoFrames: 32, videoTopK: 5 },
      secondCheck: { provider: 'Hive' },
    });
    expect(res.body.licences.map((l: { licence: string }) => l.licence)).toContain(
      'Research use only',
    );
    expect(res.body.notice).toMatch(/not proof/);
  });

  it('says clearly when the ML service is unreachable', async () => {
    const ctx = testApp();
    ctx.ml.modelInfo.mockRejectedValue(new MlServiceError(503, 'ML service unreachable'));
    const { auth } = await ctx.user();
    const res = await request(ctx.app).get('/api/v1/model-card').set(auth).expect(503);
    expect(res.body.title).toBe('Model information unavailable');
  });
});

describe('evaluation runs', () => {
  it('is empty until an admin imports an eval_lofo.py report', async () => {
    const ctx = testApp();
    const user = await ctx.user();
    const admin = await ctx.user('admin');
    expect(
      (await request(ctx.app).get('/api/v1/eval-runs').set(user.auth).expect(200)).body,
    ).toEqual({
      report: null,
    });
    await request(ctx.app).post('/api/v1/eval-runs').set(user.auth).send(lofo).expect(403);
    const imported = await request(ctx.app)
      .post('/api/v1/eval-runs')
      .set(admin.auth)
      .send(lofo)
      .expect(201);
    expect(imported.body).toMatchObject({ runs: 2, fixture: true });

    const res = await request(ctx.app).get('/api/v1/eval-runs').set(user.auth).expect(200);
    expect(res.body.report).toMatchObject({
      fixture: true,
      notice: expect.stringMatching(/fixture/i),
    });
    expect(res.body.report.runs[0]).toEqual({
      scheme: 'H-a',
      fold: 'F1',
      seed: 0,
      heldOut: true,
      metrics: { clean_auc: 0.81, flip_rate: 0.2 },
      ci95: { clean_auc: [0.7, 0.9], flip_rate: null },
    });
  });

  it('rejects documents that are not an eval_lofo.py report', async () => {
    const ctx = testApp();
    const admin = await ctx.user('admin');
    await request(ctx.app)
      .post('/api/v1/eval-runs')
      .set(admin.auth)
      .send({ schema: 2 })
      .expect(400);
    await request(ctx.app)
      .post('/api/v1/eval-runs')
      .set(admin.auth)
      .send({ ...lofo, runs: [{ scheme: 'H-a' }] })
      .expect(400);
  });
});

describe('unknown routes', () => {
  it('are 404 problem+json, with or without a token', async () => {
    const ctx = testApp();
    const { auth } = await ctx.user();
    const a = await request(ctx.app).get('/api/v1/nope').expect(404);
    expect(a.headers['content-type']).toMatch(/problem\+json/);
    await request(ctx.app).get('/api/v1/nope').set(auth).expect(404);
  });
});
