/**
 * Ask, PDF report and Filter Lab routes (in-memory repositories; not MongoDB tests).
 */
import { createRequire } from 'node:module';
import path from 'node:path';
import request from 'supertest';
import { describe, expect, it } from 'vitest';
import { MlServiceError } from '../../src/providers/mlClient.js';
import { memoryRateLimiter } from '../../src/rateLimit.js';
import { seedFinished, SECRET_SHA, testApp, VIDEO_ML } from '../helpers/testApp.js';

/** Plain text of every page of a PDF (pdf.js, the Firefox PDF engine). */
async function pdfText(bytes: Buffer): Promise<string> {
  const pdfjs = await import('pdfjs-dist/legacy/build/pdf.mjs');
  const fonts = path.join(
    path.dirname(createRequire(import.meta.url).resolve('pdfjs-dist/package.json')),
    'standard_fonts/',
  );
  const doc = await pdfjs.getDocument({ data: new Uint8Array(bytes), standardFontDataUrl: fonts })
    .promise;
  const pages: string[] = [];
  for (let i = 1; i <= doc.numPages; i++) {
    const content = await (await doc.getPage(i)).getTextContent();
    pages.push(content.items.map((it) => ('str' in it ? it.str : '')).join(' '));
  }
  return pages.join('\n').replace(/\s+/g, ' ');
}

const binary = (res: request.Response, cb: (err: Error | null, body: Buffer) => void) => {
  const chunks: Buffer[] = [];
  res.on('data', (c: Buffer) => chunks.push(c));
  res.on('end', () => cb(null, Buffer.concat(chunks)));
};

describe('POST /analyses/:id/ask', () => {
  it('answers the owner from the analysis facts, with Groq off', async () => {
    const ctx = testApp();
    const me = await ctx.user();
    const { analysis } = await seedFinished(ctx, me.id, { kind: 'video', ml: VIDEO_ML });
    const res = await request(ctx.app)
      .post(`/api/v1/analyses/${analysis.id}/ask`)
      .set(me.auth)
      .send({ question: 'Why was this video flagged?' })
      .expect(200);
    expect(res.body).toEqual({
      answer: expect.stringContaining('most suspicious frame is at 2.5 s'),
      source: 'template',
      intent: 'why',
      fallbackReason: null,
    });
    const off = await request(ctx.app)
      .post(`/api/v1/analyses/${analysis.id}/ask`)
      .set(me.auth)
      .send({ question: 'Tell me a joke about cats' })
      .expect(200);
    expect(off.body.intent).toBe('unsupported');
  });

  it('enforces auth, ownership, completion and question length', async () => {
    const ctx = testApp();
    const me = await ctx.user();
    const other = await ctx.user();
    const { analysis } = await seedFinished(ctx, me.id);
    const running = (await seedFinished(ctx, me.id, { status: 'processing' })).analysis;
    const url = `/api/v1/analyses/${analysis.id}/ask`;
    const q = { question: 'Is it fake?' };
    await request(ctx.app).post(url).send(q).expect(401);
    await request(ctx.app).post(url).set(other.auth).send(q).expect(404);
    await request(ctx.app)
      .post(url)
      .set(me.auth)
      .send({ question: 'x'.repeat(501) })
      .expect(400);
    await request(ctx.app).post(url).set(me.auth).send({ question: 'hi' }).expect(400);
    await request(ctx.app).post(url).set(me.auth).send({}).expect(400);
    await request(ctx.app)
      .post(`/api/v1/analyses/${running.id}/ask`)
      .set(me.auth)
      .send(q)
      .expect(409);
  });

  it('rate-limits questions per user', async () => {
    const limits = {
      auth: memoryRateLimiter(100, 60),
      upload: memoryRateLimiter(100, 60),
      ask: memoryRateLimiter(2, 60),
      compare: memoryRateLimiter(100, 60),
    };
    const ctx = testApp({ rateLimiters: limits });
    const me = await ctx.user();
    const other = await ctx.user();
    const { analysis } = await seedFinished(ctx, me.id);
    const mine = (await seedFinished(ctx, other.id)).analysis;
    const ask = (auth: { Authorization: string }, id: string) =>
      request(ctx.app)
        .post(`/api/v1/analyses/${id}/ask`)
        .set(auth)
        .send({ question: 'Is it fake?' });
    expect((await ask(me.auth, analysis.id)).status).toBe(200);
    expect((await ask(me.auth, analysis.id)).status).toBe(200);
    const limited = await ask(me.auth, analysis.id);
    expect(limited.status).toBe(429);
    expect(limited.body.title).toBe('Too Many Requests');
    expect(limited.headers['retry-after']).toEqual(expect.any(String));
    // The limit is per user.
    expect((await ask(other.auth, mine.id)).status).toBe(200);
  });
});

describe('GET /analyses/:id/report.pdf', () => {
  it('contains the required findings and none of the forbidden data', async () => {
    const ctx = testApp();
    const email = 'reviewer.person@example.test';
    const user = await ctx.repos.users.create({ email, passwordHash: 'x' });
    const { signTokens } = await import('../../src/middleware/auth.js');
    const auth = {
      Authorization: `Bearer ${signTokens({ id: user.id, role: 'user' }).accessToken}`,
    };
    const { analysis, media } = await seedFinished(ctx, user.id, {
      kind: 'video',
      ml: VIDEO_ML,
      externalCheckRequested: true,
    });
    ctx.repos.store.externalChecks.set('c'.repeat(24), {
      id: 'c'.repeat(24),
      mediaSha256: SECRET_SHA,
      provider: 'hive',
      status: 'ok',
      deepfakeScore: 0.12,
      aiGeneratedScore: 0.03,
      threshold: 0.9,
      verdict: 'REAL',
      latencyMs: 300,
      reason: null,
      frames: [],
    });
    await ctx.repos.analyses.update(analysis.id, { externalCheckId: 'c'.repeat(24) });
    await ctx.repos.explanations.upsert(analysis.id, {
      groqModel: null,
      promptVersion: 'template-v1',
      summary: 'The automated check found signs of face manipulation.',
      why: 'Template why text.',
      caveats: ['Automated, not proof.'],
      fallbackUsed: true,
      fallbackReason: 'groq_off',
      inputSha256: 'f'.repeat(64),
      latencyMs: null,
      usage: null,
    });

    const res = await request(ctx.app)
      .get(`/api/v1/analyses/${analysis.id}/report.pdf`)
      .set(auth)
      .buffer(true)
      .parse(binary)
      .expect(200);
    expect(res.headers['content-type']).toBe('application/pdf');
    expect(res.headers['content-disposition']).toBe(
      `attachment; filename="analysis-${analysis.id}.pdf"`,
    );
    const text = await pdfText(res.body as Buffer);
    for (const required of [
      'Automated deepfake analysis report',
      `Analysis ID: ${analysis.id}`,
      'FAKE: signs of manipulation were found',
      'Verification badge: single_check',
      'Rule applied: external_unavailable:opted_out',
      'Probability of manipulation: 91.0%',
      'Decision threshold: 50.0%',
      'Model card version: mc-test',
      'Deepfake score: 12.0%',
      'Provider threshold: 90.0%',
      'Frames sampled: 4',
      'Most suspicious frame: #25 at 2.50 s',
      '#15',
      'excluded',
      'The automated check found signs of face manipulation.',
      'Automated finding, not proof',
      'Page 1 of',
    ]) {
      expect(text, required).toContain(required);
    }
    for (const forbidden of [
      email,
      SECRET_SHA,
      'f'.repeat(64),
      media.storageKey,
      ctx.uploadDir,
      'rawResponse',
      'Bearer',
      'groq_off',
    ]) {
      expect(text, forbidden).not.toContain(forbidden);
    }
  });

  it("is only for the owner's finished analyses", async () => {
    const ctx = testApp();
    const me = await ctx.user();
    const other = await ctx.user();
    const { analysis } = await seedFinished(ctx, me.id);
    const running = (await seedFinished(ctx, me.id, { status: 'processing' })).analysis;
    await request(ctx.app).get(`/api/v1/analyses/${analysis.id}/report.pdf`).expect(401);
    await request(ctx.app)
      .get(`/api/v1/analyses/${analysis.id}/report.pdf`)
      .set(other.auth)
      .expect(404);
    await request(ctx.app)
      .get(`/api/v1/analyses/${running.id}/report.pdf`)
      .set(me.auth)
      .expect(409);
    const img = await request(ctx.app)
      .get(`/api/v1/analyses/${analysis.id}/report.pdf`)
      .set(me.auth)
      .buffer(true)
      .parse(binary)
      .expect(200);
    const text = await pdfText(img.body as Buffer);
    expect(text).toContain('Not requested for this upload');
    expect(text).not.toContain('Video frames');
  });
});

describe('POST /analyses/:id/compare (Filter Lab)', () => {
  const compareResult = (heat: string, img: string) => ({
    family: 'F1',
    strength: 0.6,
    threshold: 0.5,
    clean: {
      verdict: 'FAKE',
      probability: 0.62,
      confident: false,
      scores: { gendHead: 0.5, sbi: 0.4 },
      image: img,
      heatmap: heat,
    },
    filtered: {
      verdict: 'REAL',
      probability: 0.41,
      confident: false,
      scores: { gendHead: -0.3, sbi: 0.2 },
      image: img,
      heatmap: heat,
    },
    flipped: true,
    heatmapIoU: 0.37,
    modelCardVersion: 'mc-test',
  });

  it('stores the comparison, highlights a flip, and serves its images to the owner', async () => {
    const ctx = testApp();
    const me = await ctx.user();
    const { analysis, media } = await seedFinished(ctx, me.id);
    const heat = ctx.derived();
    const img = ctx.derived();
    ctx.ml.compare.mockResolvedValue(compareResult(heat, img));
    const res = await request(ctx.app)
      .post(`/api/v1/analyses/${analysis.id}/compare`)
      .set(me.auth)
      .send({ family: 'F1', strength: 0.6 })
      .expect(201);
    expect(ctx.ml.compare).toHaveBeenCalledWith(media.storageKey, 'F1', 0.6);
    expect(res.body).toMatchObject({
      family: 'F1',
      strength: 0.6,
      flipped: true,
      heatmapIoU: 0.37,
      clean: { verdict: 'FAKE', probability: 0.62 },
      filtered: { verdict: 'REAL', probability: 0.41 },
    });
    const list = await request(ctx.app)
      .get(`/api/v1/analyses/${analysis.id}/comparisons`)
      .set(me.auth)
      .expect(200);
    expect(list.body.items).toHaveLength(1);
    await request(ctx.app)
      .get(`/api/v1/analyses/${analysis.id}/heatmaps/${heat.slice(8)}`)
      .set(me.auth)
      .expect(200);
  });

  it('says clearly when the real pipeline is not available, and never invents a result', async () => {
    const ctx = testApp();
    const me = await ctx.user();
    const { analysis } = await seedFinished(ctx, me.id);
    ctx.ml.compare.mockRejectedValue(
      new MlServiceError(
        501,
        'ML service 501: Filters need the real pipeline; set MOCK_MODE=false.',
      ),
    );
    const res = await request(ctx.app)
      .post(`/api/v1/analyses/${analysis.id}/compare`)
      .set(me.auth)
      .send({ family: 'F2', strength: 0.5 })
      .expect(501);
    expect(res.body.detail).toBe('Filters need the real pipeline; set MOCK_MODE=false.');
    expect(ctx.repos.store.comparisons.size).toBe(0);
    ctx.ml.compare.mockRejectedValue(new MlServiceError(422, 'ML service 422: no face detected'));
    const noFace = await request(ctx.app)
      .post(`/api/v1/analyses/${analysis.id}/compare`)
      .set(me.auth)
      .send({ family: 'F2', strength: 0.5 })
      .expect(422);
    expect(noFace.body.detail).toBe('no face detected');
  });

  it('validates input, ownership, media type and that the upload still exists', async () => {
    const ctx = testApp();
    const me = await ctx.user();
    const other = await ctx.user();
    const { analysis, media } = await seedFinished(ctx, me.id);
    const video = (await seedFinished(ctx, me.id, { kind: 'video', ml: VIDEO_ML })).analysis;
    const url = `/api/v1/analyses/${analysis.id}/compare`;
    await request(ctx.app).post(url).set(me.auth).send({ family: 'F9', strength: 0.5 }).expect(400);
    await request(ctx.app).post(url).set(me.auth).send({ family: 'F1', strength: 2 }).expect(400);
    await request(ctx.app)
      .post(url)
      .set(other.auth)
      .send({ family: 'F1', strength: 0.5 })
      .expect(404);
    await request(ctx.app)
      .post(`/api/v1/analyses/${video.id}/compare`)
      .set(me.auth)
      .send({ family: 'F1', strength: 0.5 })
      .expect(422);
    await ctx.repos.media.delete(media.id);
    await request(ctx.app).post(url).set(me.auth).send({ family: 'F1', strength: 0.5 }).expect(410);
    expect(ctx.ml.compare).not.toHaveBeenCalled();
  });
});
