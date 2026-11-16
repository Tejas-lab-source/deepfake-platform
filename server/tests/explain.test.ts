import { describe, expect, it, vi } from 'vitest';
import { explainResult, type ExplainerDeps } from '../src/explain/explainer.js';
import { buildResultJson, inputSha256 } from '../src/explain/prompt.js';
import { numberIsGrounded, validateExplanation } from '../src/explain/validate.js';
import type { ExplanationInput } from '../src/explainTemplate.js';
import type { GroqClient, JsonResult } from '../src/providers/groq.js';

const input: ExplanationInput = {
  final: { verdict: 'FAKE', badge: 'second_check_disagrees' },
  ours: {
    verdict: 'FAKE',
    probability: 0.9312,
    threshold: 0.5,
    confident: true,
    uncertainReason: null,
    spread: 0.41,
  },
  external: { status: 'ok', deepfakeScore: 0.1137, threshold: 0.9, verdict: 'REAL' },
  mock: false,
};
const extras = {
  ruleApplied: 'ours_confident_fake_external_disagrees',
  detectorScores: { gendHead: 2.61, sbi: 1.9 },
  aiGeneratedScore: 0.02,
  provider: 'hive',
  modelCardVersion: 'v1',
};
const facts = buildResultJson(input, extras);

const good = {
  summary: 'The automated check found signs of face manipulation in this upload.',
  why: 'Our pipeline scored it at 93% likelihood of manipulation against a 50% threshold, and was confident. The second check scored it 11% against its 90% threshold, so the two checks disagree; because our pipeline was confident, its verdict stands.',
  caveats: [
    'This is an automated check and is not proof either way.',
    'Beauty filters and compression can change detector scores.',
  ],
};

describe('result JSON sent to Groq', () => {
  it('contains only whitelisted facts: no image, path, file name or hash', () => {
    const tainted = {
      ...input,
      filePath: '/data/uploads/abc',
      external: {
        ...input.external!,
        mediaSha256: 'f'.repeat(64),
        rawResponse: { media_base64: 'AAAA' },
      },
    } as unknown as ExplanationInput;
    const json = JSON.stringify(
      buildResultJson(tainted, { ...extras, storageKey: 'abc', fileName: 'me.png' } as never),
    );
    for (const banned of [
      '/data/uploads',
      'abc',
      'f'.repeat(64),
      'media_base64',
      'me.png',
      'base64',
    ]) {
      expect(json).not.toContain(banned);
    }
    expect(Object.keys(facts).sort()).toEqual([
      'checksDisagree',
      'finalVerdict',
      'mockMode',
      'modelCardVersion',
      'ourPipeline',
      'ruleApplied',
      'secondCheck',
      'verificationBadge',
    ]);
  });

  it('hides scores of an unavailable second check and has a stable cache key', () => {
    const off = buildResultJson({
      ...input,
      external: { status: 'quota', deepfakeScore: null, threshold: 0.9, verdict: null },
    });
    expect(off.secondCheck).toMatchObject({ status: 'quota', deepfakeScore: null, verdict: null });
    expect(inputSha256(facts, ['m'])).toBe(inputSha256(buildResultJson(input, extras), ['m']));
    expect(inputSha256(facts, ['m'])).not.toBe(inputSha256(facts, ['other']));
  });
});

describe('validateExplanation', () => {
  it('accepts a grounded, compliant explanation', () => {
    expect(validateExplanation(good, facts)).toMatchObject({ ok: true });
  });

  it('accepts numbers as given, rounded or as percentages; rejects invented ones', () => {
    const allowed = [0.9312, 0.5, -2.61];
    for (const ok of ['93', '93.1', '0.93', '0.9312', '50', '0.5', '2.61', '2.6'])
      expect(numberIsGrounded(ok, allowed)).toBe(true);
    for (const bad of ['94', '17', '2', '3', '1', '0.94', '1000'])
      expect(numberIsGrounded(bad, allowed)).toBe(false);
    // Whole numbers: exact whole values are fine; rounding a fraction to one is not.
    expect(numberIsGrounded('3', [3])).toBe(true);
    expect(numberIsGrounded('1', [0.9])).toBe(false);
    expect(numberIsGrounded('90', [0.9])).toBe(true);
  });

  it.each([
    [
      'an invented number',
      { ...good, why: `${good.why} About 1 in 4 images like this are fake.` },
      'ungrounded_number:1',
    ],
    [
      '"authentic"',
      { ...good, caveats: ['The image may still be authentic.', ...good.caveats] },
      'banned_word:authentic',
    ],
    [
      '"proven"',
      { ...good, summary: 'The automated check has proven manipulation.' },
      'banned_word:proven',
    ],
    [
      'no "automated" statement',
      { ...good, summary: 'Signs of face manipulation were found.', caveats: ['Not proof.'] },
      'missing_automated_statement',
    ],
    [
      'a summary that contradicts the verdict',
      { ...good, summary: 'The automated check found nothing unusual.' },
      'verdict_wording',
    ],
    [
      'a missing disagreement',
      { ...good, why: 'Our pipeline scored it at 93% against a 50% threshold.' },
      'disagreement_not_explained',
    ],
    ['a link', { ...good, caveats: ['See https://example.com', 'automated'] }, 'contains_link'],
    ['an extra field', { ...good, confidence: 'high' }, 'schema'],
    ['a missing field', { summary: good.summary, why: good.why }, 'schema'],
    ['too many caveats', { ...good, caveats: Array(6).fill('An automated limitation.') }, 'schema'],
  ])('rejects %s', (_name, content, reason) => {
    expect(validateExplanation(content, facts)).toEqual({ ok: false, reason });
  });

  it('requires REAL to be worded as "no manipulation" and never as genuine', () => {
    const realFacts = buildResultJson({
      ...input,
      final: { verdict: 'REAL', badge: 'single_check' },
      external: null,
    });
    const real = {
      summary: 'The automated check found no manipulation.',
      why: 'Our pipeline scored it below the threshold of 50%.',
      caveats: ['Not proof either way.'],
    };
    expect(validateExplanation(real, realFacts)).toMatchObject({ ok: true });
    expect(
      validateExplanation({ ...real, why: `${real.why} The photo is real.` }, realFacts),
    ).toMatchObject({ reason: 'real_claimed_genuine' });
  });

  it('requires mock mode to be disclosed', () => {
    const mockFacts = buildResultJson({ ...input, mock: true }, extras);
    expect(validateExplanation(good, mockFacts)).toMatchObject({ reason: 'mock_not_stated' });
  });
});

function groqReturning(result: JsonResult) {
  const completeJson = vi.fn(async () => result);
  return { groq: { configured: true, completeJson } as unknown as GroqClient, completeJson };
}

const deps = (over: Partial<ExplainerDeps> = {}): ExplainerDeps => ({
  mode: 'real',
  groq: groqReturning({
    ok: true,
    content: good,
    model: 'openai/gpt-oss-120b',
    usage: { total_tokens: 150 },
    attempts: 1,
  }).groq,
  models: ['openai/gpt-oss-120b'],
  budget: null,
  cache: { find: async () => null },
  ...over,
});

describe('explainResult', () => {
  it('uses a valid Groq explanation', async () => {
    const r = await explainResult(deps(), input, extras);
    expect(r).toMatchObject({
      fallbackUsed: false,
      groqModel: 'openai/gpt-oss-120b',
      promptVersion: 'groq-explain-v1',
      summary: good.summary,
    });
    expect(r.inputSha256).toMatch(/^[0-9a-f]{64}$/);
  });

  it.each([
    ['mock', 'groq_mock'],
    ['off', 'groq_off'],
  ] as const)('uses the template in %s mode without calling Groq', async (mode, reason) => {
    const { groq, completeJson } = groqReturning({
      ok: false,
      reason: 'x',
      model: null,
      attempts: 0,
    });
    const r = await explainResult(deps({ mode, groq }), input, extras);
    expect(r).toMatchObject({
      fallbackUsed: true,
      fallbackReason: reason,
      promptVersion: 'template-v1',
    });
    expect(completeJson).not.toHaveBeenCalled();
  });

  it('uses the template without a key', async () => {
    const groq = { configured: false, completeJson: vi.fn() } as unknown as GroqClient;
    expect(await explainResult(deps({ groq }), input, extras)).toMatchObject({
      fallbackUsed: true,
      fallbackReason: 'not_configured',
    });
  });

  it('rejects the whole Groq response when validation fails', async () => {
    const bad = { ...good, why: `${good.why} Roughly 7 out of 10 such cases are fakes.` };
    const { groq } = groqReturning({
      ok: true,
      content: bad,
      model: 'm',
      usage: null,
      attempts: 1,
    });
    const r = await explainResult(deps({ groq }), input, extras);
    expect(r).toMatchObject({
      fallbackUsed: true,
      fallbackReason: 'invalid:ungrounded_number:7',
      groqModel: null,
    });
    expect(r.why).not.toContain('7 out of 10');
  });

  it('uses the template when Groq fails, and records why', async () => {
    const { groq } = groqReturning({ ok: false, reason: 'http_503', model: 'm', attempts: 3 });
    expect(await explainResult(deps({ groq }), input, extras)).toMatchObject({
      fallbackUsed: true,
      fallbackReason: 'groq:http_503',
    });
  });

  it('reuses a cached explanation for identical facts without calling Groq', async () => {
    const { groq, completeJson } = groqReturning({
      ok: true,
      content: good,
      model: 'm',
      usage: null,
      attempts: 1,
    });
    const cached = {
      ...good,
      groqModel: 'm',
      promptVersion: 'groq-explain-v1',
      fallbackUsed: false,
      fallbackReason: null,
      inputSha256: 'x',
      latencyMs: 5,
      usage: null,
    };
    const r = await explainResult(
      deps({ groq, cache: { find: async () => cached } }),
      input,
      extras,
    );
    expect(r).toMatchObject({ fallbackUsed: false, summary: good.summary });
    expect(completeJson).not.toHaveBeenCalled();
  });

  it('never throws: internal failures fall back to the template', async () => {
    const onError = vi.fn();
    const cache = {
      find: async () => {
        throw new Error('mongo down');
      },
    };
    const r = await explainResult(deps({ cache }), input, extras, onError);
    expect(r).toMatchObject({ fallbackUsed: true, fallbackReason: 'internal_error' });
    expect(onError).toHaveBeenCalledOnce();
  });
});
