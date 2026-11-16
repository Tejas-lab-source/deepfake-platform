import { describe, expect, it } from 'vitest';
import { buildTemplateExplanation, type ExplanationInput } from '../src/explainTemplate.js';

const base: ExplanationInput = {
  final: { verdict: 'REAL', badge: 'second_check_disagrees' },
  ours: {
    verdict: 'REAL',
    probability: 0.12,
    threshold: 0.5,
    confident: true,
    uncertainReason: null,
    spread: 0.31,
  },
  external: { status: 'ok', deepfakeScore: 0.95, threshold: 0.9, verdict: 'FAKE' },
  mock: false,
};

const text = (input: ExplanationInput): string => {
  const e = buildTemplateExplanation(input);
  return [e.summary, e.why, ...e.caveats].join(' ');
};

describe('buildTemplateExplanation', () => {
  it('covers a disagreement with the second check', () => {
    expect(text(base)).toMatch(/two checks disagree/i);
    expect(text(base)).toContain('95%');
  });

  it('always says the check is automated and avoids banned wording', () => {
    const verdicts = ['FAKE', 'REAL', 'UNCERTAIN'] as const;
    for (const verdict of verdicts) {
      const out = text({ ...base, final: { verdict, badge: 'single_check' }, external: null });
      expect(out).toMatch(/automated/i);
      expect(out).not.toMatch(/authentic|proven fake/i);
    }
  });

  it('explains an UNCERTAIN reason and flags mock mode', () => {
    const out = text({
      ...base,
      final: { verdict: 'UNCERTAIN', badge: 'hive_hint' },
      ours: {
        ...base.ours,
        verdict: 'UNCERTAIN',
        uncertainReason: 'unstable_under_beautification',
      },
      mock: true,
    });
    expect(out).toMatch(/cosmetic filters/);
    expect(out).toMatch(/MOCK MODE/);
  });
});
