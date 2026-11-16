import type { Badge, Verdict } from './types.js';

export const TEMPLATE_PROMPT_VERSION = 'template-v1';

export interface ExplanationInput {
  final: { verdict: Verdict; badge: Badge };
  ours: {
    verdict: Verdict;
    probability: number | null;
    threshold: number;
    confident: boolean;
    uncertainReason: string | null;
    spread: number | null;
  };
  external: {
    status: string;
    deepfakeScore: number | null;
    threshold: number;
    verdict: 'FAKE' | 'REAL' | null;
  } | null;
  mock: boolean;
}

export interface Explanation {
  summary: string;
  why: string;
  caveats: string[];
}

const SUMMARY: Record<Verdict, string> = {
  FAKE: 'The automated check found signs of face manipulation.',
  REAL: 'The automated check found no manipulation.',
  UNCERTAIN: 'The automated check could not reach a reliable verdict.',
};

const UNCERTAIN_REASONS: Record<string, string> = {
  near_threshold: 'the score is too close to the decision threshold',
  unstable_under_beautification:
    'the score moved too much when cosmetic filters were applied to copies of the face',
  variants_disagree: 'the filtered copies of the face did not agree on a verdict',
  detectors_disagree: 'the two detectors strongly disagreed with each other',
  no_face_detected: 'no face was detected, so nothing could be scored',
};

const pct = (x: number): string => `${Math.round(x * 100)}%`;

/** Plain-English explanation built only from the result JSON. Used when Groq is unavailable. */
export function buildTemplateExplanation(input: ExplanationInput): Explanation {
  const { final, ours, external } = input;
  const why: string[] = [];

  if (ours.probability !== null) {
    why.push(
      `Our pipeline scored this upload at ${pct(ours.probability)} likelihood of manipulation, ` +
        `against a threshold of ${pct(ours.threshold)}.`,
    );
  }
  if (ours.verdict === 'UNCERTAIN') {
    const reason = ours.uncertainReason ? UNCERTAIN_REASONS[ours.uncertainReason] : undefined;
    why.push(`Our pipeline returned UNCERTAIN because ${reason ?? 'the result was not stable'}.`);
  } else if (!ours.confident) {
    why.push('That score is not far enough from the threshold to count as confident.');
  }
  if (ours.spread !== null) {
    why.push(
      `Across the filtered copies of the face, the score spread was ${ours.spread.toFixed(2)}.`,
    );
  }

  if (external && external.status === 'ok' && external.deepfakeScore !== null) {
    const says = external.verdict === 'FAKE' ? 'a deepfake' : 'not a deepfake';
    why.push(
      `The second check scored it ${pct(external.deepfakeScore)} (its threshold is ` +
        `${pct(external.threshold)}), which reads as ${says}.`,
    );
    if (final.badge === 'verified_by_2') why.push('Both checks agree.');
    if (final.badge === 'second_check_disagrees') {
      why.push('The two checks disagree. Our pipeline was confident, so its verdict stands.');
    }
    if (final.badge === 'checks_disagree') {
      why.push(
        'The two checks disagree and our pipeline was not confident, so the result is UNCERTAIN.',
      );
    }
  } else {
    why.push(
      'No second check is available for this upload, so the verdict rests on our pipeline alone.',
    );
  }

  const caveats = [
    'This is an automated check. It is not proof either way.',
    'Beauty filters, compression and low resolution can change detector scores.',
  ];
  if (final.verdict === 'REAL') {
    caveats.push('Finding no manipulation does not show that the image is original.');
  }
  if (input.mock) {
    caveats.push(
      'MOCK MODE: these scores are placeholders derived from the file hash, not from a model.',
    );
  }
  return { summary: SUMMARY[final.verdict], why: why.join(' '), caveats };
}
