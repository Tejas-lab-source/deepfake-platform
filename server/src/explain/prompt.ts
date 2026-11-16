/**
 * What Groq is told and shown. Groq receives only the whitelisted result JSON built here:
 * never image bytes, file paths, file names, storage keys or hashes.
 */
import { createHash } from 'node:crypto';
import type { ExplanationInput } from '../explainTemplate.js';

export const GROQ_PROMPT_VERSION = 'groq-explain-v1';

/** Extra numbers available to the explanation, beyond the template's input. */
export interface ResultExtras {
  ruleApplied?: string | null;
  detectorScores?: { gendHead?: number | null; sbi?: number | null } | null;
  aiGeneratedScore?: number | null;
  provider?: string | null;
  modelCardVersion?: string | null;
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** The only document Groq sees. Built field by field: anything not listed is dropped. */
export function buildResultJson(input: ExplanationInput, extras: ResultExtras = {}) {
  const ext = input.external;
  const ok = ext !== null && ext.status === 'ok' && ext.deepfakeScore !== null;
  return {
    finalVerdict: input.final.verdict,
    verificationBadge: input.final.badge,
    ruleApplied: extras.ruleApplied ?? null,
    checksDisagree:
      input.final.badge === 'second_check_disagrees' || input.final.badge === 'checks_disagree',
    ourPipeline: {
      verdict: input.ours.verdict,
      probabilityOfManipulation: num(input.ours.probability),
      threshold: num(input.ours.threshold),
      confident: input.ours.confident === true,
      uncertainReason: input.ours.uncertainReason ?? null,
      scoreSpreadAcrossFilteredCopies: num(input.ours.spread),
      detectorLogits: {
        gend: num(extras.detectorScores?.gendHead),
        sbi: num(extras.detectorScores?.sbi),
      },
    },
    secondCheck: ext
      ? {
          provider: extras.provider === 'hive-mock' ? 'Hive (mock)' : 'Hive',
          status: ext.status,
          deepfakeScore: ok ? num(ext.deepfakeScore) : null,
          aiGeneratedScore: ok ? num(extras.aiGeneratedScore) : null,
          threshold: num(ext.threshold),
          verdict: ok ? ext.verdict : null,
        }
      : null,
    mockMode: input.mock === true,
    modelCardVersion: extras.modelCardVersion ?? null,
  };
}

export type ResultJson = ReturnType<typeof buildResultJson>;

/** Cache key: the exact facts, the models and the prompt version. */
export function inputSha256(result: ResultJson, models: string[]): string {
  return createHash('sha256')
    .update(JSON.stringify({ result, models, prompt: GROQ_PROMPT_VERSION }))
    .digest('hex');
}

export const EXPLANATION_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    why: { type: 'string' },
    caveats: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'why', 'caveats'],
  additionalProperties: false,
} as const;

export const SYSTEM_PROMPT = `You explain the result of an automated deepfake check to a non-expert.
You receive one JSON document describing the result. Write JSON with:
- "summary": one or two sentences stating the final verdict.
- "why": a short paragraph explaining how the verdict was reached.
- "caveats": two to four short limitations.

Rules (all mandatory):
1. Use only facts and numbers that appear in the JSON. Do not introduce any other number, date,
   count or statistic. Probabilities and thresholds may be written as percentages.
   Avoid digits that are not taken from the JSON (write "two checks", not "2 checks").
2. Never use the word "authentic" and never say "proven fake" or that anything is proven.
3. State clearly that this is an automated check and is not proof.
4. finalVerdict REAL means "no manipulation found"; say exactly that, never that the image is
   genuine or real. FAKE means signs of manipulation were found. UNCERTAIN means no reliable
   verdict could be reached.
5. If checksDisagree is true, explain that the second check disagreed and what happened.
6. If mockMode is true, say the scores are placeholders, not model outputs.
7. Do not describe the image, the person, or anything not in the JSON.`;
