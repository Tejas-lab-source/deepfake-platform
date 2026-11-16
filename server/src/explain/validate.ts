/**
 * Validation of a Groq explanation. Any failure rejects the whole response (the caller then
 * uses the template). Checks: exact schema, lengths, banned wording, "automated" stated,
 * verdict wording consistent, disagreement covered, no links, and every number in the text
 * present in the result JSON (as given, rounded, or as a percentage).
 */
import { z } from 'zod';
import type { Explanation } from '../explainTemplate.js';
import type { ResultJson } from './prompt.js';

const schema = z
  .object({
    summary: z.string().trim().min(10).max(400),
    why: z.string().trim().min(10).max(1500),
    caveats: z.array(z.string().trim().min(3).max(300)).min(1).max(5),
  })
  .strict();

export type Validation = { ok: true; explanation: Explanation } | { ok: false; reason: string };

const BANNED: Array<[RegExp, string]> = [
  [/\bauthentic/i, 'banned_word:authentic'],
  [/\bproven\b|\bproves?\b|\bproof that\b/i, 'banned_word:proven'],
  [/https?:\/\/|www\./i, 'contains_link'],
];

const VERDICT_WORDING: Record<string, RegExp> = {
  FAKE: /manipulat|fake/i,
  REAL: /no manipulation/i,
  UNCERTAIN: /uncertain|could not|couldn't|inconclusive|no reliable/i,
};

function jsonNumbers(value: unknown, out: number[] = []): number[] {
  if (typeof value === 'number' && Number.isFinite(value)) out.push(value);
  else if (Array.isArray(value)) value.forEach((v) => jsonNumbers(v, out));
  else if (value && typeof value === 'object')
    Object.values(value).forEach((v) => jsonNumbers(v, out));
  return out;
}

/** Is `text` (a number written in the explanation) one of the JSON's numbers? Accepted:
 * the value rounded to the written number of decimals, or the value as a percentage (x100)
 * rounded likewise. A whole number written without decimals must be an exact whole-number
 * value or a percentage: 0.9 does not license "1", and 1.9 does not license "2". Signs are
 * ignored: prose may say "2.1 below zero" for -2.1. */
export function numberIsGrounded(text: string, allowed: number[]): boolean {
  const written = Math.abs(Number(text));
  const decimals = text.includes('.') ? text.split('.')[1]!.length : 0;
  const scale = 10 ** decimals;
  const roundsTo = (value: number) => Math.abs(Math.round(value * scale) / scale - written) < 1e-9;
  return allowed.some((a) => {
    const v = Math.abs(a);
    const percent = v * 100;
    if (roundsTo(percent)) return true;
    return decimals > 0 ? roundsTo(v) : Number.isInteger(v) && v === written;
  });
}

export function validateExplanation(content: unknown, facts: ResultJson): Validation {
  const parsed = schema.safeParse(content);
  if (!parsed.success) return { ok: false, reason: 'schema' };
  const e = parsed.data;
  const all = [e.summary, e.why, ...e.caveats].join('\n');

  for (const [pattern, reason] of BANNED) if (pattern.test(all)) return { ok: false, reason };
  if (!/\bautomated\b/i.test(all)) return { ok: false, reason: 'missing_automated_statement' };
  const wording = VERDICT_WORDING[facts.finalVerdict];
  if (wording && !wording.test(e.summary)) return { ok: false, reason: 'verdict_wording' };
  if (facts.finalVerdict === 'REAL' && /\b(genuine|is real|are real)\b/i.test(all)) {
    return { ok: false, reason: 'real_claimed_genuine' };
  }
  if (facts.checksDisagree && !/disagree|differ|conflict|did not agree|didn't agree/i.test(e.why)) {
    return { ok: false, reason: 'disagreement_not_explained' };
  }
  if (facts.mockMode && !/mock|placeholder/i.test(all)) {
    return { ok: false, reason: 'mock_not_stated' };
  }
  const allowed = jsonNumbers(facts);
  for (const match of all.matchAll(/\d+(?:\.\d+)?/g)) {
    if (!numberIsGrounded(match[0], allowed)) {
      return { ok: false, reason: `ungrounded_number:${match[0]}` };
    }
  }
  return { ok: true, explanation: { summary: e.summary, why: e.why, caveats: e.caveats } };
}
