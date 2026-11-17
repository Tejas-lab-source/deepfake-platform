/**
 * POST /analyses/:id/ask: answers from the approved facts only.
 *
 * The template answer always works without an LLM. Groq is used only when GROQ_MODE=real and
 * a key is configured (off by default); it gets the question plus the facts JSON, nothing
 * else, and its answer is accepted only if every number in it appears in the facts and it
 * uses no banned wording. Otherwise the template answer is returned with the reason.
 */
import type { GroqClient } from '../providers/groq.js';
import { numberIsGrounded } from '../explain/validate.js';
import type { AskFacts } from './facts.js';
import { classify, templateAnswer, UNSUPPORTED, type Intent } from './templates.js';

export const ASK_PROMPT_VERSION = 'ask-v1';

export interface AskResult {
  answer: string;
  source: 'template' | 'groq';
  intent: Intent | 'unsupported';
  fallbackReason: string | null;
}

export interface AskDeps {
  mode: 'mock' | 'real' | 'off';
  groq: GroqClient | null;
}

const BANNED: Array<[RegExp, string]> = [
  [/\bauthentic/i, 'banned_word:authentic'],
  [/\bproven\b|\bproves?\b|\bproof that\b/i, 'banned_word:proven'],
  [/https?:\/\/|www\./i, 'contains_link'],
];

function jsonNumbers(value: unknown, out: number[] = []): number[] {
  if (typeof value === 'number' && Number.isFinite(value)) out.push(value);
  else if (Array.isArray(value)) value.forEach((v) => jsonNumbers(v, out));
  else if (value && typeof value === 'object')
    Object.values(value).forEach((v) => jsonNumbers(v, out));
  return out;
}

function jsonStrings(value: unknown, out: string[] = []): string[] {
  if (typeof value === 'string' && value.length >= 3) out.push(value);
  else if (Array.isArray(value)) value.forEach((v) => jsonStrings(v, out));
  else if (value && typeof value === 'object')
    Object.values(value).forEach((v) => jsonStrings(v, out));
  return out;
}

/** Reject an answer that invents numbers or makes unsupported claims. */
export function validateAnswer(
  answer: string,
  facts: AskFacts,
): { ok: true } | { ok: false; reason: string } {
  if (answer.trim().length < 10 || answer.length > 1500) return { ok: false, reason: 'length' };
  for (const [pattern, reason] of BANNED) if (pattern.test(answer)) return { ok: false, reason };
  if (facts.result.finalVerdict === 'REAL' && /\b(genuine|is real|are real)\b/i.test(answer)) {
    return { ok: false, reason: 'real_claimed_genuine' };
  }
  const allowed = jsonNumbers(facts);
  // Digits inside a string quoted from the facts (a model card version, a provider) are not
  // claims of their own.
  let text = answer;
  for (const quoted of jsonStrings(facts).sort((x, y) => y.length - x.length)) {
    text = text.split(quoted).join(' ');
  }
  for (const match of text.matchAll(/\d+(?:\.\d+)?/g)) {
    if (!numberIsGrounded(match[0], allowed))
      return { ok: false, reason: `ungrounded_number:${match[0]}` };
  }
  return { ok: true };
}

const SYSTEM = `You answer one question about the result of an automated deepfake check.
You receive the question and a JSON document with the result. Rules (all mandatory):
1. Use only facts and numbers in the JSON; never add any other number. Probabilities may be percentages.
2. Never say "authentic", "proven" or that anything is proven. Say the check is automated and is not proof.
3. finalVerdict REAL means "no manipulation found", never that the media is genuine.
4. If the question is not about this result, say you can only answer questions about this result.
5. Answer in at most four sentences.`;

const SCHEMA = {
  type: 'object',
  properties: { answer: { type: 'string' } },
  required: ['answer'],
  additionalProperties: false,
} as const;

export async function answerQuestion(
  deps: AskDeps,
  question: string,
  facts: AskFacts,
): Promise<AskResult> {
  const intent = classify(question);
  if (!intent)
    return { answer: UNSUPPORTED, source: 'template', intent: 'unsupported', fallbackReason: null };
  const template = templateAnswer(intent, facts);
  const fallback = (reason: string | null): AskResult => ({
    answer: template,
    source: 'template',
    intent,
    fallbackReason: reason,
  });

  if (deps.mode !== 'real') return fallback(null);
  if (!deps.groq?.configured) return fallback('not_configured');
  try {
    const r = await deps.groq.completeJson({
      system: SYSTEM,
      user: `Question: ${question}\n\nResult JSON:\n${JSON.stringify(facts)}`,
      schemaName: 'answer',
      schema: SCHEMA as unknown as Record<string, unknown>,
    });
    if (!r.ok) return fallback(`groq:${r.reason}`);
    const answer = (r.content as { answer?: unknown } | null)?.answer;
    if (typeof answer !== 'string') return fallback('invalid:schema');
    const check = validateAnswer(answer, facts);
    if (!check.ok) return fallback(`invalid:${check.reason}`);
    return { answer: answer.trim(), source: 'groq', intent, fallbackReason: null };
  } catch {
    return fallback('internal_error');
  }
}
