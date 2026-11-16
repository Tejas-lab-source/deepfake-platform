/**
 * Explanation for one analysis: Groq when available and valid, otherwise the template.
 * It never throws: every failure ends in the template with a recorded reason.
 */
import type { DailyBudget } from '../budget.js';
import {
  buildTemplateExplanation,
  TEMPLATE_PROMPT_VERSION,
  type Explanation,
  type ExplanationInput,
} from '../explainTemplate.js';
import type { GroqClient, Usage } from '../providers/groq.js';
import {
  buildResultJson,
  EXPLANATION_SCHEMA,
  GROQ_PROMPT_VERSION,
  inputSha256,
  SYSTEM_PROMPT,
  type ResultExtras,
} from './prompt.js';
import { validateExplanation } from './validate.js';

export interface StoredExplanation extends Explanation {
  groqModel: string | null;
  promptVersion: string;
  fallbackUsed: boolean;
  fallbackReason: string | null;
  inputSha256: string | null;
  latencyMs: number | null;
  usage: Usage | null;
}

export interface ExplanationCache {
  /** A previous Groq explanation (not a fallback) for exactly the same facts. */
  find(inputSha256: string): Promise<StoredExplanation | null>;
}

export interface ExplainerDeps {
  mode: 'mock' | 'real' | 'off';
  groq: GroqClient | null;
  models: string[];
  budget: DailyBudget | null;
  cache: ExplanationCache;
  now?: () => number;
}

function template(input: ExplanationInput, reason: string, sha: string | null): StoredExplanation {
  return {
    ...buildTemplateExplanation(input),
    groqModel: null,
    promptVersion: TEMPLATE_PROMPT_VERSION,
    fallbackUsed: true,
    fallbackReason: reason,
    inputSha256: sha,
    latencyMs: null,
    usage: null,
  };
}

export async function explainResult(
  deps: ExplainerDeps,
  input: ExplanationInput,
  extras: ResultExtras,
  onError: (err: unknown) => void = () => undefined,
): Promise<StoredExplanation> {
  if (deps.mode !== 'real') return template(input, `groq_${deps.mode}`, null);
  if (!deps.groq?.configured) return template(input, 'not_configured', null);
  let sha: string | null = null;
  try {
    const facts = buildResultJson(input, extras);
    sha = inputSha256(facts, deps.models);
    const cached = await deps.cache.find(sha);
    if (cached) return { ...cached, inputSha256: sha };

    const now = deps.now ?? Date.now;
    const started = now();
    const budget = deps.budget;
    let denied: string | null = null;
    const result = await deps.groq.completeJson({
      system: SYSTEM_PROMPT,
      user: `Result JSON:\n${JSON.stringify(facts)}`,
      schemaName: 'explanation',
      schema: EXPLANATION_SCHEMA as unknown as Record<string, unknown>,
      reserveAttempt: budget
        ? async () => {
            const r = await budget.reserve({});
            if (!r.allowed) denied = r.reason;
            return r.allowed;
          }
        : undefined,
    });
    if (!result.ok) {
      return template(input, denied ? `budget:${denied}` : `groq:${result.reason}`, sha);
    }
    const checked = validateExplanation(result.content, facts);
    if (!checked.ok) return template(input, `invalid:${checked.reason}`, sha);
    return {
      ...checked.explanation,
      groqModel: result.model,
      promptVersion: GROQ_PROMPT_VERSION,
      fallbackUsed: false,
      fallbackReason: null,
      inputSha256: sha,
      latencyMs: now() - started,
      usage: result.usage,
    };
  } catch (err) {
    onError(err);
    return template(input, 'internal_error', sha);
  }
}
