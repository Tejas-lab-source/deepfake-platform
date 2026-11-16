/**
 * Groq chat-completions client (OpenAI-compatible) for strict JSON output.
 *
 *   POST {GROQ_BASE_URL}/chat/completions   Authorization: Bearer <key>
 *   response_format: { type: "json_schema", json_schema: { name, strict: true, schema } }
 *
 * Text only: callers pass a system prompt and a JSON document; nothing else is sent.
 * Models are tried in order: a model that is missing, retired or rejects the request
 * (400/404/422) moves on to the next; auth failures stop at once; 429, 5xx, timeouts and
 * network errors are retried, then stop. Every HTTP attempt reserves budget first.
 */
import { z } from 'zod';
import { fetchWithTimeout, HttpError, readJson, withRetries, type FetchLike } from '../http.js';

export interface GroqOptions {
  apiKey: string;
  baseUrl: string;
  /** Primary model first, then fallbacks. */
  models: string[];
  timeoutMs: number;
  retries: number;
  maxCompletionTokens: number;
  reasoningEffort: 'low' | 'medium' | 'high';
  temperature: number;
  fetchImpl?: FetchLike;
  sleep?: (ms: number) => Promise<void>;
}

export interface JsonRequest {
  system: string;
  user: string;
  schemaName: string;
  schema: Record<string, unknown>;
  /** Called before every billable HTTP attempt; false stops with reason "budget". */
  reserveAttempt?: () => Promise<boolean>;
}

export type JsonResult =
  | { ok: true; content: unknown; model: string; usage: Usage | null; attempts: number }
  | { ok: false; reason: string; model: string | null; attempts: number };

export interface Usage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

const envelope = z
  .object({
    model: z.string().optional(),
    choices: z
      .array(
        z
          .object({
            finish_reason: z.string().nullable().optional(),
            message: z
              .object({
                content: z.string().nullable().optional(),
                refusal: z.string().nullable().optional(),
              })
              .passthrough(),
          })
          .passthrough(),
      )
      .min(1),
    usage: z
      .object({
        prompt_tokens: z.number().optional(),
        completion_tokens: z.number().optional(),
        total_tokens: z.number().optional(),
      })
      .passthrough()
      .optional(),
  })
  .passthrough();

class BudgetDenied extends Error {}

const NEXT_MODEL_STATUSES = new Set([400, 404, 422]);

export class GroqClient {
  private readonly fetchImpl: FetchLike;

  constructor(private readonly opts: GroqOptions) {
    this.fetchImpl = opts.fetchImpl ?? ((url, init) => fetch(url, init));
  }

  get configured(): boolean {
    return this.opts.apiKey.length > 0 && this.opts.models.length > 0;
  }

  async completeJson(req: JsonRequest): Promise<JsonResult> {
    if (!this.configured) return { ok: false, reason: 'not_configured', model: null, attempts: 0 };
    let attempts = 0;
    let last: JsonResult = { ok: false, reason: 'no_model', model: null, attempts };
    for (const model of this.opts.models) {
      let json: unknown;
      try {
        json = await withRetries(
          async () => {
            if (req.reserveAttempt && !(await req.reserveAttempt())) throw new BudgetDenied();
            attempts++;
            const res = await fetchWithTimeout(
              this.fetchImpl,
              'groq',
              `${this.opts.baseUrl.replace(/\/$/, '')}/chat/completions`,
              {
                method: 'POST',
                headers: {
                  authorization: `Bearer ${this.opts.apiKey}`,
                  'content-type': 'application/json',
                },
                body: JSON.stringify({
                  model,
                  messages: [
                    { role: 'system', content: req.system },
                    { role: 'user', content: req.user },
                  ],
                  response_format: {
                    type: 'json_schema',
                    json_schema: { name: req.schemaName, strict: true, schema: req.schema },
                  },
                  temperature: this.opts.temperature,
                  max_completion_tokens: this.opts.maxCompletionTokens,
                  reasoning_effort: this.opts.reasoningEffort,
                }),
              },
              this.opts.timeoutMs,
            );
            return readJson(res, 'groq');
          },
          { retries: this.opts.retries, sleep: this.opts.sleep },
        );
      } catch (err) {
        if (err instanceof BudgetDenied) return { ok: false, reason: 'budget', model, attempts };
        if (err instanceof HttpError) {
          const status = err.status ?? 0;
          last = {
            ok: false,
            reason: err.kind === 'http' ? `http_${status}` : err.kind,
            model,
            attempts,
          };
          if (err.kind === 'http' && NEXT_MODEL_STATUSES.has(status)) continue;
          return last;
        }
        return { ok: false, reason: 'client_error', model, attempts };
      }
      const parsed = envelope.safeParse(json);
      if (!parsed.success) return { ok: false, reason: 'invalid_response', model, attempts };
      const choice = parsed.data.choices[0]!;
      if (choice.message.refusal) return { ok: false, reason: 'refusal', model, attempts };
      if (choice.finish_reason && choice.finish_reason !== 'stop') {
        const reason =
          choice.finish_reason === 'length' ? 'truncated' : `finish_${choice.finish_reason}`;
        return { ok: false, reason, model, attempts };
      }
      const text = choice.message.content;
      if (!text) return { ok: false, reason: 'empty', model, attempts };
      try {
        return {
          ok: true,
          content: JSON.parse(text) as unknown,
          model: parsed.data.model ?? model,
          usage: parsed.data.usage ?? null,
          attempts,
        };
      } catch {
        return { ok: false, reason: 'invalid_json', model, attempts };
      }
    }
    return last;
  }
}
