import { config, secretValues } from './config.js';

export const REDACTED = '[REDACTED]';

// Credential shapes worth hiding even when they are not our own configured values.
const PATTERNS: RegExp[] = [
  /\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{8,}/gi,
  /\bgsk_[A-Za-z0-9]{10,}/g, // Groq key format
];

const registered = new Set<string>();

/** Add values to hide. Values shorter than 8 characters are ignored (too likely to collide). */
export function registerSecrets(values: Iterable<string>): void {
  for (const value of values) {
    if (value && value.length >= 8) registered.add(value);
  }
}

registerSecrets(secretValues(config));

/** Replace every registered secret and every credential-shaped token in `text`. */
export function scrub(text: string): string {
  let out = text;
  for (const secret of registered) {
    if (out.includes(secret)) out = out.split(secret).join(REDACTED);
  }
  for (const pattern of PATTERNS) {
    out = out.replace(pattern, (match) =>
      /^(bearer|basic)\s/i.test(match) ? `${match.split(/\s+/)[0]} ${REDACTED}` : REDACTED,
    );
  }
  return out;
}

const SECRET_KEY = /^(authorization|api[-_]?key|x-api-key|.*_API_KEY|password|token)$/i;

/** Deep copy of `value` with secrets scrubbed from every string and credential-named fields
 * removed. Safe on cycles and very deep objects. */
export function scrubDeep(value: unknown, depth = 0, seen = new WeakSet<object>()): unknown {
  if (typeof value === 'string') return scrub(value);
  if (value === null || typeof value !== 'object') return value;
  if (depth > 12) return '[TRUNCATED]';
  if (seen.has(value)) return '[CIRCULAR]';
  seen.add(value);
  if (value instanceof Error) {
    return {
      type: value.name,
      message: scrub(value.message),
      stack: value.stack && scrub(value.stack),
    };
  }
  if (Array.isArray(value)) return value.map((v) => scrubDeep(v, depth + 1, seen));
  const out: Record<string, unknown> = {};
  for (const [key, v] of Object.entries(value)) {
    out[key] = SECRET_KEY.test(key) ? REDACTED : scrubDeep(v, depth + 1, seen);
  }
  return out;
}
