import { config } from '../config.js';
import type { MediaKind, MlResult } from '../types.js';

export class MlServiceError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function post(path: string, body: unknown): Promise<MlResult> {
  const res = await fetch(`${config.ML_SERVICE_URL}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-internal-secret': config.INTERNAL_SECRET },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(config.ML_TIMEOUT_MS),
  });
  if (!res.ok) {
    const text = await res.text();
    let detail = text;
    try {
      const parsed = JSON.parse(text) as { detail?: unknown };
      if (typeof parsed.detail === 'string') detail = parsed.detail;
    } catch {
      // not JSON: keep the raw text
    }
    throw new MlServiceError(res.status, `ML service ${res.status}: ${detail}`);
  }
  return (await res.json()) as MlResult;
}

/** `storageKey` is the file's path relative to the shared upload volume. */
export function analyzeMedia(kind: MediaKind, storageKey: string): Promise<MlResult> {
  return kind === 'image'
    ? post('/analyze/image', { path: storageKey, k: config.K, explain: true })
    : post('/analyze/video', { path: storageKey, k: config.K });
}
