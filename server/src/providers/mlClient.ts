/**
 * Client for the private ML service. Every call carries the internal secret; responses are
 * returned as parsed JSON. Errors become MlServiceError with the service's status and its
 * `detail` text (never the secret).
 */
import { config } from '../config.js';
import type { FetchLike } from '../http.js';
import type { MediaKind, MlResult } from '../types.js';

export class MlServiceError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export interface MlOptions {
  baseUrl: string;
  secret: string;
  timeoutMs: number;
  /** Shorter timeout for metadata calls (model info, progress). */
  metaTimeoutMs?: number;
  k: number;
  /** Copies per sampled video frame. */
  videoK: number;
  fetchImpl?: FetchLike;
}

export interface AnalyzeOptions {
  /** Lets the caller poll /progress/{id} while a video is being analysed. */
  progressId?: string;
}

export interface MlProgress {
  stage: string;
  done: number;
  total: number;
  unit: string;
}

export type FilterFamily = 'F1' | 'F2' | 'F3' | 'F4';

export interface MlApi {
  analyze(kind: MediaKind, storageKey: string, opts?: AnalyzeOptions): Promise<MlResult>;
  modelInfo(): Promise<Record<string, unknown>>;
  compare(
    storageKey: string,
    family: FilterFamily,
    strength: number,
  ): Promise<Record<string, unknown>>;
  progress(progressId: string): Promise<MlProgress | null>;
}

export class MlClient implements MlApi {
  private readonly fetchImpl: FetchLike;

  constructor(private readonly opts: MlOptions) {
    this.fetchImpl = opts.fetchImpl ?? ((url, init) => fetch(url, init));
  }

  private async call(
    method: 'GET' | 'POST',
    path: string,
    body: unknown,
    timeoutMs: number,
  ): Promise<unknown> {
    let res: Response;
    try {
      res = await this.fetchImpl(`${this.opts.baseUrl}${path}`, {
        method,
        headers: {
          ...(body === undefined ? {} : { 'content-type': 'application/json' }),
          'x-internal-secret': this.opts.secret,
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs),
      });
    } catch (err) {
      const timedOut = err instanceof Error && err.name === 'TimeoutError';
      throw new MlServiceError(
        503,
        timedOut ? `ML service did not answer within ${timeoutMs} ms` : 'ML service unreachable',
      );
    }
    const text = await res.text();
    if (!res.ok) {
      let detail = text.slice(0, 500);
      try {
        const parsed = JSON.parse(text) as { detail?: unknown };
        if (typeof parsed.detail === 'string') detail = parsed.detail;
      } catch {
        // not JSON: keep the (truncated) raw text
      }
      throw new MlServiceError(res.status, `ML service ${res.status}: ${detail}`);
    }
    try {
      return JSON.parse(text) as unknown;
    } catch {
      throw new MlServiceError(502, 'ML service returned invalid JSON');
    }
  }

  analyze(kind: MediaKind, storageKey: string, opts: AnalyzeOptions = {}): Promise<MlResult> {
    const progress = opts.progressId ? { progressId: opts.progressId } : {};
    return (
      kind === 'image'
        ? this.call(
            'POST',
            '/analyze/image',
            { path: storageKey, k: this.opts.k, explain: true },
            this.opts.timeoutMs,
          )
        : this.call(
            'POST',
            '/analyze/video',
            { path: storageKey, k: this.opts.videoK, ...progress },
            this.opts.timeoutMs,
          )
    ) as Promise<MlResult>;
  }

  async modelInfo(): Promise<Record<string, unknown>> {
    return (await this.call(
      'GET',
      '/model-info',
      undefined,
      this.opts.metaTimeoutMs ?? 10_000,
    )) as Record<string, unknown>;
  }

  async compare(
    storageKey: string,
    family: FilterFamily,
    strength: number,
  ): Promise<Record<string, unknown>> {
    return (await this.call(
      'POST',
      '/compare',
      { path: storageKey, family, strength },
      this.opts.timeoutMs,
    )) as Record<string, unknown>;
  }

  async progress(progressId: string): Promise<MlProgress | null> {
    try {
      const p = (await this.call(
        'GET',
        `/progress/${encodeURIComponent(progressId)}`,
        undefined,
        this.opts.metaTimeoutMs ?? 5_000,
      )) as Partial<MlProgress> | null;
      if (!p || typeof p.done !== 'number' || typeof p.total !== 'number') return null;
      return {
        stage: String(p.stage ?? 'scoring'),
        done: p.done,
        total: p.total,
        unit: String(p.unit ?? 'frames'),
      };
    } catch {
      return null; // progress is best effort; the analysis itself reports real failures
    }
  }
}

/** The production client from config. */
export const configuredMlClient = (): MlClient =>
  new MlClient({
    baseUrl: config.ML_SERVICE_URL,
    secret: config.INTERNAL_SECRET,
    timeoutMs: config.ML_TIMEOUT_MS,
    k: config.K,
    videoK: config.VIDEO_K,
  });
