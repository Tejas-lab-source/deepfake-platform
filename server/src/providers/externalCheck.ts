/**
 * The second check, end to end: consent, cache, one-request-per-file lock, budget, provider.
 *
 *   no consent              -> nothing is sent (returns null)
 *   cached "ok" for sha256  -> reused; no request, no budget
 *   another request for the same file in flight -> wait for its result
 *   otherwise               -> reserve budget per attempt, call the provider, store the result
 *
 * Only "ok" results are reused. Every outcome is stored so it is visible on the analysis.
 */
import type { DailyBudget, DenyReason, InflightLock } from '../budget.js';
import { scrubDeep } from '../secrets.js';
import {
  verifyWithTimeout,
  type FrameScore,
  type Verifier,
  type VerifierResult,
} from './verifier.js';

export interface StoredCheck {
  _id: unknown;
  status: 'ok' | 'error' | 'timeout' | 'quota';
  deepfakeScore: number | null;
  aiGeneratedScore: number | null;
  verdict: 'FAKE' | 'REAL' | null;
  reason?: string | null;
}

export interface NewCheck {
  mediaSha256: string;
  provider: string;
  status: StoredCheck['status'];
  deepfakeScore: number | null;
  aiGeneratedScore: number | null;
  threshold: number;
  verdict: 'FAKE' | 'REAL' | null;
  latencyMs: number | null;
  rawResponse: unknown;
  reason: string | null;
  frames: FrameScore[];
}

export interface CheckStore {
  findCachedOk(sha256: string, provider: string): Promise<StoredCheck | null>;
  create(doc: NewCheck): Promise<StoredCheck>;
}

export interface ExternalCheckDeps {
  /** null when the second check is switched off (HIVE_MODE=off). */
  verifier: Verifier | null;
  /** The provider name used when the check is off, for the stored record. */
  provider: string;
  /** null for providers that cost nothing (the mock). */
  budget: DailyBudget | null;
  lock: InflightLock;
  store: CheckStore;
  threshold: number;
  /** Upper bound for one whole provider call, including any configured retries. */
  timeoutMs: number;
  pollMs?: number;
  now?: () => number;
  sleep?: (ms: number) => Promise<void>;
}

export interface ExternalCheckRequest {
  sha256: string;
  filePath: string;
  mime?: string;
  userId: string;
  /** Requests from the demo pool (admin role) may use the reserved budget. */
  demo: boolean;
  /** The uploader's explicit per-upload consent to send the file to the provider. */
  consent: boolean;
}

const defaultSleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

function verdictOf(result: VerifierResult, threshold: number): 'FAKE' | 'REAL' | null {
  if (result.status !== 'ok' || result.deepfakeScore === null) return null;
  return result.deepfakeScore >= threshold ? 'FAKE' : 'REAL';
}

export async function runExternalCheck(
  deps: ExternalCheckDeps,
  req: ExternalCheckRequest,
): Promise<StoredCheck | null> {
  // Without explicit consent nothing is sent, cached or recorded.
  if (req.consent !== true) return null;
  const provider = deps.verifier?.provider ?? deps.provider;
  const base = { mediaSha256: req.sha256, provider, threshold: deps.threshold };

  if (!deps.verifier) {
    return deps.store.create({
      ...base,
      status: 'error',
      deepfakeScore: null,
      aiGeneratedScore: null,
      verdict: null,
      latencyMs: null,
      rawResponse: null,
      reason: 'disabled',
      frames: [],
    });
  }

  const cached = await deps.store.findCachedOk(req.sha256, provider);
  if (cached) return cached;

  const sleep = deps.sleep ?? defaultSleep;
  const now = deps.now ?? Date.now;
  const lockKey = `${provider}:${req.sha256}`;
  const lockTtl = deps.timeoutMs + 5_000;
  let token = await deps.lock.acquire(lockKey, lockTtl);
  if (!token) {
    // Someone is already sending this file: wait for their result instead of paying twice.
    const deadline = now() + lockTtl;
    while (now() < deadline) {
      await sleep(deps.pollMs ?? 250);
      const peer = await deps.store.findCachedOk(req.sha256, provider);
      if (peer) return peer;
      if (!(await deps.lock.isHeld(lockKey))) {
        token = await deps.lock.acquire(lockKey, lockTtl);
        if (token) break;
      }
    }
    if (!token) {
      return deps.store.create({
        ...base,
        status: 'timeout',
        deepfakeScore: null,
        aiGeneratedScore: null,
        verdict: null,
        latencyMs: null,
        rawResponse: null,
        reason: 'peer_in_flight',
        frames: [],
      });
    }
  }

  try {
    // Re-check: the result may have arrived while we were acquiring the lock.
    const late = await deps.store.findCachedOk(req.sha256, provider);
    if (late) return late;

    let denied: DenyReason | null = null;
    const budget = deps.budget;
    const reserveAttempt = budget
      ? async () => {
          const r = await budget.reserve({ userId: req.userId, demo: req.demo });
          if (!r.allowed) denied = r.reason;
          return r.allowed;
        }
      : undefined;
    const started = now();
    const result = await verifyWithTimeout(deps.verifier, req.filePath, deps.timeoutMs, {
      mime: req.mime,
      reserveAttempt,
    });
    if (result.rateLimited && budget) await budget.markExhausted();
    const reason = denied ? `budget:${denied}` : (result.reason ?? null);
    return await deps.store.create({
      ...base,
      status: result.status,
      deepfakeScore: result.deepfakeScore,
      aiGeneratedScore: result.aiGeneratedScore,
      verdict: verdictOf(result, deps.threshold),
      latencyMs: now() - started,
      rawResponse: scrubDeep(result.raw),
      reason,
      frames: result.frames ?? [],
    });
  } finally {
    if (token) await deps.lock.release(lockKey, token).catch(() => false);
  }
}

/** Like runExternalCheck, but any failure (Redis, MongoDB, provider) becomes `null`, which
 * the pipeline reports as an unavailable second check. It never throws. */
export async function safeExternalCheck(
  deps: ExternalCheckDeps,
  req: ExternalCheckRequest,
  onError: (err: unknown) => void,
): Promise<StoredCheck | null> {
  try {
    return await runExternalCheck(deps, req);
  } catch (err) {
    onError(err);
    return null;
  }
}
