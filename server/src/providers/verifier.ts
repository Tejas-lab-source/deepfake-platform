export type VerifierStatus = 'ok' | 'error' | 'timeout' | 'quota';

/** Score of one video frame (images have a single frame at timestamp 0). */
export interface FrameScore {
  frameIndex: number;
  timestamp: number | null;
  deepfake: number;
  aiGenerated: number;
}

export interface VerifierResult {
  status: VerifierStatus;
  deepfakeScore: number | null;
  aiGeneratedScore: number | null;
  raw: unknown;
  /** Why the check did not produce scores (for example "not_configured", "auth"). */
  reason?: string;
  /** Per-frame scores; deepfakeScore and aiGeneratedScore are their maxima. */
  frames?: FrameScore[];
  /** The provider itself refused for rate limiting (HTTP 429). */
  rateLimited?: boolean;
}

export interface VerifyOptions {
  /** MIME type of the file, from the upload's magic-byte check. */
  mime?: string;
  /** Called before every billable attempt; returning false stops with status "quota". */
  reserveAttempt?: () => Promise<boolean>;
}

/** A second-opinion provider. Hive is one implementation; others can be swapped in. */
export interface Verifier {
  readonly provider: string;
  verify(filePath: string, options?: VerifyOptions): Promise<VerifierResult>;
}

/** Resolve with status `timeout` if the provider takes longer than `ms`. */
export async function verifyWithTimeout(
  verifier: Verifier,
  filePath: string,
  ms: number,
): Promise<VerifierResult> {
  let timer: NodeJS.Timeout | undefined;
  const timeout = new Promise<VerifierResult>((resolve) => {
    timer = setTimeout(
      () => resolve({ status: 'timeout', deepfakeScore: null, aiGeneratedScore: null, raw: null }),
      ms,
    );
  });
  try {
    return await Promise.race([
      verifier.verify(filePath).catch((err: unknown): VerifierResult => ({
        status: 'error',
        deepfakeScore: null,
        aiGeneratedScore: null,
        raw: { error: err instanceof Error ? err.message : String(err) },
      })),
      timeout,
    ]);
  } finally {
    clearTimeout(timer);
  }
}
