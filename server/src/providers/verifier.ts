export type VerifierStatus = 'ok' | 'error' | 'timeout' | 'quota';

export interface VerifierResult {
  status: VerifierStatus;
  deepfakeScore: number | null;
  aiGeneratedScore: number | null;
  raw: unknown;
}

/** A second-opinion provider. Hive is one implementation; others can be swapped in. */
export interface Verifier {
  readonly provider: string;
  verify(filePath: string): Promise<VerifierResult>;
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
