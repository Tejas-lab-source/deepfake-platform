import type { Badge, Verdict } from './types.js';

/** `opted_out` means the user switched the external check off for this upload. */
export type ExternalStatus = 'ok' | 'error' | 'timeout' | 'quota' | 'opted_out';

export interface ReconcileInput {
  ours: { verdict: Verdict; confident: boolean };
  external: { status: ExternalStatus; deepfakeScore: number | null };
  /** Hive says FAKE when deepfakeScore >= hiveThreshold. */
  hiveThreshold: number;
}

export interface ReconcileOutput {
  verdict: Verdict;
  badge: Badge;
  ruleApplied: string;
  externalVerdict: 'FAKE' | 'REAL' | null;
}

/**
 * Merge our verdict with the second opinion. Pure function.
 *
 * The second opinion never flips our verdict: it can confirm it, or downgrade a
 * non-confident verdict to UNCERTAIN.
 */
export function reconcile({ ours, external, hiveThreshold }: ReconcileInput): ReconcileOutput {
  if (external.status !== 'ok' || external.deepfakeScore === null) {
    const why = external.status === 'ok' ? 'error' : external.status;
    return {
      verdict: ours.verdict,
      badge: 'single_check',
      ruleApplied: `external_unavailable:${why}`,
      externalVerdict: null,
    };
  }
  const externalVerdict = external.deepfakeScore >= hiveThreshold ? 'FAKE' : 'REAL';

  if (ours.verdict === 'UNCERTAIN') {
    return {
      verdict: 'UNCERTAIN',
      badge: 'hive_hint',
      ruleApplied: 'ours_uncertain',
      externalVerdict,
    };
  }
  if (ours.verdict === externalVerdict) {
    return {
      verdict: ours.verdict,
      badge: 'verified_by_2',
      ruleApplied: `both_${ours.verdict.toLowerCase()}`,
      externalVerdict,
    };
  }
  if (ours.confident) {
    return {
      verdict: ours.verdict,
      badge: 'second_check_disagrees',
      ruleApplied: `ours_confident_${ours.verdict.toLowerCase()}_external_disagrees`,
      externalVerdict,
    };
  }
  return {
    verdict: 'UNCERTAIN',
    badge: 'checks_disagree',
    ruleApplied: 'ours_not_confident_external_disagrees',
    externalVerdict,
  };
}
