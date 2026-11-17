/**
 * What clients see of an analysis. Fields are listed explicitly: internal data (the media id,
 * file hash, storage key, raw provider responses, Groq fallback internals) never leaves here.
 */
import type { AnalysisRecord, ExplanationRecord, ExternalCheckRecord } from '../data/types.js';

export function presentListItem(a: AnalysisRecord) {
  return {
    id: a.id,
    kind: a.kind,
    status: a.status,
    stage: a.stage,
    final: a.final,
    ours: a.ours ? { probability: a.ours.probability } : null,
    createdAt: a.createdAt,
  };
}

export function presentExternal(external: ExternalCheckRecord | null) {
  return (
    external && {
      provider: external.provider,
      status: external.status,
      deepfakeScore: external.deepfakeScore,
      aiGeneratedScore: external.aiGeneratedScore,
      threshold: external.threshold,
      verdict: external.verdict,
      latencyMs: external.latencyMs,
      // Why there are no scores, if any (not_configured, disabled, budget:user_limit, ...).
      reason: external.reason ?? null,
      // Per-frame scores for videos; the scores above are their maxima.
      frames: external.frames ?? [],
    }
  );
}

export function presentAnalysis(
  a: AnalysisRecord,
  external: ExternalCheckRecord | null,
  explanation: ExplanationRecord | null,
) {
  return {
    id: a.id,
    kind: a.kind,
    status: a.status,
    stage: a.stage,
    progress: a.progress,
    error: a.error,
    createdAt: a.createdAt,
    updatedAt: a.updatedAt,
    completedAt: a.completedAt,
    externalCheckRequested: a.externalCheckRequested,
    final: a.final,
    ours: a.ours,
    external: presentExternal(external),
    explanation: explanation && {
      summary: explanation.summary,
      why: explanation.why,
      caveats: explanation.caveats,
      groqModel: explanation.groqModel,
      fallbackUsed: explanation.fallbackUsed,
    },
    ml: a.ml,
  };
}
