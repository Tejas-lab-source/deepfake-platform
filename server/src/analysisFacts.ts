/**
 * The approved facts about one analysis, shared by the explanation job, the Ask endpoint and
 * the PDF report. Everything here is a verdict, a score, a threshold or a count: never an
 * image, file name, path, hash, storage key, email or raw provider response.
 */
import type { AnalysisRecord, ExternalCheckRecord } from './data/types.js';
import type { ResultExtras } from './explain/prompt.js';
import type { ExplanationInput } from './explainTemplate.js';

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

export function explanationInput(
  a: AnalysisRecord,
  external: ExternalCheckRecord | null,
  hiveThreshold: number,
): ExplanationInput | null {
  if (!a.final || !a.ours) return null;
  const ml = a.ml as { mock?: boolean } | null;
  return {
    final: a.final as ExplanationInput['final'],
    ours: {
      verdict: a.ours.verdict as ExplanationInput['ours']['verdict'],
      probability: a.ours.probability ?? null,
      threshold: a.ours.threshold ?? 0.5,
      confident: a.ours.confident ?? false,
      uncertainReason: a.ours.uncertainReason ?? null,
      spread: a.ours.spread ?? null,
    },
    external: external
      ? {
          status: external.status,
          deepfakeScore: external.deepfakeScore ?? null,
          threshold: hiveThreshold,
          verdict:
            external.status === 'ok' && typeof external.deepfakeScore === 'number'
              ? external.deepfakeScore >= hiveThreshold
                ? 'FAKE'
                : 'REAL'
              : null,
        }
      : null,
    mock: Boolean(ml?.mock),
  };
}

export function resultExtras(
  a: AnalysisRecord,
  external: ExternalCheckRecord | null,
): ResultExtras {
  const ml = a.ml as { faces?: Array<{ scores?: { gendHead?: number; sbi?: number } }> } | null;
  return {
    ruleApplied: a.final?.ruleApplied ?? null,
    detectorScores: ml?.faces?.[0]?.scores ?? null,
    aiGeneratedScore: external?.aiGeneratedScore ?? null,
    provider: external?.provider ?? null,
    modelCardVersion: a.modelCardVersion ?? null,
  };
}

export interface FrameFact {
  index: number;
  timestamp: number | null;
  hasFace: boolean;
  decoded: boolean;
  logit: number | null;
  probability: number | null;
  verdict: string | null;
  explained: boolean;
}

export interface VideoFacts {
  framesSampled: number;
  framesScored: number;
  durationSec: number | null;
  fps: number | null;
  topK: number | null;
  aggregateLogit: number | null;
  mostSuspicious: { index: number; timestamp: number | null; logit: number } | null;
  leastSuspicious: { index: number; timestamp: number | null; logit: number } | null;
  frames: FrameFact[];
}

type Ref = { index?: unknown; timestamp?: unknown; logit?: unknown } | null | undefined;
const ref = (r: Ref): VideoFacts['mostSuspicious'] =>
  r && typeof r.index === 'number' && num(r.logit) !== null
    ? { index: r.index, timestamp: num(r.timestamp), logit: num(r.logit)! }
    : null;

/** The video part of an ML result, validated field by field (null for images). */
export function videoFacts(a: AnalysisRecord): VideoFacts | null {
  if (a.kind !== 'video' || !a.ml) return null;
  const ml = a.ml as Record<string, unknown>;
  const rawFrames = Array.isArray(ml.frames) ? (ml.frames as Array<Record<string, unknown>>) : [];
  const frames: FrameFact[] = rawFrames
    .filter((f) => typeof f.index === 'number')
    .map((f) => ({
      index: f.index as number,
      timestamp: num(f.timestamp),
      hasFace: f.hasFace === true,
      decoded: f.decoded !== false,
      logit: num(f.logit),
      probability: num(f.probability),
      verdict: typeof f.verdict === 'string' ? f.verdict : null,
      explained: f.explained === true,
    }));
  const video = (ml.video ?? {}) as Record<string, unknown>;
  const agg = (ml.aggregate ?? {}) as Record<string, unknown>;
  return {
    framesSampled: num(ml.framesSampled) ?? frames.length,
    framesScored: num(ml.framesScored) ?? frames.filter((f) => f.logit !== null).length,
    durationSec: num(video.durationSec),
    fps: num(video.fps),
    topK: num(agg.k),
    aggregateLogit: num(agg.logit),
    mostSuspicious: ref(ml.mostSuspicious as Ref),
    leastSuspicious: ref(ml.leastSuspicious as Ref),
    frames,
  };
}
