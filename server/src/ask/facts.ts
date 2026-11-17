/**
 * The only context /ask may use: the whitelisted result JSON (the same document Groq gets
 * for explanations) plus a validated summary of the video frames and the first face's
 * heatmap/filter cues. No image bytes, file names, paths, hashes or personal data.
 */
import { explanationInput, resultExtras, videoFacts, type VideoFacts } from '../analysisFacts.js';
import type { AnalysisRecord, ExternalCheckRecord } from '../data/types.js';
import { buildResultJson, type ResultJson } from '../explain/prompt.js';

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

export interface AskFacts {
  kind: 'image' | 'video';
  result: ResultJson;
  secondCheckRequested: boolean;
  secondCheckReason: string | null;
  faces: {
    count: number;
    heatmapIoU: number | null;
    beautifiedLikelihood: number | null;
    hasHeatmap: boolean;
  };
  video:
    | (Omit<VideoFacts, 'frames'> & {
        frames: Array<{
          index: number;
          timestamp: number | null;
          hasFace: boolean;
          logit: number | null;
        }>;
      })
    | null;
}

export function buildAskFacts(
  a: AnalysisRecord,
  external: ExternalCheckRecord | null,
  hiveThreshold: number,
): AskFacts | null {
  const input = explanationInput(a, external, hiveThreshold);
  if (!input) return null;
  const ml = (a.ml ?? {}) as { faces?: Array<Record<string, unknown>> };
  const face = ml.faces?.[0];
  const video = videoFacts(a);
  return {
    kind: a.kind,
    result: buildResultJson(input, resultExtras(a, external)),
    secondCheckRequested: a.externalCheckRequested,
    secondCheckReason: external?.reason ?? null,
    faces: {
      count: ml.faces?.length ?? 0,
      heatmapIoU: num(face?.heatmapIoU),
      beautifiedLikelihood: num(face?.beautifiedLikelihood),
      hasHeatmap: Boolean(face?.heatmaps),
    },
    video: video && {
      ...video,
      frames: video.frames.map(({ index, timestamp, hasFace, logit }) => ({
        index,
        timestamp,
        hasFace,
        logit,
      })),
    },
  };
}
