export type Verdict = 'FAKE' | 'REAL' | 'UNCERTAIN';
export type MediaKind = 'image' | 'video';

export type Badge =
  'verified_by_2' | 'second_check_disagrees' | 'checks_disagree' | 'hive_hint' | 'single_check';

/**
 * Progress stages, in order. `failed` can replace any stage after `queued`.
 * Phase 5 added `extracting_frames` (videos only) and `aggregating`; the earlier stage names
 * are unchanged, so existing clients still understand every event they knew before.
 *   queued            waiting for a worker
 *   detecting_faces   preprocessing: decoding the upload, finding faces
 *   extracting_frames video only: sampling frames
 *   scoring           running the detectors (per face, or per sampled frame)
 *   aggregating       combining copies/frames into one verdict, heatmaps
 *   verifying         the optional second check, then the reconciler
 *   explaining        writing the explanation (the report text)
 */
export const STAGES = [
  'queued',
  'detecting_faces',
  'extracting_frames',
  'scoring',
  'aggregating',
  'verifying',
  'explaining',
  'done',
] as const;
export type Stage = (typeof STAGES)[number] | 'failed';

/** What the ML service returns for an image or a video. Extra fields are kept as-is. */
export interface MlResult {
  verdict: Verdict;
  probability: number | null;
  threshold: number;
  confident: boolean;
  uncertainReason: string | null;
  faces?: Array<{
    scores?: { gendHead?: number; sbi?: number };
    spread?: number;
    [key: string]: unknown;
  }>;
  /** Video: one entry per sampled frame (Phase 5 adds timestamp, decoded, probability...). */
  frames?: Array<{
    index: number;
    hasFace: boolean;
    logit: number | null;
    timestamp?: number | null;
    [key: string]: unknown;
  }>;
  modelCardVersion: string;
  mock?: boolean;
  [key: string]: unknown;
}
