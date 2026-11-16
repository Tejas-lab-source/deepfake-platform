export type Verdict = 'FAKE' | 'REAL' | 'UNCERTAIN';
export type MediaKind = 'image' | 'video';

export type Badge =
  'verified_by_2' | 'second_check_disagrees' | 'checks_disagree' | 'hive_hint' | 'single_check';

/** Progress stages, in order. `failed` can replace any stage after `queued`. */
export const STAGES = [
  'queued',
  'detecting_faces',
  'scoring',
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
  frames?: Array<{ index: number; hasFace: boolean; logit: number | null }>;
  modelCardVersion: string;
  mock?: boolean;
  [key: string]: unknown;
}
