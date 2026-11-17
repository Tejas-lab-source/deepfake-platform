export type Verdict = 'FAKE' | 'REAL' | 'UNCERTAIN';
export type Badge =
  'verified_by_2' | 'second_check_disagrees' | 'checks_disagree' | 'hive_hint' | 'single_check';
export type Stage =
  | 'queued'
  | 'detecting_faces'
  | 'extracting_frames'
  | 'scoring'
  | 'aggregating'
  | 'verifying'
  | 'explaining'
  | 'done'
  | 'failed';
export type Status = 'queued' | 'processing' | 'done' | 'failed';

export interface Progress {
  done: number;
  total: number;
  unit: string;
}

export interface Face {
  box: [number, number, number, number];
  scores: { gendHead: number; sbi: number | null };
  variantLogits: Record<string, number>;
  spread: number;
  beautifiedLikelihood: number | null;
  probability?: number;
  verdict?: Verdict;
  crops?: { gend: string; sbi: string } | null;
  heatmaps: { gend: string; sbi: string | null } | null;
  heatmapIoU: number | null;
}

export interface Frame {
  index: number;
  timestamp: number | null;
  hasFace: boolean;
  logit: number | null;
  decoded?: boolean;
  probability?: number | null;
  verdict?: Verdict | null;
  explained?: boolean;
  faces?: Face[];
}

export interface FrameRef {
  index: number;
  timestamp: number | null;
  logit: number;
}

export interface MlResult {
  kind?: 'video';
  faces?: Face[];
  frames?: Frame[];
  framesSampled?: number;
  framesScored?: number;
  video?: { fps: number; frameCount: number; durationSec: number; sampledFrames: number } | null;
  aggregate?: { method: string; k: number; logit: number } | null;
  mostSuspicious?: FrameRef | null;
  leastSuspicious?: FrameRef | null;
  mock?: boolean;
  modelCardVersion?: string;
}

export interface Analysis {
  id: string;
  kind: 'image' | 'video';
  status: Status;
  stage: Stage;
  progress?: Progress | null;
  error: string | null;
  createdAt: string;
  completedAt?: string | null;
  externalCheckRequested: boolean;
  final: { verdict: Verdict; badge: Badge; ruleApplied: string } | null;
  ours: {
    verdict: Verdict;
    probability: number | null;
    threshold: number;
    confident: boolean;
    uncertainReason: string | null;
    spread: number | null;
  } | null;
  external: {
    provider: string;
    status: 'ok' | 'error' | 'timeout' | 'quota';
    deepfakeScore: number | null;
    aiGeneratedScore: number | null;
    threshold: number;
    verdict: 'FAKE' | 'REAL' | null;
    latencyMs: number | null;
    reason?: string | null;
  } | null;
  explanation: {
    summary: string;
    why: string;
    caveats: string[];
    groqModel: string | null;
    fallbackUsed: boolean;
  } | null;
  ml: MlResult | null;
}

export interface AnalysisListItem {
  id: string;
  kind: 'image' | 'video';
  status: Status;
  stage: Stage;
  final: { verdict: Verdict; badge: Badge } | null;
  ours: { probability: number | null } | null;
  createdAt: string;
}

export interface Page<T> {
  page: number;
  limit: number;
  total: number;
  items: T[];
}

export interface ModelCard {
  mockMode: boolean;
  detectors: {
    primary: string;
    secondary: string;
    sbiLoaded: boolean | null;
    headLoaded: boolean | null;
    device: string | null;
  };
  pipeline: {
    beautifiedCopiesPerImage: number | null;
    beautifiedCopiesPerVideoFrame: number;
    maxVideoFrames: number | null;
    videoTopK: number | null;
  };
  calibration: {
    version: string | null;
    status: string | null;
    fixture: boolean;
    temperature: number | null;
    threshold: number | null;
    uncertainMargin: number | null;
    confidentMargin: number | null;
    spreadLimit: number | null;
    detectorDisagreeLimit: number | null;
    sources: Record<string, unknown>;
  };
  secondCheck: { provider: string; mode: string; threshold: number; rule: string };
  explanations: { provider: string; mode: string; rule: string };
  licences: Array<{ component: string; licence: string }>;
  notice: string;
}

export interface EvalRun {
  scheme: string;
  fold: string;
  seed: number;
  heldOut: boolean;
  metrics: Record<string, number | null>;
  ci95: Record<string, [number, number] | null>;
}

export interface EvalReport {
  id: string;
  created: string | null;
  fixture: boolean;
  notice: string | null;
  aggregate: Record<string, Record<string, unknown>>;
  successCriteria: Array<Record<string, unknown>>;
  runs: EvalRun[];
}

export interface ComparisonSide {
  verdict: Verdict | null;
  probability: number | null;
  confident: boolean | null;
  uncertainReason: string | null;
  scores: { gendHead: number; sbi: number | null } | null;
  image: string | null;
  heatmap: string | null;
}

export interface Comparison {
  id: string;
  family: 'F1' | 'F2' | 'F3' | 'F4';
  strength: number;
  threshold: number | null;
  clean: ComparisonSide;
  filtered: ComparisonSide;
  flipped: boolean;
  heatmapIoU: number | null;
  createdAt: string;
}

export interface AskResponse {
  answer: string;
  source: 'template' | 'groq';
  intent: string;
  fallbackReason: string | null;
}

export interface Session {
  user: { id: string; email: string; role: 'user' | 'admin' };
  accessToken: string;
  refreshToken: string;
}
