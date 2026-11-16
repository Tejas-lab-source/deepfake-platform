export type Verdict = 'FAKE' | 'REAL' | 'UNCERTAIN';
export type Badge =
  'verified_by_2' | 'second_check_disagrees' | 'checks_disagree' | 'hive_hint' | 'single_check';
export type Stage =
  'queued' | 'detecting_faces' | 'scoring' | 'verifying' | 'explaining' | 'done' | 'failed';

export interface Face {
  box: [number, number, number, number];
  scores: { gendHead: number; sbi: number };
  variantLogits: Record<string, number>;
  spread: number;
  beautifiedLikelihood: number;
  heatmaps: { gend: string; sbi: string } | null;
  heatmapIoU: number | null;
}

export interface Analysis {
  id: string;
  kind: 'image' | 'video';
  status: 'queued' | 'processing' | 'done' | 'failed';
  stage: Stage;
  error: string | null;
  createdAt: string;
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
  } | null;
  explanation: {
    summary: string;
    why: string;
    caveats: string[];
    groqModel: string | null;
    fallbackUsed: boolean;
  } | null;
  ml: {
    faces?: Face[];
    frames?: Array<{ index: number; hasFace: boolean; logit: number | null }>;
    mock?: boolean;
    modelCardVersion?: string;
  } | null;
}

export interface Session {
  user: { id: string; email: string; role: 'user' | 'admin' };
  accessToken: string;
  refreshToken: string;
}
