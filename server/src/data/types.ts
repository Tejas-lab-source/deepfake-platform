/**
 * Records and repository interfaces. Routes and workers depend on these interfaces, never on
 * Mongoose directly: production wires the MongoDB implementation (data/mongo.ts), and the
 * HTTP route tests wire an in-memory fake (tests/helpers/memoryRepos.ts). The fake is test
 * code only and is never part of the runtime.
 */
import type { MediaKind, Stage } from '../types.js';

export type Role = 'user' | 'admin';
export type AnalysisStatus = 'queued' | 'processing' | 'done' | 'failed';

export interface UserRecord {
  id: string;
  email: string;
  passwordHash: string;
  role: Role;
  createdAt: Date;
}

export interface MediaRecord {
  id: string;
  ownerId: string;
  kind: MediaKind;
  storageKey: string;
  mime: string;
  sha256: string;
  sizeBytes: number;
  createdAt: Date;
  expiresAt: Date;
}

/** Measured progress inside a stage, e.g. frames scored so far out of the frames sampled. */
export interface Progress {
  done: number;
  total: number;
  unit: string;
}

export interface FinalVerdict {
  verdict: string;
  badge: string;
  ruleApplied: string;
}

export interface OursSummary {
  verdict: string;
  probability: number | null;
  threshold: number;
  confident: boolean;
  uncertainReason: string | null;
  detectorScores: unknown;
  spread: number | null;
}

export interface AnalysisRecord {
  id: string;
  ownerId: string;
  mediaId: string;
  mediaSha256: string | null;
  kind: MediaKind;
  status: AnalysisStatus;
  stage: Stage;
  progress: Progress | null;
  error: string | null;
  attempts: number;
  externalCheckRequested: boolean;
  final: FinalVerdict | null;
  ours: OursSummary | null;
  externalCheckId: string | null;
  faces: unknown[];
  frames: unknown[];
  heatmapKeys: string[];
  modelCardVersion: string | null;
  ml: Record<string, unknown> | null;
  completedAt: Date | null;
  createdAt: Date;
  updatedAt: Date;
}

export type AnalysisPatch = Partial<
  Omit<AnalysisRecord, 'id' | 'ownerId' | 'mediaId' | 'kind' | 'createdAt' | 'updatedAt'>
>;

export interface ExternalCheckRecord {
  id: string;
  mediaSha256: string;
  provider: string;
  status: 'ok' | 'error' | 'timeout' | 'quota';
  deepfakeScore: number | null;
  aiGeneratedScore: number | null;
  threshold: number;
  verdict: 'FAKE' | 'REAL' | null;
  latencyMs: number | null;
  reason: string | null;
  frames: Array<{ frameIndex: number; timestamp: number; deepfake: number; aiGenerated: number }>;
}

export interface ExplanationRecord {
  analysisId: string;
  groqModel: string | null;
  promptVersion: string;
  summary: string;
  why: string;
  caveats: string[];
  fallbackUsed: boolean;
  fallbackReason: string | null;
}

/** What the explainer stores (Phase 4 shape). */
export interface ExplanationWrite {
  groqModel: string | null;
  promptVersion: string;
  summary: string;
  why: string;
  caveats: string[];
  fallbackUsed: boolean;
  fallbackReason: string | null;
  inputSha256: string | null;
  latencyMs: number | null;
  usage: unknown;
}

export interface ComparisonRecord {
  id: string;
  analysisId: string;
  ownerId: string;
  family: string;
  strength: number;
  threshold: number | null;
  clean: Record<string, unknown>;
  filtered: Record<string, unknown>;
  flipped: boolean;
  heatmapIoU: number | null;
  heatmapKeys: string[];
  modelCardVersion: string | null;
  createdAt: Date;
}

export type NewComparison = Omit<ComparisonRecord, 'id' | 'createdAt'>;

export interface EvalRunRecord {
  scheme: string;
  fold: string;
  seed: number;
  heldOut: boolean;
  metrics: Record<string, number | null>;
  ci95: Record<string, [number, number] | null>;
}

export interface EvalReportRecord {
  id: string;
  created: string | null;
  importedAt: Date;
  fixture: boolean;
  notice: string | null;
  protocol: unknown;
  inputs: Record<string, unknown>;
  aggregate: Record<string, unknown>;
  ablation: unknown;
  successCriteria: unknown[];
  heatmapIoU: unknown;
  codeCommit: string | null;
  runs: EvalRunRecord[];
}

export type NewEvalReport = Omit<EvalReportRecord, 'id' | 'importedAt'>;

export interface Page<T> {
  items: T[];
  total: number;
}

export interface UserRepo {
  findByEmail(email: string): Promise<UserRecord | null>;
  findById(id: string): Promise<UserRecord | null>;
  create(doc: { email: string; passwordHash: string; role?: Role }): Promise<UserRecord>;
}

export interface MediaRepo {
  create(doc: Omit<MediaRecord, 'id' | 'createdAt'>): Promise<MediaRecord>;
  findById(id: string): Promise<MediaRecord | null>;
  delete(id: string): Promise<MediaRecord | null>;
}

export interface AnalysisRepo {
  create(doc: {
    ownerId: string;
    mediaId: string;
    mediaSha256: string;
    kind: MediaKind;
    externalCheckRequested: boolean;
  }): Promise<AnalysisRecord>;
  findById(id: string): Promise<AnalysisRecord | null>;
  findOwned(id: string, ownerId: string): Promise<AnalysisRecord | null>;
  listOwned(ownerId: string, page: number, limit: number): Promise<Page<AnalysisRecord>>;
  /** An unfinished analysis of the same file by the same owner (duplicate upload). */
  findActiveDuplicate(ownerId: string, sha256: string): Promise<AnalysisRecord | null>;
  /** Analyses still queued or processing whose last update is older than `before`. */
  findStale(before: Date): Promise<AnalysisRecord[]>;
  countByExternalCheck(externalCheckId: string): Promise<number>;
  update(id: string, patch: AnalysisPatch): Promise<void>;
  delete(id: string): Promise<void>;
}

export interface ExplanationRepo {
  findByAnalysis(analysisId: string): Promise<ExplanationRecord | null>;
  upsert(analysisId: string, doc: ExplanationWrite): Promise<void>;
  /** A previous Groq (not template) explanation for exactly the same facts and prompt. */
  findCachedGroq(inputSha256: string, promptVersion: string): Promise<ExplanationWrite | null>;
  deleteByAnalysis(analysisId: string): Promise<void>;
}

export interface ExternalCheckRepo {
  findById(id: string): Promise<ExternalCheckRecord | null>;
  delete(id: string): Promise<void>;
}

export interface ComparisonRepo {
  create(doc: NewComparison): Promise<ComparisonRecord>;
  listByAnalysis(analysisId: string): Promise<ComparisonRecord[]>;
  deleteByAnalysis(analysisId: string): Promise<void>;
}

export interface EvalReportRepo {
  latest(): Promise<EvalReportRecord | null>;
  create(doc: NewEvalReport): Promise<EvalReportRecord>;
}

export interface Repos {
  users: UserRepo;
  media: MediaRepo;
  analyses: AnalysisRepo;
  explanations: ExplanationRepo;
  externalChecks: ExternalCheckRepo;
  comparisons: ComparisonRepo;
  evalReports: EvalReportRepo;
}
