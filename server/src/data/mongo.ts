/** MongoDB (Mongoose) implementation of the repositories: the production data layer. */
import mongoose from 'mongoose';
import {
  Analysis,
  Comparison,
  EvalReport,
  Explanation,
  ExternalCheck,
  Media,
  User,
} from '../models/index.js';
import type { Stage } from '../types.js';
import type {
  AnalysisRecord,
  AnalysisStatus,
  ComparisonRecord,
  EvalReportRecord,
  ExplanationRecord,
  ExplanationWrite,
  ExternalCheckRecord,
  MediaRecord,
  Repos,
  Role,
  UserRecord,
} from './types.js';

type Doc = Record<string, unknown> & { _id: unknown };

const str = (v: unknown): string => String(v);
const strOrNull = (v: unknown): string | null => (v == null ? null : String(v));
const isId = (id: string): boolean => mongoose.isValidObjectId(id);

function toUser(d: Doc): UserRecord {
  return {
    id: str(d._id),
    email: d.email as string,
    passwordHash: d.passwordHash as string,
    role: (d.role as Role) ?? 'user',
    createdAt: d.createdAt as Date,
  };
}

function toMedia(d: Doc): MediaRecord {
  return {
    id: str(d._id),
    ownerId: str(d.ownerId),
    kind: d.kind as MediaRecord['kind'],
    storageKey: d.storageKey as string,
    mime: d.mime as string,
    sha256: d.sha256 as string,
    sizeBytes: d.sizeBytes as number,
    createdAt: d.createdAt as Date,
    expiresAt: d.expiresAt as Date,
  };
}

export function toAnalysis(d: Doc): AnalysisRecord {
  return {
    id: str(d._id),
    ownerId: str(d.ownerId),
    mediaId: str(d.mediaId),
    mediaSha256: (d.mediaSha256 as string | null | undefined) ?? null,
    kind: d.kind as AnalysisRecord['kind'],
    status: d.status as AnalysisStatus,
    stage: d.stage as Stage,
    progress: (d.progress as AnalysisRecord['progress']) ?? null,
    error: (d.error as string | null | undefined) ?? null,
    attempts: (d.attempts as number | undefined) ?? 0,
    externalCheckRequested: Boolean(d.externalCheckRequested),
    final: (d.final as AnalysisRecord['final']) ?? null,
    ours: (d.ours as AnalysisRecord['ours']) ?? null,
    externalCheckId: strOrNull(d.externalCheckId),
    faces: (d.faces as unknown[]) ?? [],
    frames: (d.frames as unknown[]) ?? [],
    heatmapKeys: (d.heatmapKeys as string[]) ?? [],
    modelCardVersion: (d.modelCardVersion as string | null | undefined) ?? null,
    ml: (d.ml as AnalysisRecord['ml']) ?? null,
    completedAt: (d.completedAt as Date | null | undefined) ?? null,
    createdAt: d.createdAt as Date,
    updatedAt: d.updatedAt as Date,
  };
}

function toExternal(d: Doc): ExternalCheckRecord {
  return {
    id: str(d._id),
    mediaSha256: d.mediaSha256 as string,
    provider: d.provider as string,
    status: d.status as ExternalCheckRecord['status'],
    deepfakeScore: (d.deepfakeScore as number | null) ?? null,
    aiGeneratedScore: (d.aiGeneratedScore as number | null) ?? null,
    threshold: d.threshold as number,
    verdict: (d.verdict as ExternalCheckRecord['verdict']) ?? null,
    latencyMs: (d.latencyMs as number | null) ?? null,
    reason: (d.reason as string | null | undefined) ?? null,
    frames: (d.frames as ExternalCheckRecord['frames']) ?? [],
  };
}

function toExplanation(d: Doc): ExplanationRecord {
  return {
    analysisId: str(d.analysisId),
    groqModel: (d.groqModel as string | null) ?? null,
    promptVersion: d.promptVersion as string,
    summary: d.summary as string,
    why: d.why as string,
    caveats: (d.caveats as string[]) ?? [],
    fallbackUsed: Boolean(d.fallbackUsed),
    fallbackReason: (d.fallbackReason as string | null | undefined) ?? null,
  };
}

function toComparison(d: Doc): ComparisonRecord {
  return {
    id: str(d._id),
    analysisId: str(d.analysisId),
    ownerId: str(d.ownerId),
    family: d.family as string,
    strength: d.strength as number,
    threshold: (d.threshold as number | null) ?? null,
    clean: d.clean as Record<string, unknown>,
    filtered: d.filtered as Record<string, unknown>,
    flipped: Boolean(d.flipped),
    heatmapIoU: (d.heatmapIoU as number | null) ?? null,
    heatmapKeys: (d.heatmapKeys as string[]) ?? [],
    modelCardVersion: (d.modelCardVersion as string | null) ?? null,
    createdAt: d.createdAt as Date,
  };
}

function toEvalReport(d: Doc): EvalReportRecord {
  return {
    id: str(d._id),
    created: (d.created as string | null) ?? null,
    importedAt: d.importedAt as Date,
    fixture: Boolean(d.fixture),
    notice: (d.notice as string | null) ?? null,
    protocol: d.protocol ?? null,
    inputs: (d.inputs as Record<string, unknown>) ?? {},
    aggregate: (d.aggregate as Record<string, unknown>) ?? {},
    ablation: d.ablation ?? null,
    successCriteria: (d.successCriteria as unknown[]) ?? [],
    heatmapIoU: d.heatmapIoU ?? null,
    codeCommit: (d.codeCommit as string | null) ?? null,
    runs: (d.runs as EvalReportRecord['runs']) ?? [],
  };
}

const lean = <T>(q: { lean: () => Promise<unknown> }): Promise<T> => q.lean() as Promise<T>;

export function mongoRepos(): Repos {
  return {
    users: {
      async findByEmail(email) {
        const d = await lean<Doc | null>(User.findOne({ email }));
        return d && toUser(d);
      },
      async findById(id) {
        if (!isId(id)) return null;
        const d = await lean<Doc | null>(User.findById(id));
        return d && toUser(d);
      },
      async create(doc) {
        const d = await User.create(doc);
        return toUser(d.toObject() as Doc);
      },
    },
    media: {
      async create(doc) {
        const d = await Media.create(doc);
        return toMedia(d.toObject() as Doc);
      },
      async findById(id) {
        if (!isId(id)) return null;
        const d = await lean<Doc | null>(Media.findById(id));
        return d && toMedia(d);
      },
      async delete(id) {
        if (!isId(id)) return null;
        const d = await lean<Doc | null>(Media.findByIdAndDelete(id));
        return d && toMedia(d);
      },
    },
    analyses: {
      async create(doc) {
        const d = await Analysis.create(doc);
        return toAnalysis(d.toObject() as Doc);
      },
      async findById(id) {
        if (!isId(id)) return null;
        const d = await lean<Doc | null>(Analysis.findById(id));
        return d && toAnalysis(d);
      },
      async findOwned(id, ownerId) {
        if (!isId(id) || !isId(ownerId)) return null;
        const d = await lean<Doc | null>(Analysis.findOne({ _id: id, ownerId }));
        return d && toAnalysis(d);
      },
      async listOwned(ownerId, page, limit) {
        const filter = { ownerId };
        const [items, total] = await Promise.all([
          lean<Doc[]>(
            Analysis.find(filter)
              .sort({ createdAt: -1 })
              .skip((page - 1) * limit)
              .limit(limit)
              // The list never needs the full ML result.
              .select('-ml -faces -frames'),
          ),
          Analysis.countDocuments(filter),
        ]);
        return { items: items.map(toAnalysis), total };
      },
      async findActiveDuplicate(ownerId, sha256) {
        const d = await lean<Doc | null>(
          Analysis.findOne({
            ownerId,
            mediaSha256: sha256,
            status: { $in: ['queued', 'processing'] },
          }).sort({ createdAt: -1 }),
        );
        return d && toAnalysis(d);
      },
      async findStale(before) {
        const docs = await lean<Doc[]>(
          Analysis.find({
            status: { $in: ['queued', 'processing'] },
            updatedAt: { $lt: before },
          })
            .select('-ml -faces -frames')
            .limit(100),
        );
        return docs.map(toAnalysis);
      },
      countByExternalCheck(externalCheckId) {
        if (!isId(externalCheckId)) return Promise.resolve(0);
        return Analysis.countDocuments({ externalCheckId });
      },
      async update(id, patch) {
        await Analysis.updateOne({ _id: id }, patch);
      },
      async delete(id) {
        await Analysis.deleteOne({ _id: id });
      },
    },
    explanations: {
      async findByAnalysis(analysisId) {
        if (!isId(analysisId)) return null;
        const d = await lean<Doc | null>(Explanation.findOne({ analysisId }));
        return d && toExplanation(d);
      },
      async deleteByAnalysis(analysisId) {
        await Explanation.deleteOne({ analysisId });
      },
      async upsert(analysisId, doc) {
        await Explanation.findOneAndUpdate({ analysisId }, doc, { upsert: true });
      },
      async findCachedGroq(inputSha256, promptVersion) {
        return lean<ExplanationWrite | null>(
          Explanation.findOne({ inputSha256, fallbackUsed: false, promptVersion })
            .sort({ createdAt: -1 })
            .select('-_id -analysisId -createdAt -updatedAt -__v'),
        );
      },
    },
    externalChecks: {
      async findById(id) {
        if (!isId(id)) return null;
        const d = await lean<Doc | null>(ExternalCheck.findById(id));
        return d && toExternal(d);
      },
      async delete(id) {
        await ExternalCheck.deleteOne({ _id: id });
      },
    },
    comparisons: {
      async create(doc) {
        const d = await Comparison.create(doc);
        return toComparison(d.toObject() as Doc);
      },
      async listByAnalysis(analysisId) {
        const docs = await lean<Doc[]>(Comparison.find({ analysisId }).sort({ createdAt: -1 }));
        return docs.map(toComparison);
      },
      async deleteByAnalysis(analysisId) {
        await Comparison.deleteMany({ analysisId });
      },
    },
    evalReports: {
      async latest() {
        const d = await lean<Doc | null>(EvalReport.findOne().sort({ importedAt: -1 }));
        return d && toEvalReport(d);
      },
      async create(doc) {
        const d = await EvalReport.create(doc);
        return toEvalReport(d.toObject() as Doc);
      },
    },
  };
}
