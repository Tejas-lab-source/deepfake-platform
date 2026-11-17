/**
 * In-memory repositories for HTTP route tests. TEST CODE ONLY: production always uses
 * src/data/mongo.ts. Behaviour mirrors the Mongo implementation closely enough for routes
 * (ids are 24-hex like ObjectIds; lists sort newest first), but these tests are not MongoDB
 * integration tests.
 */
import { randomBytes } from 'node:crypto';
import type {
  AnalysisRecord,
  ComparisonRecord,
  EvalReportRecord,
  ExplanationRecord,
  ExplanationWrite,
  ExternalCheckRecord,
  MediaRecord,
  Repos,
  UserRecord,
} from '../../src/data/types.js';

export const newId = (): string => randomBytes(12).toString('hex');

export interface MemoryRepos extends Repos {
  store: {
    users: Map<string, UserRecord>;
    media: Map<string, MediaRecord>;
    analyses: Map<string, AnalysisRecord>;
    explanations: Map<string, ExplanationRecord & Partial<ExplanationWrite>>;
    externalChecks: Map<string, ExternalCheckRecord>;
    comparisons: Map<string, ComparisonRecord>;
    evalReports: EvalReportRecord[];
  };
}

const clone = <T>(v: T): T => structuredClone(v);

export function memoryRepos(): MemoryRepos {
  const store: MemoryRepos['store'] = {
    users: new Map(),
    media: new Map(),
    analyses: new Map(),
    explanations: new Map(),
    externalChecks: new Map(),
    comparisons: new Map(),
    evalReports: [],
  };
  let last = 0;
  // Real time, strictly increasing, so "newest first" and staleness are well defined.
  const now = (): Date => {
    last = Math.max(Date.now(), last + 1);
    return new Date(last);
  };

  return {
    store,
    users: {
      async findByEmail(email) {
        return clone([...store.users.values()].find((u) => u.email === email) ?? null);
      },
      async findById(id) {
        return clone(store.users.get(id) ?? null);
      },
      async create(doc) {
        const u: UserRecord = {
          id: newId(),
          email: doc.email,
          passwordHash: doc.passwordHash,
          role: doc.role ?? 'user',
          createdAt: now(),
        };
        store.users.set(u.id, u);
        return clone(u);
      },
    },
    media: {
      async create(doc) {
        const m: MediaRecord = { ...doc, id: newId(), createdAt: now() };
        store.media.set(m.id, m);
        return clone(m);
      },
      async findById(id) {
        return clone(store.media.get(id) ?? null);
      },
      async delete(id) {
        const m = store.media.get(id) ?? null;
        store.media.delete(id);
        return m;
      },
    },
    analyses: {
      async create(doc) {
        const t = now();
        const a: AnalysisRecord = {
          id: newId(),
          ownerId: doc.ownerId,
          mediaId: doc.mediaId,
          mediaSha256: doc.mediaSha256,
          kind: doc.kind,
          status: 'queued',
          stage: 'queued',
          progress: null,
          error: null,
          attempts: 0,
          externalCheckRequested: doc.externalCheckRequested,
          final: null,
          ours: null,
          externalCheckId: null,
          faces: [],
          frames: [],
          heatmapKeys: [],
          modelCardVersion: null,
          ml: null,
          completedAt: null,
          createdAt: t,
          updatedAt: t,
        };
        store.analyses.set(a.id, a);
        return clone(a);
      },
      async findById(id) {
        return clone(store.analyses.get(id) ?? null);
      },
      async findOwned(id, ownerId) {
        const a = store.analyses.get(id);
        return clone(a && a.ownerId === ownerId ? a : null);
      },
      async listOwned(ownerId, page, limit) {
        const all = [...store.analyses.values()]
          .filter((a) => a.ownerId === ownerId)
          .sort((x, y) => y.createdAt.getTime() - x.createdAt.getTime());
        return { items: clone(all.slice((page - 1) * limit, page * limit)), total: all.length };
      },
      async findActiveDuplicate(ownerId, sha256) {
        return clone(
          [...store.analyses.values()].find(
            (a) =>
              a.ownerId === ownerId &&
              a.mediaSha256 === sha256 &&
              (a.status === 'queued' || a.status === 'processing'),
          ) ?? null,
        );
      },
      async findStale(before) {
        return clone(
          [...store.analyses.values()].filter(
            (a) =>
              (a.status === 'queued' || a.status === 'processing') &&
              a.updatedAt.getTime() < before.getTime(),
          ),
        );
      },
      async countByExternalCheck(externalCheckId) {
        return [...store.analyses.values()].filter((a) => a.externalCheckId === externalCheckId)
          .length;
      },
      async update(id, patch) {
        const a = store.analyses.get(id);
        if (a) store.analyses.set(id, { ...a, ...clone(patch), updatedAt: now() });
      },
      async delete(id) {
        store.analyses.delete(id);
      },
    },
    explanations: {
      async findByAnalysis(analysisId) {
        return clone(store.explanations.get(analysisId) ?? null);
      },
      async deleteByAnalysis(analysisId) {
        store.explanations.delete(analysisId);
      },
      async upsert(analysisId, doc) {
        store.explanations.set(analysisId, { analysisId, ...clone(doc) });
      },
      async findCachedGroq(inputSha256, promptVersion) {
        const hit = [...store.explanations.values()].find(
          (e) =>
            e.inputSha256 === inputSha256 && !e.fallbackUsed && e.promptVersion === promptVersion,
        );
        return clone((hit as ExplanationWrite | undefined) ?? null);
      },
    },
    externalChecks: {
      async findById(id) {
        return clone(store.externalChecks.get(id) ?? null);
      },
      async delete(id) {
        store.externalChecks.delete(id);
      },
    },
    comparisons: {
      async create(doc) {
        const c: ComparisonRecord = { ...clone(doc), id: newId(), createdAt: now() };
        store.comparisons.set(c.id, c);
        return clone(c);
      },
      async listByAnalysis(analysisId) {
        return clone(
          [...store.comparisons.values()]
            .filter((c) => c.analysisId === analysisId)
            .sort((x, y) => y.createdAt.getTime() - x.createdAt.getTime()),
        );
      },
      async deleteByAnalysis(analysisId) {
        for (const [id, c] of store.comparisons)
          if (c.analysisId === analysisId) store.comparisons.delete(id);
      },
    },
    evalReports: {
      async latest() {
        return clone(store.evalReports.at(-1) ?? null);
      },
      async create(doc) {
        const r: EvalReportRecord = { ...clone(doc), id: newId(), importedAt: now() };
        store.evalReports.push(r);
        return clone(r);
      },
    },
  };
}
