import { Router, type Request } from 'express';
import { z } from 'zod';
import { config } from '../config.js';
import type { AnalysisRecord } from '../data/types.js';
import type { AppDeps } from '../deps.js';
import { logger } from '../logger.js';
import { requireAuth } from '../middleware/auth.js';
import { HttpProblem } from '../middleware/problem.js';
import { createUpload, inspectUpload } from '../middleware/upload.js';
import { rateLimit } from '../rateLimit.js';
import { collectDerivedKeys, DERIVED_NAME } from '../storage.js';
import { presentAnalysis, presentListItem } from './present.js';

/** Analysis ids are MongoDB ObjectIds; anything else cannot exist (404, never 500). */
const OBJECT_ID = /^[0-9a-f]{24}$/i;

export function analysesRouter(deps: AppDeps): Router {
  const { repos, files } = deps;
  const router = Router();
  router.use(requireAuth);
  const upload = createUpload(files.uploadDir, deps.limits);

  async function owned(req: Request): Promise<AnalysisRecord> {
    const id = String(req.params.id ?? '');
    const analysis = OBJECT_ID.test(id) ? await repos.analyses.findOwned(id, req.user!.id) : null;
    // Another user's analysis is reported exactly like a missing one.
    if (!analysis) throw new HttpProblem(404, 'Not Found', 'No such analysis');
    return analysis;
  }

  router.post(
    '/',
    rateLimit(deps.rateLimiters.upload, (req) => req.user!.id),
    upload.single('file'),
    async (req, res) => {
      if (!req.file) {
        throw new HttpProblem(400, 'Invalid request', 'Attach the upload as field "file".');
      }
      const info = await inspectUpload(req.file, deps.limits);
      // The external check sends the file to a third party, so it is off unless asked for.
      const parsed = z.enum(['true', 'false']).default('false').safeParse(req.body?.externalCheck);
      if (!parsed.success) {
        await files.removeUpload(info.storageKey);
        throw new HttpProblem(400, 'Invalid request', 'externalCheck must be "true" or "false".');
      }
      const ownerId = req.user!.id;

      // The same file uploaded again while its analysis is still running: reuse that analysis.
      const running = await repos.analyses.findActiveDuplicate(ownerId, info.sha256);
      if (running) {
        await files.removeUpload(info.storageKey);
        res
          .status(202)
          .json({ analysisId: running.id, jobId: `analysis-${running.id}`, duplicate: true });
        return;
      }

      const media = await repos.media.create({
        ownerId,
        ...info,
        expiresAt: new Date(Date.now() + config.MEDIA_TTL_HOURS * 3600_000),
      });
      const analysis = await repos.analyses.create({
        ownerId,
        mediaId: media.id,
        mediaSha256: info.sha256,
        kind: info.kind,
        externalCheckRequested: parsed.data === 'true',
      });
      const jobId = await deps.queue.enqueue(analysis.id);
      res.status(202).json({ analysisId: analysis.id, jobId });
    },
  );

  router.get('/', async (req, res) => {
    const { page, limit } = z
      .object({
        page: z.coerce.number().int().min(1).max(10_000).default(1),
        limit: z.coerce.number().int().min(1).max(50).default(20),
      })
      .parse(req.query);
    const { items, total } = await repos.analyses.listOwned(req.user!.id, page, limit);
    res.json({ page, limit, total, items: items.map(presentListItem) });
  });

  router.get('/:id', async (req, res) => {
    const a = await owned(req);
    const [external, explanation] = await Promise.all([
      a.externalCheckId ? repos.externalChecks.findById(a.externalCheckId) : null,
      repos.explanations.findByAnalysis(a.id),
    ]);
    res.json(presentAnalysis(a, external, explanation));
  });

  /** Derived images (heatmaps, face crops, filtered copies) that belong to this analysis. */
  router.get('/:id/heatmaps/:name', async (req, res) => {
    const a = await owned(req);
    const name = String(req.params.name);
    if (!DERIVED_NAME.test(name)) {
      throw new HttpProblem(400, 'Invalid request', 'Not a valid image name.');
    }
    const comparisons = await repos.comparisons.listByAnalysis(a.id);
    const allowed = new Set([
      ...a.heatmapKeys,
      ...collectDerivedKeys(a.ml),
      ...comparisons.flatMap((c) => c.heatmapKeys),
    ]);
    if (!allowed.has(`derived/${name}`)) {
      throw new HttpProblem(404, 'Not Found', 'No such image for this analysis');
    }
    const file = await files.readDerived(name);
    if (!file.ok) {
      if (file.reason === 'corrupt') logger.warn({ analysisId: a.id }, 'derived image is corrupt');
      throw new HttpProblem(
        404,
        'Not Found',
        file.reason === 'corrupt'
          ? 'The image file is damaged.'
          : 'The image is no longer available (it may have expired).',
      );
    }
    res
      .status(200)
      .set({
        'Content-Type': 'image/png',
        'Cache-Control': 'private, max-age=3600',
        'Content-Disposition': 'inline',
      })
      .send(file.bytes);
  });

  router.delete('/:id', async (req, res) => {
    const a = await owned(req);
    await deleteAnalysis(deps, a);
    res.status(204).end();
  });

  return router;
}

/** Remove an analysis and everything it produced: upload, derived images, results. */
export async function deleteAnalysis(deps: AppDeps, a: AnalysisRecord): Promise<void> {
  const { repos, files } = deps;
  const comparisons = await repos.comparisons.listByAnalysis(a.id);
  const derived = new Set([
    ...a.heatmapKeys,
    ...collectDerivedKeys(a.ml),
    ...comparisons.flatMap((c) => c.heatmapKeys),
  ]);
  const media = await repos.media.delete(a.mediaId);
  if (media) await files.removeUpload(media.storageKey);
  await files.removeDerived([...derived]);
  await Promise.all([
    repos.explanations.deleteByAnalysis(a.id),
    repos.comparisons.deleteByAnalysis(a.id),
    repos.analyses.delete(a.id),
  ]);
  // The cached second-check result goes too once no other analysis refers to it.
  if (a.externalCheckId && (await repos.analyses.countByExternalCheck(a.externalCheckId)) === 0) {
    await repos.externalChecks.delete(a.externalCheckId);
  }
}
