import { Router } from 'express';
import mongoose from 'mongoose';
import { z } from 'zod';
import { config } from '../config.js';
import { requireAuth } from '../middleware/auth.js';
import { HttpProblem } from '../middleware/problem.js';
import { inspectUpload, removeStored, upload } from '../middleware/upload.js';
import { Analysis, Explanation, ExternalCheck, Media } from '../models/index.js';
import { getAnalysisQueue } from '../queue/queues.js';

export const analysesRouter = Router();
analysesRouter.use(requireAuth);

async function findOwned(id: string | undefined, ownerId: string) {
  const analysis = mongoose.isValidObjectId(id)
    ? await Analysis.findOne({ _id: id, ownerId }).lean()
    : null;
  if (!analysis) throw new HttpProblem(404, 'Not Found', 'No such analysis');
  return analysis;
}

analysesRouter.post('/', upload.single('file'), async (req, res) => {
  if (!req.file)
    throw new HttpProblem(400, 'Invalid request', 'Attach the upload as field "file".');
  const info = await inspectUpload(req.file);
  // The external check sends the file to a third party, so it is off unless asked for.
  const externalCheck = z.enum(['true', 'false']).default('false').parse(req.body?.externalCheck);
  const ownerId = req.user!.id;
  const media = await Media.create({
    ownerId,
    ...info,
    expiresAt: new Date(Date.now() + config.MEDIA_TTL_HOURS * 3600_000),
  });
  const analysis = await Analysis.create({
    ownerId,
    mediaId: media._id,
    kind: info.kind,
    externalCheckRequested: externalCheck === 'true',
  });
  const analysisId = analysis.id as string;
  const job = await getAnalysisQueue().add(
    'analyze',
    { analysisId },
    { jobId: `analysis-${analysisId}` },
  );
  res.status(202).json({ analysisId, jobId: job.id });
});

analysesRouter.get('/', async (req, res) => {
  const { page, limit } = z
    .object({
      page: z.coerce.number().int().min(1).default(1),
      limit: z.coerce.number().int().min(1).max(50).default(20),
    })
    .parse(req.query);
  const filter = { ownerId: req.user!.id };
  const [items, total] = await Promise.all([
    Analysis.find(filter)
      .sort({ createdAt: -1 })
      .skip((page - 1) * limit)
      .limit(limit)
      .select('kind status stage final ours.probability createdAt')
      .lean(),
    Analysis.countDocuments(filter),
  ]);
  res.json({
    page,
    limit,
    total,
    items: items.map(({ _id, ...rest }) => ({ id: String(_id), ...rest })),
  });
});

analysesRouter.get('/:id', async (req, res) => {
  const a = await findOwned(req.params.id, req.user!.id);
  const [external, explanation] = await Promise.all([
    a.externalCheckId ? ExternalCheck.findById(a.externalCheckId).lean() : null,
    Explanation.findOne({ analysisId: a._id }).lean(),
  ]);
  res.json({
    id: String(a._id),
    kind: a.kind,
    status: a.status,
    stage: a.stage,
    error: a.error,
    createdAt: a.createdAt,
    externalCheckRequested: a.externalCheckRequested,
    final: a.final,
    ours: a.ours,
    external: external && {
      provider: external.provider,
      status: external.status,
      deepfakeScore: external.deepfakeScore,
      aiGeneratedScore: external.aiGeneratedScore,
      threshold: external.threshold,
      verdict: external.verdict,
      latencyMs: external.latencyMs,
    },
    explanation: explanation && {
      summary: explanation.summary,
      why: explanation.why,
      caveats: explanation.caveats,
      groqModel: explanation.groqModel,
      fallbackUsed: explanation.fallbackUsed,
    },
    ml: a.ml,
  });
});

analysesRouter.delete('/:id', async (req, res) => {
  const a = await findOwned(req.params.id, req.user!.id);
  const media = await Media.findByIdAndDelete(a.mediaId).lean();
  if (media) await removeStored(media.storageKey);
  await Promise.all([
    Explanation.deleteOne({ analysisId: a._id }),
    Analysis.deleteOne({ _id: a._id }),
  ]);
  res.status(204).end();
});
