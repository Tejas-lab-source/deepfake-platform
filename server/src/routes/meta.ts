/**
 * Model card and evaluation results. The model card combines what the ML service reports
 * (detectors, calibration values and where each came from) with the server's own decision
 * settings and the licences of everything used. Evaluation reports are imported from the
 * offline leave-one-family-out run (ml-service/offline/eval_lofo.py) by an admin.
 */
import express, { Router, type RequestHandler } from 'express';
import { z } from 'zod';
import { config } from '../config.js';
import type { EvalRunRecord, NewEvalReport } from '../data/types.js';
import type { AppDeps } from '../deps.js';
import { requireAuth } from '../middleware/auth.js';
import { HttpProblem } from '../middleware/problem.js';
import { MlServiceError } from '../providers/mlClient.js';

export const LICENCES = [
  { component: 'GenD (CLIP ViT-L/14), yermandy/GenD_CLIP_L_14', licence: 'MIT' },
  { component: 'SBI EfficientNet-B4 weights (FF++ c23)', licence: 'Research use only' },
  { component: 'MediaPipe Face Landmarker', licence: 'Apache-2.0' },
  { component: 'facenet-pytorch (MTCNN)', licence: 'MIT' },
  {
    component: 'CodeFormer / GFPGAN (F4, test only, optional)',
    licence: 'Non-commercial / Apache-2.0',
  },
  { component: 'Hive deepfake detection API (second check)', licence: "Hive's terms of service" },
  { component: 'Groq API (explanations; off by default)', licence: "Groq's terms of service" },
];

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** Only the documented, non-sensitive fields of the ML service's /model-info. */
export function presentModelCard(info: Record<string, unknown>) {
  const cal = (info.calibration ?? {}) as Record<string, unknown>;
  return {
    mockMode: info.mockMode === true,
    detectors: {
      primary: typeof info.gend === 'string' ? info.gend : 'yermandy/GenD_CLIP_L_14',
      secondary: 'SBI EfficientNet-B4 (FF++ c23)',
      sbiLoaded: info.sbiLoaded ?? null,
      headLoaded: info.headLoaded ?? null,
      device: info.device ?? null,
      fp16: info.fp16 ?? null,
    },
    pipeline: {
      beautifiedCopiesPerImage: num(info.k),
      beautifiedCopiesPerVideoFrame: config.VIDEO_K,
      maxVideoFrames: num(info.maxFrames),
      videoTopK: num(info.videoTopK),
    },
    calibration: {
      version: typeof cal.version === 'string' ? cal.version : null,
      status: info.calibrationStatus ?? cal.calibration_status ?? null,
      fixture: info.fixtureModelCard === true || cal.fixture === true,
      temperature: num(cal.temperature),
      threshold: num(cal.threshold),
      uncertainMargin: num(cal.uncertain_margin),
      confidentMargin: num(cal.confident_margin),
      spreadLimit: num(cal.spread_limit),
      detectorDisagreeLimit: num(cal.detector_disagree_limit),
      fusion: cal.fusion ?? null,
      sources: cal.sources ?? {},
      headSha256: info.headSha256 ?? cal.head_sha256 ?? null,
    },
    secondCheck: {
      provider: 'Hive',
      mode: config.HIVE_MODE,
      threshold: config.HIVE_THRESHOLD,
      rule: 'Hive can confirm our verdict or downgrade it to UNCERTAIN; it never flips it.',
    },
    explanations: {
      provider: 'Groq',
      mode: config.GROQ_MODE,
      rule: 'Groq only words the result JSON; it never sees media and never sets the verdict.',
    },
    licences: LICENCES,
    notice: 'Automated, probabilistic detection. A result is not proof of manipulation.',
  };
}

const metric = z.number().nullable();
const lofoReport = z.object({
  schema: z.literal(1),
  protocol: z.unknown().optional(),
  fixture: z.boolean(),
  notice: z.string().nullable().optional(),
  created: z.string().nullable().optional(),
  inputs: z.record(z.unknown()).optional(),
  runs: z
    .array(
      z.object({
        scheme: z.string().max(20),
        fold: z.string().max(20),
        seed: z.number().int(),
        held_out: z.boolean(),
        metrics: z.record(metric),
      }),
    )
    .max(1000),
  per_fold: z.record(z.record(z.unknown())),
  aggregate_over_held_out_folds: z.record(z.unknown()),
  ablation_shipped_head_clean_test_inputs: z.unknown().optional(),
  success_criteria: z.array(z.unknown()).optional(),
  heatmap_iou_seed0: z.unknown().optional(),
  code_commit: z.string().nullable().optional(),
});

/** Convert eval_lofo.py's output into stored runs (one per scheme x fold x seed). */
export function evalReportFromLofo(raw: unknown): NewEvalReport {
  const r = lofoReport.parse(raw);
  const runs: EvalRunRecord[] = r.runs.map((run) => {
    const fold = r.per_fold[`${run.scheme}__${run.fold}`] ?? {};
    const ci95: EvalRunRecord['ci95'] = {};
    for (const name of Object.keys(run.metrics)) {
      const ci = (fold[name] as { ci95?: unknown } | undefined)?.ci95;
      ci95[name] =
        Array.isArray(ci) && ci.length === 2 && ci.every((v) => typeof v === 'number')
          ? [ci[0] as number, ci[1] as number]
          : null;
    }
    return {
      scheme: run.scheme,
      fold: run.fold,
      seed: run.seed,
      heldOut: run.held_out,
      metrics: run.metrics,
      ci95,
    };
  });
  return {
    created: r.created ?? null,
    fixture: r.fixture,
    notice: r.notice ?? null,
    protocol: r.protocol ?? null,
    inputs: r.inputs ?? {},
    aggregate: r.aggregate_over_held_out_folds,
    ablation: r.ablation_shipped_head_clean_test_inputs ?? null,
    successCriteria: r.success_criteria ?? [],
    heatmapIoU: r.heatmap_iou_seed0 ?? null,
    codeCommit: r.code_commit ?? null,
    runs,
  };
}

const requireAdmin: RequestHandler = (req, _res, next) => {
  if (req.user?.role !== 'admin') {
    return next(new HttpProblem(403, 'Forbidden', 'Only an admin can import evaluation results'));
  }
  next();
};

export function metaRouter(deps: AppDeps): Router {
  const router = Router();

  router.get('/model-card', requireAuth, async (_req, res) => {
    try {
      res.json(presentModelCard(await deps.ml.modelInfo()));
    } catch (err) {
      if (err instanceof MlServiceError) {
        throw new HttpProblem(
          503,
          'Model information unavailable',
          'The ML service is not reachable right now.',
        );
      }
      throw err;
    }
  });

  router.get('/eval-runs', requireAuth, async (_req, res) => {
    const report = await deps.repos.evalReports.latest();
    res.json({ report });
  });

  router.post(
    '/eval-runs',
    requireAuth,
    requireAdmin,
    express.json({ limit: '5mb' }),
    async (req, res) => {
      const report = await deps.repos.evalReports.create(evalReportFromLofo(req.body));
      res.status(201).json({ id: report.id, runs: report.runs.length, fixture: report.fixture });
    },
  );

  return router;
}
