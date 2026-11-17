import type { Analysis, AnalysisListItem, Frame } from '../api/types';

export const imageAnalysis = (over: Partial<Analysis> = {}): Analysis => ({
  id: 'a1',
  kind: 'image',
  status: 'done',
  stage: 'done',
  error: null,
  createdAt: '2026-11-17T10:00:00Z',
  externalCheckRequested: false,
  final: { verdict: 'FAKE', badge: 'single_check', ruleApplied: 'external_unavailable:opted_out' },
  ours: {
    verdict: 'FAKE',
    probability: 0.91,
    threshold: 0.5,
    confident: true,
    uncertainReason: null,
    spread: 0.2,
  },
  external: null,
  explanation: {
    summary: 'The automated check found signs of manipulation.',
    why: 'Our pipeline scored 91% against a 50% threshold.',
    caveats: ['Automated, not proof.'],
    groqModel: null,
    fallbackUsed: true,
  },
  ml: {
    modelCardVersion: 'mc-1',
    faces: [
      {
        box: [10, 20, 100, 100],
        scores: { gendHead: 2.1, sbi: 1.3 },
        variantLogits: { clean: 2, F1: 2.2 },
        spread: 0.2,
        beautifiedLikelihood: 0.31,
        heatmaps: {
          gend: 'derived/' + 'a'.repeat(32) + '.png',
          sbi: 'derived/' + 'b'.repeat(32) + '.png',
        },
        crops: {
          gend: 'derived/' + 'c'.repeat(32) + '.png',
          sbi: 'derived/' + 'd'.repeat(32) + '.png',
        },
        heatmapIoU: 0.42,
      },
    ],
  },
  ...over,
});

export const frames: Frame[] = [
  { index: 5, timestamp: 0.5, hasFace: true, logit: 0.7, probability: 0.67, explained: true },
  { index: 15, timestamp: 1.5, hasFace: false, logit: null, decoded: true },
  { index: 25, timestamp: 2.5, hasFace: true, logit: 2.4, probability: 0.92, explained: true },
  { index: 35, timestamp: 3.5, hasFace: false, logit: null, decoded: false },
];

export const videoAnalysis = (): Analysis =>
  imageAnalysis({
    id: 'v1',
    kind: 'video',
    ml: {
      kind: 'video',
      faces: [],
      frames,
      framesSampled: 4,
      framesScored: 2,
      video: { fps: 10, frameCount: 40, durationSec: 4, sampledFrames: 4 },
      aggregate: { method: 'top_k_mean', k: 2, logit: 1.55 },
      mostSuspicious: { index: 25, timestamp: 2.5, logit: 2.4 },
      leastSuspicious: { index: 5, timestamp: 0.5, logit: 0.7 },
      modelCardVersion: 'mc-1',
    },
  });

export const listItem = (i: number, over: Partial<AnalysisListItem> = {}): AnalysisListItem => ({
  id: `id${i}`,
  kind: 'image',
  status: 'done',
  stage: 'done',
  final: { verdict: 'REAL', badge: 'single_check' },
  ours: { probability: 0.12 },
  createdAt: '2026-11-17T10:00:00Z',
  ...over,
});
