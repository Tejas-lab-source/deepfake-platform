/**
 * Offline answers to questions about one analysis. Deterministic, built only from AskFacts,
 * and independent of any LLM. Every number written is taken from the facts (as given or as a
 * percentage), which validateAnswer() checks.
 */
import type { AskFacts } from './facts.js';

export const INTENTS = [
  'verdict',
  'why',
  'scores',
  'second_check',
  'frames',
  'heatmap',
  'uncertainty',
  'filters',
  'limitations',
  'model',
] as const;
export type Intent = (typeof INTENTS)[number];

/** Keyword rules, most specific first. A question matching none is not about the result. */
const RULES: Array<[Intent, RegExp]> = [
  [
    'second_check',
    /\b(hive|second (check|opinion)|external|other (check|detector|service)|verif)/i,
  ],
  ['frames', /\b(frames?|timestamps?|seconds?|moments?|timeline|when|which part|clip|scene)\b/i],
  ['heatmap', /\b(heat ?maps?|grad-?cam|highlight|regions?|where (in|on) the (face|image))\b/i],
  [
    'uncertainty',
    /\b(uncertain|unsure|inconclusive|not sure|undecided|why not (fake|real)|margin)\b/i,
  ],
  ['filters', /\b(filters?|beaut|smooth|makeup|reshap|retouch|edited|edit)\w*/i],
  [
    'scores',
    /\b(scores?|probabilit|confiden|threshold|percent|%|logits?|numbers?|how (sure|likely|confident))\w*/i,
  ],
  [
    'limitations',
    /\b(limit|reliab|accura|trust|wrong|mistake|false|error|proof|prove|certain|guarantee)\w*/i,
  ],
  ['model', /\b(models?|detectors?|gend|sbi|clip|efficientnet|version|model card|calibrat)\w*/i],
  ['why', /\b(why|how did|reason|explain|flagged|based on|what made|cause)\b/i],
  [
    'verdict',
    /\b(verdict|result|fake|real|manipulat|deepfake|genuine|authentic|outcome|decision|conclu|is (it|this))\w*/i,
  ],
];

export function classify(question: string): Intent | null {
  for (const [intent, pattern] of RULES) if (pattern.test(question)) return intent;
  return null;
}

export const UNSUPPORTED =
  'I can only answer questions about this analysis result: its verdict, the scores and ' +
  'threshold, the second check, video frames, heatmaps, beauty filters, uncertainty and the ' +
  'limits of automated detection.';

const DISCLAIMER = 'This is an automated check and is not proof either way.';

const pct = (v: number): string => `${Math.round(v * 100)}%`;
const fixed = (v: number): string => (Math.round(v * 100) / 100).toString();
const secs = (v: number | null): string => (v === null ? 'an unknown time' : `${fixed(v)} s`);

const VERDICT_TEXT: Record<string, string> = {
  FAKE: 'signs of manipulation were found',
  REAL: 'no manipulation found',
  UNCERTAIN: 'no reliable verdict could be reached',
};

const BADGE_TEXT: Record<string, string> = {
  verified_by_2: 'both our pipeline and the second check agree',
  second_check_disagrees:
    'the second check disagrees, but our pipeline was confident, so its verdict stands',
  checks_disagree: 'the two checks disagree and our pipeline was not confident',
  hive_hint: 'our pipeline could not decide; the second check is shown only as a hint',
  single_check: 'only our pipeline was used',
};

const UNCERTAIN_TEXT: Record<string, string> = {
  near_threshold: 'the probability was too close to the decision threshold',
  unstable_under_beautification:
    'the score changed too much between the beautified copies of the face',
  variants_disagree: 'the beautified copies of the face did not all lead to the same verdict',
  detectors_disagree: 'the two detectors strongly disagreed',
  no_face_detected:
    'no face was detected, and faces that are not detected are never scored as real',
};

function verdict(f: AskFacts): string {
  const r = f.result;
  const what = VERDICT_TEXT[r.finalVerdict] ?? r.finalVerdict;
  return `Final result: ${r.finalVerdict}, meaning ${what}. Verification: ${
    BADGE_TEXT[r.verificationBadge] ?? r.verificationBadge
  }.`;
}

function ourScore(f: AskFacts): string {
  const o = f.result.ourPipeline;
  if (o.probabilityOfManipulation === null || o.threshold === null) {
    return 'Our pipeline produced no probability for this upload.';
  }
  const side = o.probabilityOfManipulation >= o.threshold ? 'at or above' : 'below';
  return (
    `Our pipeline estimated a ${pct(o.probabilityOfManipulation)} probability of manipulation, ` +
    `${side} its ${pct(o.threshold)} decision threshold` +
    (o.confident ? ', and it was confident.' : ', but it was not confident.')
  );
}

function secondCheck(f: AskFacts): string {
  const s = f.result.secondCheck;
  if (!f.secondCheckRequested || !s) {
    return 'No second check was run: external verification was not requested for this upload.';
  }
  if (s.status !== 'ok' || s.deepfakeScore === null || s.threshold === null) {
    return `The second check (${s.provider}) did not return a score (status: ${s.status}), so only our pipeline decided.`;
  }
  const agree = f.result.checksDisagree
    ? 'It disagrees with our pipeline.'
    : 'It agrees with our pipeline.';
  return (
    `The second check (${s.provider}) gave a deepfake score of ${pct(s.deepfakeScore)} against ` +
    `its ${pct(s.threshold)} threshold, so it reads ${s.verdict ?? 'no verdict'}. ${agree} ` +
    'The second check can confirm our verdict or turn it into UNCERTAIN, but it never flips it.'
  );
}

function frames(f: AskFacts): string {
  const v = f.video;
  if (f.kind !== 'video' || !v)
    return 'This analysis is of a single image, so there are no video frames.';
  const parts = [
    `${v.framesSampled} frames were sampled evenly from the video and ${v.framesScored} of them contained a face that could be scored.`,
    'Frames without a detected face are shown as gaps and are not counted as real.',
  ];
  if (v.topK !== null)
    parts.push(`The video verdict combines the ${v.topK} most suspicious frames.`);
  if (v.mostSuspicious) {
    parts.push(`The most suspicious frame is at ${secs(v.mostSuspicious.timestamp)}.`);
  }
  if (v.leastSuspicious && v.leastSuspicious.index !== v.mostSuspicious?.index) {
    parts.push(`The least suspicious frame is at ${secs(v.leastSuspicious.timestamp)}.`);
  }
  return parts.join(' ');
}

function heatmap(f: AskFacts): string {
  if (!f.faces.hasHeatmap) {
    return 'There is no heatmap for this result (heatmaps need the real model pipeline and a detected face).';
  }
  const base =
    'The heatmaps show which parts of the face most influenced each detector (Grad-CAM). ' +
    'They show where the models looked, not proof of where an edit is.';
  return f.faces.heatmapIoU === null
    ? base
    : `${base} The overlap between the heatmaps of the clean and the most changed beautified copy was ${fixed(f.faces.heatmapIoU)} (the maximum, one, would mean identical regions).`;
}

function uncertainty(f: AskFacts): string {
  const o = f.result.ourPipeline;
  if (f.result.finalVerdict !== 'UNCERTAIN') {
    return `The result is not UNCERTAIN. ${ourScore(f)}`;
  }
  if (o.verdict === 'UNCERTAIN') {
    const reason = o.uncertainReason
      ? (UNCERTAIN_TEXT[o.uncertainReason] ?? o.uncertainReason)
      : 'the decision rules were not met';
    return `The result is UNCERTAIN because ${reason}.`;
  }
  return 'Our pipeline reached a verdict but was not confident, and the second check disagreed, so the result was downgraded to UNCERTAIN.';
}

function filters(f: AskFacts): string {
  const o = f.result.ourPipeline;
  const parts = [
    'The pipeline also scores beautified copies of each face (skin smoothing, makeup, reshaping) so that beauty filters do not flip the verdict.',
  ];
  if (o.scoreSpreadAcrossFilteredCopies !== null) {
    parts.push(
      `The spread of the score across those copies was ${fixed(o.scoreSpreadAcrossFilteredCopies)}; a large spread makes the result UNCERTAIN.`,
    );
  }
  if (f.faces.beautifiedLikelihood !== null) {
    parts.push(
      `A training-free smoothing cue rated the face ${fixed(f.faces.beautifiedLikelihood)} on a zero-to-one scale; it is a heuristic, not a verdict.`,
    );
  }
  return parts.join(' ');
}

function limitations(): string {
  return (
    'Automated deepfake detection is probabilistic: it can miss manipulations and it can flag ' +
    'unedited media, especially after heavy compression, unusual lighting or new editing tools. ' +
    'Use the result as one piece of evidence and check important cases by other means.'
  );
}

function model(f: AskFacts): string {
  const v = f.result.modelCardVersion;
  return (
    'Our pipeline uses two frozen detectors: GenD (a CLIP vision transformer) with a ' +
    'filter-invariant linear head, and SBI (an EfficientNet convolutional network). Their scores ' +
    'are fused and calibrated' +
    (v ? ` (model card ${v}).` : '.') +
    ' The Model Card page lists the thresholds and where each came from.'
  );
}

export function templateAnswer(intent: Intent, f: AskFacts): string {
  const mock = f.result.mockMode
    ? ' Mock mode is on: the scores are placeholders, not model outputs.'
    : '';
  let body: string;
  switch (intent) {
    case 'verdict':
      body = `${verdict(f)} ${ourScore(f)}`;
      break;
    case 'why':
      body = [verdict(f), ourScore(f), f.kind === 'video' ? frames(f) : '', secondCheck(f)]
        .filter(Boolean)
        .join(' ');
      break;
    case 'scores':
      body = `${ourScore(f)} ${secondCheck(f)}`;
      break;
    case 'second_check':
      body = secondCheck(f);
      break;
    case 'frames':
      body = frames(f);
      break;
    case 'heatmap':
      body = heatmap(f);
      break;
    case 'uncertainty':
      body = uncertainty(f);
      break;
    case 'filters':
      body = filters(f);
      break;
    case 'limitations':
      body = limitations();
      break;
    case 'model':
      body = model(f);
      break;
  }
  return `${body}${mock} ${DISCLAIMER}`;
}
