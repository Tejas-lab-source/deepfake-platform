/**
 * The reviewer PDF for one completed analysis.
 *
 * buildReportData() picks the approved fields; renderReportPdf() lays them out. The report
 * never contains: API keys, email addresses, original file names, filesystem paths, SHA-256
 * hashes, storage keys or raw provider responses (none of them are in ReportData).
 */
import PDFDocument from 'pdfkit';
import { videoFacts, type VideoFacts } from '../analysisFacts.js';
import type { AnalysisRecord, ExplanationRecord, ExternalCheckRecord } from '../data/types.js';

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

export interface ReportData {
  analysisId: string;
  kind: 'image' | 'video';
  createdAt: string;
  completedAt: string | null;
  generatedAt: string;
  mock: boolean;
  modelCardVersion: string | null;
  final: { verdict: string; badge: string; ruleApplied: string };
  ours: {
    verdict: string;
    probability: number | null;
    threshold: number;
    confident: boolean;
    uncertainReason: string | null;
    gendLogit: number | null;
    sbiLogit: number | null;
    spread: number | null;
  };
  checks: {
    facesDetected: number;
    beautifiedLikelihood: number | null;
    heatmapIoU: number | null;
    variantLogits: Record<string, number>;
  };
  secondCheck:
    | { requested: false }
    | {
        requested: true;
        provider: string;
        status: string;
        deepfakeScore: number | null;
        threshold: number | null;
        verdict: string | null;
        reason: string | null;
      };
  explanation: { summary: string; why: string; caveats: string[]; source: string } | null;
  video: VideoFacts | null;
}

export function buildReportData(
  a: AnalysisRecord,
  external: ExternalCheckRecord | null,
  explanation: ExplanationRecord | null,
  now: Date = new Date(),
): ReportData {
  if (!a.final || !a.ours) throw new Error('report needs a finished analysis');
  const ml = (a.ml ?? {}) as Record<string, unknown>;
  const face = (Array.isArray(ml.faces) ? ml.faces[0] : undefined) as
    Record<string, unknown> | undefined;
  const scores = (face?.scores ?? {}) as Record<string, unknown>;
  const variants = (face?.variantLogits ?? {}) as Record<string, unknown>;
  return {
    analysisId: a.id,
    kind: a.kind,
    createdAt: a.createdAt.toISOString(),
    completedAt: a.completedAt ? a.completedAt.toISOString() : null,
    generatedAt: now.toISOString(),
    mock: ml.mock === true,
    modelCardVersion: a.modelCardVersion,
    final: a.final,
    ours: {
      verdict: a.ours.verdict,
      probability: a.ours.probability,
      threshold: a.ours.threshold,
      confident: a.ours.confident,
      uncertainReason: a.ours.uncertainReason,
      gendLogit: num(scores.gendHead),
      sbiLogit: num(scores.sbi),
      spread: a.ours.spread,
    },
    checks: {
      facesDetected: Array.isArray(ml.faces) ? ml.faces.length : 0,
      beautifiedLikelihood: num(face?.beautifiedLikelihood),
      heatmapIoU: num(face?.heatmapIoU),
      variantLogits: Object.fromEntries(
        Object.entries(variants).filter((e): e is [string, number] => num(e[1]) !== null),
      ),
    },
    secondCheck: a.externalCheckRequested
      ? {
          requested: true,
          provider: external?.provider ?? 'hive',
          status: external?.status ?? 'error',
          deepfakeScore: external?.deepfakeScore ?? null,
          threshold: external?.threshold ?? null,
          verdict: external?.verdict ?? null,
          reason: external?.reason ?? null,
        }
      : { requested: false },
    explanation: explanation && {
      summary: explanation.summary,
      why: explanation.why,
      caveats: explanation.caveats,
      source: explanation.fallbackUsed
        ? 'template (no LLM)'
        : `Groq (${explanation.groqModel ?? 'model'})`,
    },
    video: videoFacts(a),
  };
}

const NOTICE =
  'Automated finding, not proof. Deepfake detection is probabilistic: it can miss manipulations ' +
  'and can flag unedited media. Treat this report as one piece of evidence.';

const VERDICT_TEXT: Record<string, string> = {
  FAKE: 'FAKE: signs of manipulation were found',
  REAL: 'No manipulation found',
  UNCERTAIN: 'UNCERTAIN: no reliable verdict could be reached',
};

const pct = (v: number | null): string => (v === null ? 'n/a' : `${(v * 100).toFixed(1)}%`);
const dec = (v: number | null, d = 3): string => (v === null ? 'n/a' : v.toFixed(d));
const secs = (v: number | null): string => (v === null ? 'n/a' : `${v.toFixed(2)} s`);

export function renderReportPdf(r: ReportData): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const doc = new PDFDocument({
      size: 'A4',
      margin: 50,
      bufferPages: true,
      info: {
        Title: `Deepfake analysis report ${r.analysisId}`,
        Author: 'Deepfake detection platform',
      },
    });
    const chunks: Buffer[] = [];
    doc.on('data', (c: Buffer) => chunks.push(c));
    doc.on('end', () => resolve(Buffer.concat(chunks)));
    doc.on('error', reject);

    const heading = (text: string) => {
      doc.moveDown(0.8).font('Helvetica-Bold').fontSize(13).fillColor('#111827').text(text);
      doc.moveDown(0.3).font('Helvetica').fontSize(10).fillColor('#111827');
    };
    const row = (label: string, value: string) => {
      doc
        .font('Helvetica-Bold')
        .text(`${label}: `, { continued: true })
        .font('Helvetica')
        .text(value);
    };

    doc.font('Helvetica-Bold').fontSize(18).text('Automated deepfake analysis report');
    doc.moveDown(0.4).font('Helvetica').fontSize(9).fillColor('#92400e').text(NOTICE);
    doc.fillColor('#111827').fontSize(10);
    if (r.mock) {
      doc
        .moveDown(0.4)
        .font('Helvetica-Bold')
        .fillColor('#b45309')
        .text('MOCK MODE: the scores are placeholders derived from the file, not model outputs.');
      doc.fillColor('#111827').font('Helvetica');
    }

    heading('Analysis');
    row('Analysis ID', r.analysisId);
    row('Media type', r.kind);
    row('Submitted (UTC)', r.createdAt);
    row('Completed (UTC)', r.completedAt ?? 'n/a');
    row('Report generated (UTC)', r.generatedAt);
    row('Model card version', r.modelCardVersion ?? 'n/a');

    heading('Final result');
    row('Verdict', VERDICT_TEXT[r.final.verdict] ?? r.final.verdict);
    row('Verification badge', r.final.badge);
    row('Rule applied', r.final.ruleApplied);

    heading('Our pipeline');
    row('Verdict', r.ours.verdict);
    row('Probability of manipulation', pct(r.ours.probability));
    row('Decision threshold', pct(r.ours.threshold));
    row('Confident', r.ours.confident ? 'yes' : 'no');
    row('Uncertain reason', r.ours.uncertainReason ?? 'none');
    row('GenD head logit', dec(r.ours.gendLogit));
    row('SBI logit', dec(r.ours.sbiLogit));
    row('Score spread across beautified copies', dec(r.ours.spread));

    heading('Pipeline checks');
    row('Faces detected (deciding image/frame)', String(r.checks.facesDetected));
    row('Beautification cue (0-1, heuristic)', dec(r.checks.beautifiedLikelihood));
    row('Heatmap overlap, clean vs beautified (IoU)', dec(r.checks.heatmapIoU));
    const variants = Object.entries(r.checks.variantLogits);
    if (variants.length > 0) {
      row('Logit per copy', variants.map(([k, v]) => `${k} ${v.toFixed(3)}`).join(', '));
    }

    heading('Second check');
    if (!r.secondCheck.requested) {
      doc.text('Not requested for this upload (the media was not sent to any third party).');
    } else {
      const s = r.secondCheck;
      row('Provider', s.provider);
      row('Status', s.status + (s.reason ? ` (${s.reason})` : ''));
      row('Deepfake score', pct(s.deepfakeScore));
      row('Provider threshold', pct(s.threshold));
      row('Provider verdict', s.verdict ?? 'n/a');
      doc.text(
        'The second check can confirm our verdict or downgrade it to UNCERTAIN; it never flips it.',
      );
    }

    if (r.video) {
      const v = r.video;
      heading('Video frames');
      row('Frames sampled', String(v.framesSampled));
      row('Frames with a scored face', String(v.framesScored));
      row('Duration', secs(v.durationSec));
      row(
        'Aggregation',
        v.topK === null
          ? 'n/a'
          : `mean of the top ${v.topK} frame logits = ${dec(v.aggregateLogit)}`,
      );
      row(
        'Most suspicious frame',
        v.mostSuspicious
          ? `#${v.mostSuspicious.index} at ${secs(v.mostSuspicious.timestamp)} (logit ${dec(v.mostSuspicious.logit)})`
          : 'n/a',
      );
      row(
        'Least suspicious frame',
        v.leastSuspicious
          ? `#${v.leastSuspicious.index} at ${secs(v.leastSuspicious.timestamp)} (logit ${dec(v.leastSuspicious.logit)})`
          : 'n/a',
      );
      doc
        .moveDown(0.4)
        .font('Helvetica-Bold')
        .text('Frame  Time        Face   Logit     Probability  Verdict');
      doc.font('Courier').fontSize(9);
      for (const f of v.frames) {
        const line = [
          `#${f.index}`.padEnd(6),
          secs(f.timestamp).padEnd(11),
          (f.decoded ? (f.hasFace ? 'yes' : 'no') : 'n/d').padEnd(6),
          dec(f.logit).padEnd(9),
          pct(f.probability).padEnd(12),
          f.verdict ?? (f.hasFace ? '' : 'excluded'),
        ].join(' ');
        doc.text(line);
      }
      doc.font('Helvetica').fontSize(10);
      doc.text(
        'Frames without a detected face are excluded from the verdict, never counted as real.',
      );
    }

    heading('Explanation');
    if (r.explanation) {
      row('Written by', r.explanation.source);
      doc.moveDown(0.2).text(r.explanation.summary).moveDown(0.3).text(r.explanation.why);
      if (r.explanation.caveats.length > 0) {
        doc.moveDown(0.3).font('Helvetica-Bold').text('Caveats').font('Helvetica');
        for (const c of r.explanation.caveats) doc.text(`- ${c}`);
      }
    } else {
      doc.text('No explanation is available.');
    }

    // The notice again on every page footer.
    const range = doc.bufferedPageRange();
    for (let i = range.start; i < range.start + range.count; i++) {
      doc.switchToPage(i);
      const bottom = doc.page.margins.bottom;
      doc.page.margins.bottom = 0;
      doc
        .font('Helvetica')
        .fontSize(8)
        .fillColor('#6b7280')
        .text(
          `Automated finding, not proof. Page ${i + 1} of ${range.count}.`,
          50,
          doc.page.height - 35,
          {
            width: doc.page.width - 100,
            align: 'center',
            lineBreak: false,
          },
        );
      doc.page.margins.bottom = bottom;
    }
    doc.end();
  });
}
