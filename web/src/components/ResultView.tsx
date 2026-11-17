import { useState } from 'react';
import { Link } from 'react-router-dom';
import { api, saveBlob } from '../api/client';
import type { Analysis } from '../api/types';
import AskPanel from './AskPanel';
import ConfidenceGauge from './ConfidenceGauge';
import FaceGallery from './FaceGallery';
import FrameTimeline from './FrameTimeline';
import { card, Disclaimer } from './states';
import { BADGE_LABEL, pct, VERDICT_LABEL, VERDICT_STYLE } from './verdict';

export const UNCERTAIN_REASON: Record<string, string> = {
  near_threshold: 'Score too close to the threshold',
  unstable_under_beautification: 'Score unstable under beauty filters',
  variants_disagree: 'Filtered copies disagree',
  detectors_disagree: 'The two detectors disagree',
  no_face_detected: 'No face detected (never counted as real)',
};

const RULE_TEXT: Record<string, string> = {
  both_fake: 'Both checks found manipulation.',
  both_real: 'Neither check found manipulation.',
  ours_confident_fake_external_disagrees: 'Our confident FAKE stands; the second check disagrees.',
  ours_confident_real_external_disagrees:
    'Our confident result stands; the second check disagrees.',
  ours_not_confident_external_disagrees: 'Not confident and the checks disagree: UNCERTAIN.',
  ours_uncertain: 'Our pipeline could not decide; the second check is only a hint.',
};

const UNAVAILABLE_TEXT: Record<string, string> = {
  opted_out: 'the second check was switched off for this upload',
  error: 'the second check failed',
  timeout: 'the second check timed out',
  quota: 'the second check had no budget left',
};

export function ruleText(rule: string): string {
  if (rule.startsWith('external_unavailable:')) {
    const why = rule.slice('external_unavailable:'.length);
    return `Only our pipeline decided (${UNAVAILABLE_TEXT[why] ?? why}).`;
  }
  return RULE_TEXT[rule] ?? rule;
}

function downloadJson(analysis: Analysis) {
  saveBlob(
    new Blob([JSON.stringify(analysis, null, 2)], { type: 'application/json' }),
    `analysis-${analysis.id}.json`,
  );
}

function PdfButton({ id }: { id: string }) {
  const [state, setState] = useState<'idle' | 'busy' | 'error'>('idle');
  return (
    <span className="inline-flex items-center gap-2">
      <button
        className="rounded border border-slate-300 bg-white px-4 py-2 text-sm hover:bg-slate-100 disabled:opacity-50"
        disabled={state === 'busy'}
        onClick={() => {
          setState('busy');
          api
            .report(id)
            .then((b) => {
              saveBlob(b, `analysis-${id}.pdf`);
              setState('idle');
            })
            .catch(() => setState('error'));
        }}
      >
        {state === 'busy' ? 'Preparing PDF…' : 'Download PDF report'}
      </button>
      {state === 'error' && (
        <span role="alert" className="text-sm text-rose-700">
          The report could not be generated.
        </span>
      )}
    </span>
  );
}

export default function ResultView({
  analysis,
  preview,
}: {
  analysis: Analysis;
  preview?: string | null;
}) {
  const { final, ours, external, explanation, ml } = analysis;
  if (!final || !ours) return null;
  const style = VERDICT_STYLE[final.verdict];
  const faces = ml?.faces ?? [];
  const mock = Boolean(ml?.mock);
  const isVideo = analysis.kind === 'video';

  return (
    <div className="space-y-4">
      <section className={`rounded-lg border p-5 ${style.panel}`} aria-label="Final verdict">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm text-slate-600">Final verdict</p>
            <h2 className={`text-3xl font-bold ${style.text}`}>{VERDICT_LABEL[final.verdict]}</h2>
          </div>
          <span className="rounded-full border border-slate-300 bg-white px-3 py-1 text-sm font-medium">
            {BADGE_LABEL[final.badge]}
          </span>
        </div>
        {final.verdict === 'UNCERTAIN' && (
          <p className="mt-3 text-sm font-medium text-amber-900">
            No reliable verdict:{' '}
            {ours.uncertainReason
              ? (UNCERTAIN_REASON[ours.uncertainReason] ?? ours.uncertainReason)
              : 'the two checks disagree and our pipeline was not confident'}
            .
          </p>
        )}
        <p className="mt-3 text-sm text-slate-700">
          This is an automated check. It is not proof either way.
        </p>
        {mock && (
          <p className="mt-2 text-sm font-medium text-amber-900">
            Mock mode: the scores are placeholders derived from the file, not model outputs.
          </p>
        )}
      </section>

      <div className="grid gap-4 md:grid-cols-2">
        <section className={card}>
          <h3 className="mb-3 font-semibold">Our pipeline</h3>
          {ours.probability !== null ? (
            <ConfidenceGauge
              label="Likelihood of manipulation"
              value={ours.probability}
              threshold={ours.threshold}
              verdict={ours.verdict}
            />
          ) : (
            <p className="text-sm text-slate-600">No score: nothing could be scored.</p>
          )}
          <dl className="mt-4 grid grid-cols-2 gap-y-1 text-sm">
            <dt className="text-slate-500">Verdict</dt>
            <dd>{VERDICT_LABEL[ours.verdict]}</dd>
            <dt className="text-slate-500">Confident</dt>
            <dd>{ours.confident ? 'Yes' : 'No'}</dd>
            {ours.uncertainReason && (
              <>
                <dt className="text-slate-500">Why uncertain</dt>
                <dd>{UNCERTAIN_REASON[ours.uncertainReason] ?? ours.uncertainReason}</dd>
              </>
            )}
            {ours.spread !== null && (
              <>
                <dt className="text-slate-500">Spread across filters</dt>
                <dd className="tabular-nums">{ours.spread.toFixed(2)}</dd>
              </>
            )}
          </dl>
        </section>

        <section className={card}>
          <h3 className="mb-3 font-semibold">
            Second check{external ? ` (${external.provider})` : ''}
          </h3>
          {external?.status === 'ok' && external.deepfakeScore !== null ? (
            <>
              <ConfidenceGauge
                label="Deepfake score"
                value={external.deepfakeScore}
                threshold={external.threshold}
                verdict={external.verdict === 'FAKE' ? 'FAKE' : 'REAL'}
              />
              {external.aiGeneratedScore !== null && (
                <p className="mt-4 text-sm text-slate-600">
                  AI-generated score:{' '}
                  <span className="tabular-nums">{pct(external.aiGeneratedScore)}</span>
                </p>
              )}
            </>
          ) : (
            <p className="text-sm text-slate-600">
              {analysis.externalCheckRequested
                ? `Not available for this upload (${external?.reason ?? external?.status ?? 'error'}).`
                : 'Switched off for this upload: the file was not sent to any third party.'}
            </p>
          )}
          <p className="mt-3 text-xs text-slate-500">
            The second check can confirm our verdict or turn it into Uncertain; it never flips it.
          </p>
        </section>
      </div>

      <section className={card}>
        <h3 className="mb-2 font-semibold">Pipeline checks</h3>
        <ul className="space-y-1 text-sm">
          <li>
            Faces:{' '}
            {isVideo
              ? `${ml?.framesScored ?? 0} of ${ml?.framesSampled ?? 0} sampled frames had a scored face`
              : `${faces.length} detected`}
            .
          </li>
          <li>
            Beauty-filter stability:{' '}
            {ours.spread === null
              ? 'not measured'
              : `spread ${ours.spread.toFixed(2)} across copies`}
            .
          </li>
          <li>Decision rule: {ruleText(final.ruleApplied)}</li>
          <li>
            Model card: {ml?.modelCardVersion ?? 'unknown'} (
            <Link className="underline" to="/model-card">
              details
            </Link>
            ).
          </li>
        </ul>
      </section>

      {explanation && (
        <section className={card}>
          <h3 className="mb-2 font-semibold">Explanation</h3>
          <p className="font-medium">{explanation.summary}</p>
          <p className="mt-2 text-sm text-slate-700">{explanation.why}</p>
          <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-slate-600">
            {explanation.caveats.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
          <p className="mt-3 text-xs text-slate-500">
            {explanation.fallbackUsed
              ? 'Written from a fixed template, not by an LLM.'
              : `Worded by ${explanation.groqModel ?? 'an LLM'} from the result data only.`}
          </p>
        </section>
      )}

      {isVideo && ml && (
        <section className={card}>
          <h3 className="mb-3 font-semibold">Frame timeline</h3>
          <FrameTimeline analysisId={analysis.id} ml={ml} threshold={ours.threshold} />
        </section>
      )}

      <section className={card}>
        <h3 className="mb-3 font-semibold">{isVideo ? 'Most suspicious frame: faces' : 'Faces'}</h3>
        {preview && !isVideo && (
          <img src={preview} alt="Your upload" className="mb-4 max-h-56 rounded" />
        )}
        {faces.length === 0 && isVideo ? (
          <p className="text-sm text-slate-600">
            {mock
              ? 'Mock mode has no face boxes, scores per face or heatmaps; see the timeline for the frames.'
              : (ml?.framesScored ?? 0) === 0
                ? 'No face was detected in any sampled frame, so nothing was scored.'
                : 'No face details are available for this video.'}
          </p>
        ) : (
          <FaceGallery analysisId={analysis.id} faces={faces} mock={mock} />
        )}
      </section>

      <section className={card}>
        <h3 className="mb-3 font-semibold">Ask about this result</h3>
        <AskPanel analysisId={analysis.id} />
      </section>

      <div className="flex flex-wrap items-center gap-3">
        <button
          className="rounded border border-slate-300 bg-white px-4 py-2 text-sm hover:bg-slate-100"
          onClick={() => downloadJson(analysis)}
        >
          Download JSON
        </button>
        <PdfButton id={analysis.id} />
        {analysis.kind === 'image' && (
          <Link className="text-sm underline" to={`/filter-lab?analysis=${analysis.id}`}>
            Try beauty filters in the Filter Lab
          </Link>
        )}
      </div>
      <Disclaimer />
    </div>
  );
}
