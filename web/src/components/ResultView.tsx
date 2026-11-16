import type { Analysis } from '../api/types';
import ConfidenceGauge from './ConfidenceGauge';
import { BADGE_LABEL, pct, VERDICT_LABEL, VERDICT_STYLE } from './verdict';

const UNCERTAIN_REASON: Record<string, string> = {
  near_threshold: 'Score too close to the threshold',
  unstable_under_beautification: 'Score unstable under beauty filters',
  variants_disagree: 'Filtered copies disagree',
  detectors_disagree: 'The two detectors disagree',
  no_face_detected: 'No face detected',
};

const card = 'rounded-lg border border-slate-200 bg-white p-5';

function downloadJson(analysis: Analysis) {
  const blob = new Blob([JSON.stringify(analysis, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `analysis-${analysis.id}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export default function ResultView({
  analysis,
  preview,
}: {
  analysis: Analysis;
  preview: string | null;
}) {
  const { final, ours, external, explanation, ml } = analysis;
  if (!final || !ours) return null;
  const style = VERDICT_STYLE[final.verdict];
  const face = ml?.faces?.[0];

  return (
    <div className="space-y-4">
      <section className={`rounded-lg border p-5 ${style.panel}`}>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm text-slate-600">Final verdict</p>
            <h2 className={`text-3xl font-bold ${style.text}`}>{VERDICT_LABEL[final.verdict]}</h2>
          </div>
          <span className="rounded-full border border-slate-300 bg-white px-3 py-1 text-sm font-medium">
            {BADGE_LABEL[final.badge]}
          </span>
        </div>
        <p className="mt-3 text-sm text-slate-700">
          This is an automated check. It is not proof either way.
        </p>
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
                ? `Not available for this upload (${external?.status ?? 'error'}).`
                : 'Switched off for this upload.'}
            </p>
          )}
        </section>
      </div>

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
          {explanation.fallbackUsed && (
            <p className="mt-3 text-xs text-slate-500">
              Written from a fixed template, not by an LLM.
            </p>
          )}
        </section>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        {preview && (
          <section className={card}>
            <h3 className="mb-3 font-semibold">Upload</h3>
            <img src={preview} alt="Uploaded file" className="max-h-72 rounded" />
          </section>
        )}
        {face && (
          <section className={card}>
            <h3 className="mb-3 font-semibold">Score per filtered copy</h3>
            <table className="w-full text-sm">
              <thead className="text-left text-slate-500">
                <tr>
                  <th className="font-normal">Copy</th>
                  <th className="text-right font-normal">Logit</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(face.variantLogits).map(([name, logit]) => (
                  <tr key={name} className="border-t border-slate-100">
                    <td className="py-1">{name}</td>
                    <td className="py-1 text-right tabular-nums">{logit.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-3 text-xs text-slate-500">
              Above 0 leans manipulated, below 0 leans not. Heatmaps are not produced in mock mode.
            </p>
          </section>
        )}
      </div>

      <button
        className="rounded border border-slate-300 bg-white px-4 py-2 text-sm hover:bg-slate-100"
        onClick={() => downloadJson(analysis)}
      >
        Download JSON
      </button>
    </div>
  );
}
