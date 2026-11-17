import { api } from '../api/client';
import { card, ErrorState, Loading, PageHeader } from '../components/states';
import { useAsync } from '../hooks/useAsync';

const fmt = (v: number | null): string =>
  v === null ? 'n/a' : String(Math.round(v * 1000) / 1000);

function sourceText(v: unknown): string {
  if (typeof v === 'string') return v;
  if (v && typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

export default function ModelCard() {
  const mc = useAsync(() => api.modelCard(), []);
  return (
    <div className="space-y-4">
      <PageHeader title="Model card">
        What decides the verdict, the values it uses, and where each value came from.
      </PageHeader>
      {mc.status === 'loading' && <Loading />}
      {mc.status === 'error' && <ErrorState message={mc.error} onRetry={mc.reload} />}
      {mc.status === 'success' && (
        <>
          {(mc.data.mockMode || mc.data.calibration.fixture) && (
            <p className="rounded border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
              {mc.data.mockMode
                ? 'Mock mode: no models are loaded and the scores are placeholders.'
                : 'This model card was fitted on a fixture: its numbers are not meaningful.'}
            </p>
          )}
          <div className="grid gap-4 md:grid-cols-2">
            <section className={card}>
              <h2 className="mb-2 font-semibold">Detectors (pretrained, frozen)</h2>
              <dl className="grid grid-cols-2 gap-y-1 text-sm">
                <dt className="text-slate-500">Primary</dt>
                <dd>{mc.data.detectors.primary}</dd>
                <dt className="text-slate-500">Secondary</dt>
                <dd>{mc.data.detectors.secondary}</dd>
                <dt className="text-slate-500">SBI loaded</dt>
                <dd>
                  {mc.data.detectors.sbiLoaded == null
                    ? 'n/a'
                    : mc.data.detectors.sbiLoaded
                      ? 'yes'
                      : 'no'}
                </dd>
                <dt className="text-slate-500">Invariance head</dt>
                <dd>
                  {mc.data.detectors.headLoaded == null
                    ? 'n/a'
                    : mc.data.detectors.headLoaded
                      ? 'loaded'
                      : 'not loaded (GenD logit)'}
                </dd>
                <dt className="text-slate-500">Device</dt>
                <dd>{mc.data.detectors.device ?? 'n/a'}</dd>
              </dl>
            </section>
            <section className={card}>
              <h2 className="mb-2 font-semibold">Pipeline</h2>
              <dl className="grid grid-cols-2 gap-y-1 text-sm">
                <dt className="text-slate-500">Beautified copies per image</dt>
                <dd>{mc.data.pipeline.beautifiedCopiesPerImage ?? 'n/a'}</dd>
                <dt className="text-slate-500">Copies per video frame</dt>
                <dd>{mc.data.pipeline.beautifiedCopiesPerVideoFrame}</dd>
                <dt className="text-slate-500">Max video frames</dt>
                <dd>{mc.data.pipeline.maxVideoFrames ?? 'n/a'}</dd>
                <dt className="text-slate-500">Video top-k</dt>
                <dd>{mc.data.pipeline.videoTopK ?? 'n/a'}</dd>
              </dl>
            </section>
          </div>
          <section className={card}>
            <h2 className="mb-2 font-semibold">
              Calibration ({mc.data.calibration.version ?? 'unversioned'})
            </h2>
            <p className="mb-2 text-sm text-slate-600">
              Status: {mc.data.calibration.status ?? 'unknown'}
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-slate-500">
                  <tr>
                    <th className="font-normal">Value</th>
                    <th className="font-normal">Setting</th>
                  </tr>
                </thead>
                <tbody>
                  {(
                    [
                      ['Threshold', mc.data.calibration.threshold],
                      ['Temperature', mc.data.calibration.temperature],
                      ['Uncertain margin', mc.data.calibration.uncertainMargin],
                      ['Confident margin', mc.data.calibration.confidentMargin],
                      ['Spread limit', mc.data.calibration.spreadLimit],
                      ['Detector disagreement limit', mc.data.calibration.detectorDisagreeLimit],
                    ] as const
                  ).map(([label, v]) => (
                    <tr key={label} className="border-t border-slate-100">
                      <td className="py-1">{label}</td>
                      <td className="py-1 tabular-nums">{fmt(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <h3 className="mt-3 text-sm font-semibold">Where the values came from</h3>
            {Object.keys(mc.data.calibration.sources).length === 0 ? (
              <p className="text-sm text-slate-600">No sources recorded (built-in defaults).</p>
            ) : (
              <ul className="mt-1 space-y-1 text-sm">
                {Object.entries(mc.data.calibration.sources).map(([k, v]) => (
                  <li key={k}>
                    <span className="font-medium">{k}:</span> {sourceText(v)}
                  </li>
                ))}
              </ul>
            )}
          </section>
          <div className="grid gap-4 md:grid-cols-2">
            <section className={card}>
              <h2 className="mb-2 font-semibold">Second check</h2>
              <p className="text-sm">
                {mc.data.secondCheck.provider} (mode: {mc.data.secondCheck.mode}), FAKE at a score
                of {fmt(mc.data.secondCheck.threshold)} or more.
              </p>
              <p className="mt-1 text-sm text-slate-600">{mc.data.secondCheck.rule}</p>
            </section>
            <section className={card}>
              <h2 className="mb-2 font-semibold">Explanations</h2>
              <p className="text-sm">
                {mc.data.explanations.provider} (mode: {mc.data.explanations.mode})
              </p>
              <p className="mt-1 text-sm text-slate-600">{mc.data.explanations.rule}</p>
            </section>
          </div>
          <section className={card}>
            <h2 className="mb-2 font-semibold">Licences</h2>
            <ul className="space-y-1 text-sm">
              {mc.data.licences.map((l) => (
                <li key={l.component}>
                  {l.component}: <span className="font-medium">{l.licence}</span>
                </li>
              ))}
            </ul>
            <p className="mt-2 text-xs text-slate-500">An academic, non-commercial project.</p>
          </section>
          <p className="text-sm text-slate-600">{mc.data.notice}</p>
        </>
      )}
    </div>
  );
}
