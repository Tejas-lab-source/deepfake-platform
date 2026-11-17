import { api } from '../api/client';
import type { EvalReport } from '../api/types';
import { card, Empty, ErrorState, Loading, PageHeader } from '../components/states';
import { useAsync } from '../hooks/useAsync';

const METRICS: Array<[string, string, 'higher' | 'lower']> = [
  ['clean_auc', 'Clean AUC', 'higher'],
  ['fpr_beautified_real', 'FPR, beautified real', 'lower'],
  ['fnr_beautified_fake', 'FNR, beautified fake', 'lower'],
  ['flip_rate', 'Flip rate', 'lower'],
  ['uncertain_rate', 'UNCERTAIN rate', 'lower'],
  ['heatmap_iou', 'Heatmap IoU', 'higher'],
];

const fmt = (v: unknown): string => (typeof v === 'number' ? v.toFixed(3) : '—');

function meanOf(entry: unknown): unknown {
  return entry && typeof entry === 'object' ? (entry as { mean?: unknown }).mean : undefined;
}

/** A small bar per scheme for one metric (the per-scheme aggregate over held-out folds). */
function MetricBars({
  report,
  metric,
  label,
}: {
  report: EvalReport;
  metric: string;
  label: string;
}) {
  const rows = Object.entries(report.aggregate)
    .map(([scheme, m]) => [scheme, meanOf(m[metric])] as const)
    .filter((r): r is readonly [string, number] => typeof r[1] === 'number');
  if (rows.length === 0) return null;
  const max = Math.max(...rows.map((r) => Math.abs(r[1])), 1e-9);
  return (
    <figure className={card}>
      <figcaption className="mb-2 text-sm font-semibold">
        {label} (mean over held-out folds)
      </figcaption>
      <ul className="space-y-1">
        {rows.map(([scheme, v]) => (
          <li key={scheme} className="flex items-center gap-2 text-sm">
            <span className="w-10 shrink-0">{scheme}</span>
            <span
              className="h-3 rounded bg-slate-700"
              style={{ width: `${(Math.abs(v) / max) * 70}%` }}
            />
            <span className="tabular-nums">{v.toFixed(3)}</span>
          </li>
        ))}
      </ul>
    </figure>
  );
}

export default function Results() {
  const res = useAsync(() => api.evalRuns(), []);
  return (
    <div className="space-y-4">
      <PageHeader title="Evaluation results">
        Leave-one-family-out evaluation of the six head schemes (H-a to H-f), as imported from the
        offline evaluation run.
      </PageHeader>
      {res.status === 'loading' && <Loading />}
      {res.status === 'error' && <ErrorState message={res.error} onRetry={res.reload} />}
      {res.status === 'success' && !res.data.report && (
        <Empty title="No evaluation results yet">
          Run <code>offline/eval_lofo.py</code> on the dataset, then an admin imports its JSON with{' '}
          <code>POST /api/v1/eval-runs</code>. Nothing is shown until real results exist.
        </Empty>
      )}
      {res.status === 'success' && res.data.report && <Report report={res.data.report} />}
    </div>
  );
}

function Report({ report }: { report: EvalReport }) {
  const metricsPresent = METRICS.filter(([k]) => report.runs.some((r) => k in r.metrics));
  return (
    <>
      {report.fixture && (
        <p
          role="alert"
          className="rounded border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900"
        >
          {report.notice ?? 'Fixture run'}: these numbers test the plumbing and are not accuracy
          results.
        </p>
      )}
      <p className="text-sm text-slate-600">
        Created {report.created ?? 'unknown'} · {report.runs.length} runs (scheme × fold × seed)
      </p>
      <div className="grid gap-4 md:grid-cols-2">
        {metricsPresent.slice(0, 4).map(([k, label]) => (
          <MetricBars key={k} report={report} metric={k} label={label} />
        ))}
      </div>
      <section className={card}>
        <h2 className="mb-2 font-semibold">All runs</h2>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead className="text-left text-slate-500">
              <tr>
                <th className="font-normal">Scheme</th>
                <th className="font-normal">Held out</th>
                <th className="font-normal">Seed</th>
                {metricsPresent.map(([k, label, dir]) => (
                  <th key={k} className="font-normal" title={`${dir} is better`}>
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {report.runs.map((r) => (
                <tr key={`${r.scheme}-${r.fold}-${r.seed}`} className="border-t border-slate-100">
                  <td className="py-1">{r.scheme}</td>
                  <td>
                    {r.fold}
                    {r.heldOut ? '' : ' (not held out)'}
                  </td>
                  <td>{r.seed}</td>
                  {metricsPresent.map(([k]) => {
                    const ci = r.ci95[k];
                    return (
                      <td
                        key={k}
                        className="tabular-nums"
                        title={ci ? `95% CI ${ci[0].toFixed(3)}–${ci[1].toFixed(3)}` : undefined}
                      >
                        {fmt(r.metrics[k])}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-slate-500">
          Hover a value for its bootstrap 95% CI where available.
        </p>
      </section>
      {report.successCriteria.length > 0 && (
        <section className={card}>
          <h2 className="mb-2 font-semibold">Pre-registered success criteria</h2>
          <ul className="space-y-1 text-sm">
            {report.successCriteria.map((c, i) => (
              <li key={i}>
                {String(c.name ?? c.criterion ?? `Criterion ${i + 1}`)}:{' '}
                <span className="font-medium">
                  {c.met === true ? 'met' : c.met === false ? 'not met' : 'not evaluated'}
                </span>
                {c.fixture_only ? (
                  <span className="text-amber-800"> ({String(c.fixture_only)})</span>
                ) : null}
              </li>
            ))}
          </ul>
        </section>
      )}
    </>
  );
}
