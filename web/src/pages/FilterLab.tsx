import { useState, type FormEvent } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { api, ApiError } from '../api/client';
import type { Comparison, ComparisonSide } from '../api/types';
import AuthedImage from '../components/AuthedImage';
import { card, Disclaimer, Empty, ErrorState, Loading, PageHeader } from '../components/states';
import { pct, VERDICT_LABEL, VERDICT_STYLE } from '../components/verdict';
import { useAsync } from '../hooks/useAsync';

const FAMILIES: Array<[Comparison['family'], string]> = [
  ['F1', 'F1 · Skin smoothing'],
  ['F2', 'F2 · Makeup and tone'],
  ['F3', 'F3 · Face reshaping'],
  ['F4', 'F4 · Generative restoration (optional, test only)'],
];

export default function FilterLab() {
  const [params, setParams] = useSearchParams();
  const analysisId = params.get('analysis');
  const images = useAsync(() => api.listAnalyses(1, 50), []);
  const choices = images.data?.items.filter((a) => a.kind === 'image' && a.status === 'done') ?? [];

  return (
    <div className="space-y-4">
      <PageHeader title="Filter Lab">
        Apply one beauty filter to the face in an analysed image and compare the verdict before and
        after. A changed verdict is highlighted.
      </PageHeader>
      {images.status === 'loading' && <Loading />}
      {images.status === 'error' && <ErrorState message={images.error} onRetry={images.reload} />}
      {images.status === 'success' && choices.length === 0 && (
        <Empty title="No analysed images yet">
          <Link className="underline" to="/">
            Analyse an image
          </Link>{' '}
          first; the Filter Lab works on its stored upload.
        </Empty>
      )}
      {choices.length > 0 && (
        <label className="block text-sm">
          <span className="font-medium">Image</span>
          <select
            className="mt-1 block w-full rounded border border-slate-300 bg-white px-3 py-2 sm:w-96"
            value={analysisId ?? ''}
            onChange={(e) => setParams(e.target.value ? { analysis: e.target.value } : {})}
          >
            <option value="">Choose an analysed image…</option>
            {choices.map((a) => (
              <option key={a.id} value={a.id}>
                {new Date(a.createdAt).toLocaleString()} ·{' '}
                {a.final ? VERDICT_LABEL[a.final.verdict] : ''}
              </option>
            ))}
          </select>
        </label>
      )}
      {analysisId && <Lab analysisId={analysisId} />}
      <Disclaimer />
    </div>
  );
}

function Lab({ analysisId }: { analysisId: string }) {
  const [family, setFamily] = useState<Comparison['family']>('F1');
  const [strength, setStrength] = useState(0.6);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ message: string; unavailable: boolean } | null>(null);
  const history = useAsync(() => api.comparisons(analysisId), [analysisId]);
  const [latest, setLatest] = useState<Comparison | null>(null);

  async function run(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setLatest(await api.compare(analysisId, family, strength));
      history.reload();
    } catch (err) {
      setError({
        message: err instanceof Error ? err.message : 'The comparison failed',
        unavailable: err instanceof ApiError && err.status === 501,
      });
    } finally {
      setBusy(false);
    }
  }

  const shown = latest ?? history.data?.items[0] ?? null;

  return (
    <div className="space-y-4">
      <form onSubmit={run} className={`${card} flex flex-col gap-4 sm:flex-row sm:items-end`}>
        <label className="text-sm">
          <span className="font-medium">Filter</span>
          <select
            value={family}
            onChange={(e) => setFamily(e.target.value as Comparison['family'])}
            className="mt-1 block rounded border border-slate-300 bg-white px-3 py-2"
          >
            {FAMILIES.map(([f, label]) => (
              <option key={f} value={f}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex-1 text-sm">
          <span className="font-medium">Strength {Math.round(strength * 100)}%</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={strength}
            onChange={(e) => setStrength(Number(e.target.value))}
            className="mt-2 block w-full"
            aria-label="Filter strength"
          />
        </label>
        <button
          type="submit"
          disabled={busy}
          className="rounded bg-slate-900 px-4 py-2 text-sm text-white disabled:opacity-50"
        >
          {busy ? 'Comparing…' : 'Compare'}
        </button>
      </form>
      {error && (
        <ErrorState
          message={
            error.unavailable
              ? `The Filter Lab needs the real ML pipeline (model weights loaded, MOCK_MODE=false). ${error.message}`
              : error.message
          }
        />
      )}
      {shown && <ComparisonView analysisId={analysisId} c={shown} />}
      {history.status === 'success' && history.data.items.length > 1 && (
        <section className={card}>
          <h2 className="mb-2 font-semibold">Earlier comparisons</h2>
          <ul className="space-y-1 text-sm">
            {history.data.items.map((c) => (
              <li key={c.id}>
                {c.family} at {Math.round(c.strength * 100)}%: {c.clean.verdict ?? '—'} →{' '}
                {c.filtered.verdict ?? '—'}
                {c.flipped && (
                  <span className="ml-2 font-semibold text-rose-700">verdict changed</span>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function Side({
  analysisId,
  title,
  side,
}: {
  analysisId: string;
  title: string;
  side: ComparisonSide;
}) {
  return (
    <section className={card}>
      <h3 className="mb-2 font-semibold">{title}</h3>
      <p className={`text-lg font-bold ${side.verdict ? VERDICT_STYLE[side.verdict].text : ''}`}>
        {side.verdict ? VERDICT_LABEL[side.verdict] : 'No verdict'}
      </p>
      <p className="text-sm text-slate-600">
        Probability {side.probability === null ? 'n/a' : pct(side.probability)}
        {side.confident === false && ' · not confident'}
      </p>
      <div className="mt-3 grid grid-cols-2 gap-2">
        {side.image && (
          <AuthedImage
            analysisId={analysisId}
            imageKey={side.image}
            alt={`${title} face`}
            className="aspect-square w-full rounded"
          />
        )}
        {side.heatmap && (
          <AuthedImage
            analysisId={analysisId}
            imageKey={side.heatmap}
            alt={`${title} heatmap`}
            className="aspect-square w-full rounded"
          />
        )}
      </div>
    </section>
  );
}

function ComparisonView({ analysisId, c }: { analysisId: string; c: Comparison }) {
  return (
    <div className="space-y-3">
      {c.flipped ? (
        <p
          role="status"
          className="rounded border border-rose-300 bg-rose-50 p-3 text-sm font-semibold text-rose-800"
        >
          The verdict changed under this filter ({c.family} at {Math.round(c.strength * 100)}%).
        </p>
      ) : (
        <p className="rounded border border-emerald-300 bg-emerald-50 p-3 text-sm text-emerald-900">
          The verdict stayed the same under this filter ({c.family} at{' '}
          {Math.round(c.strength * 100)}%).
        </p>
      )}
      <div className="grid gap-4 md:grid-cols-2">
        <Side analysisId={analysisId} title="Original" side={c.clean} />
        <Side analysisId={analysisId} title="Filtered" side={c.filtered} />
      </div>
      {c.heatmapIoU !== null && (
        <p className="text-sm text-slate-600">
          Heatmap overlap (IoU of the top regions):{' '}
          <span className="tabular-nums">{c.heatmapIoU.toFixed(2)}</span>
        </p>
      )}
    </div>
  );
}
