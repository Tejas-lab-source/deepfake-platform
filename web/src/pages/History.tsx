import { useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import { Empty, ErrorState, Loading, PageHeader } from '../components/states';
import { pct, VERDICT_LABEL, VERDICT_STYLE } from '../components/verdict';
import { useAsync } from '../hooks/useAsync';

const PAGE_SIZE = 10;

const STATUS_TEXT = { queued: 'Queued', processing: 'Processing', done: 'Done', failed: 'Failed' };

export default function History() {
  const [page, setPage] = useState(1);
  const list = useAsync(() => api.listAnalyses(page, PAGE_SIZE), [page]);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  async function remove(id: string) {
    if (
      !window.confirm(
        'Delete this analysis, its upload and all its results? This cannot be undone.',
      )
    ) {
      return;
    }
    setDeleting(id);
    setDeleteError(null);
    try {
      await api.deleteAnalysis(id);
      // Step back if that was the last item on a later page.
      if (list.data && list.data.items.length === 1 && page > 1) setPage(page - 1);
      else list.reload();
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : 'Could not delete');
    } finally {
      setDeleting(null);
    }
  }

  const pages = list.data ? Math.max(1, Math.ceil(list.data.total / PAGE_SIZE)) : 1;

  return (
    <div className="space-y-4">
      <PageHeader title="History">
        Your past analyses. Uploads expire; results stay until you delete them.
      </PageHeader>
      {deleteError && <ErrorState message={deleteError} />}
      {list.status === 'loading' && <Loading />}
      {list.status === 'error' && <ErrorState message={list.error} onRetry={list.reload} />}
      {list.status === 'success' && list.data.total === 0 && (
        <Empty title="No analyses yet">
          <Link className="underline" to="/">
            Upload an image or video
          </Link>{' '}
          to get started.
        </Empty>
      )}
      {list.status === 'success' && list.data.items.length > 0 && (
        <>
          <ul className="divide-y divide-slate-200 rounded-lg border border-slate-200 bg-white">
            {list.data.items.map((a) => (
              <li
                key={a.id}
                className="flex flex-col gap-2 p-4 sm:flex-row sm:items-center sm:justify-between"
              >
                <div>
                  <Link to={`/analyses/${a.id}`} className="font-medium underline">
                    {a.kind === 'video' ? 'Video' : 'Image'} ·{' '}
                    {new Date(a.createdAt).toLocaleString()}
                  </Link>
                  <p className="text-sm">
                    {a.final ? (
                      <span className={VERDICT_STYLE[a.final.verdict].text}>
                        {VERDICT_LABEL[a.final.verdict]}
                        {a.ours?.probability != null && ` · ${pct(a.ours.probability)}`}
                      </span>
                    ) : (
                      <span className="text-slate-500">{STATUS_TEXT[a.status]}</span>
                    )}
                  </p>
                </div>
                <button
                  onClick={() => void remove(a.id)}
                  disabled={deleting === a.id}
                  className="self-start rounded border border-rose-300 px-3 py-1 text-sm text-rose-700 hover:bg-rose-50 disabled:opacity-50 sm:self-auto"
                >
                  {deleting === a.id ? 'Deleting…' : 'Delete'}
                </button>
              </li>
            ))}
          </ul>
          <nav className="flex items-center justify-between text-sm" aria-label="Pages">
            <button
              disabled={page <= 1}
              onClick={() => setPage(page - 1)}
              className="rounded border border-slate-300 px-3 py-1 disabled:opacity-40"
            >
              Previous
            </button>
            <span>
              Page {page} of {pages} · {list.data.total} total
            </span>
            <button
              disabled={page >= pages}
              onClick={() => setPage(page + 1)}
              className="rounded border border-slate-300 px-3 py-1 disabled:opacity-40"
            >
              Next
            </button>
          </nav>
        </>
      )}
    </div>
  );
}
