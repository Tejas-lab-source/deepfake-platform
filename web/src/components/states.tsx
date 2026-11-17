import type { ReactNode } from 'react';

export const card = 'rounded-lg border border-slate-200 bg-white p-5';

export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <p role="status" className="flex items-center gap-2 text-sm text-slate-600">
      <span className="h-3 w-3 animate-spin rounded-full border-2 border-slate-400 border-t-transparent" />
      {label}
    </p>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div
      role="alert"
      className="rounded border border-rose-300 bg-rose-50 p-3 text-sm text-rose-800"
    >
      <p>{message}</p>
      {onRetry && (
        <button
          className="mt-2 rounded border border-rose-300 bg-white px-3 py-1 text-rose-800 hover:bg-rose-100"
          onClick={onRetry}
        >
          Try again
        </button>
      )}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 bg-white p-6 text-center">
      <p className="font-medium">{title}</p>
      {children && <div className="mt-1 text-sm text-slate-600">{children}</div>}
    </div>
  );
}

/** Shown on every result page. */
export function Disclaimer() {
  return (
    <p className="rounded border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600">
      Automated, probabilistic finding, not proof. Deepfake detectors can miss manipulations and can
      flag unedited media.
    </p>
  );
}

export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div>
      <h1 className="text-2xl font-semibold">{title}</h1>
      {children && <p className="mt-1 text-sm text-slate-600">{children}</p>}
    </div>
  );
}
