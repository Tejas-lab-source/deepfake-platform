import type { Verdict } from '../api/types';
import { pct, VERDICT_STYLE } from './verdict';

interface Props {
  label: string;
  value: number;
  threshold: number;
  verdict: Verdict;
}

/** A 0-100% bar with the decision threshold marked. */
export default function ConfidenceGauge({ label, value, threshold, verdict }: Props) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-sm">
        <span className="text-slate-600">{label}</span>
        <span className="font-medium tabular-nums">{pct(value)}</span>
      </div>
      <div
        className="relative h-3 rounded-full bg-slate-200"
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(value * 100)}
      >
        <div
          className={`h-3 rounded-full ${VERDICT_STYLE[verdict].bar}`}
          style={{ width: `${Math.max(2, value * 100)}%` }}
        />
        <div
          className="absolute -top-1 h-5 w-0.5 bg-slate-900"
          style={{ left: `${threshold * 100}%` }}
        />
      </div>
      <div className="mt-1 text-xs text-slate-500">Threshold {pct(threshold)} (black mark)</div>
    </div>
  );
}
