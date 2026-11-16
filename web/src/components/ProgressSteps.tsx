import type { Stage } from '../api/types';

const STEPS: Array<{ stage: Stage; label: string }> = [
  { stage: 'queued', label: 'Queued' },
  { stage: 'detecting_faces', label: 'Detecting faces' },
  { stage: 'scoring', label: 'Scoring' },
  { stage: 'verifying', label: 'Verifying' },
  { stage: 'explaining', label: 'Explaining' },
  { stage: 'done', label: 'Done' },
];

export default function ProgressSteps({ stage }: { stage: Stage }) {
  const current = STEPS.findIndex((s) => s.stage === stage);
  return (
    <ol className="flex flex-wrap gap-2 text-sm" aria-label="Progress">
      {STEPS.map((step, i) => {
        const state = i < current ? 'past' : i === current ? 'now' : 'next';
        const style = {
          past: 'bg-slate-200 text-slate-600',
          now: 'bg-slate-900 text-white',
          next: 'border border-slate-200 text-slate-400',
        }[state];
        return (
          <li
            key={step.stage}
            aria-current={state === 'now' ? 'step' : undefined}
            className={`rounded-full px-3 py-1 ${style}`}
          >
            {step.label}
          </li>
        );
      })}
    </ol>
  );
}
