import type { Progress, Stage } from '../api/types';

interface Step {
  stage: Stage;
  label: string;
  videoOnly?: boolean;
}

const STEPS: Step[] = [
  { stage: 'queued', label: 'Queued' },
  { stage: 'detecting_faces', label: 'Preprocessing' },
  { stage: 'extracting_frames', label: 'Extracting frames', videoOnly: true },
  { stage: 'scoring', label: 'Running detectors' },
  { stage: 'aggregating', label: 'Aggregating' },
  { stage: 'verifying', label: 'Second check' },
  { stage: 'explaining', label: 'Writing the report' },
  { stage: 'done', label: 'Done' },
];

interface Props {
  stage: Stage;
  progress?: Progress | null;
  kind?: 'image' | 'video';
  /** Upload progress, 0 to 1, while the file is being sent. */
  upload?: number | null;
}

/** The job's stages, the current one highlighted, with measured progress where known. */
export default function ProgressSteps({ stage, progress, kind, upload }: Props) {
  const steps = STEPS.filter((s) => !s.videoOnly || kind !== 'image');
  const current = steps.findIndex((s) => s.stage === stage);
  return (
    <div className="space-y-2">
      {upload != null && upload < 1 && (
        <Bar label="Uploading" fraction={upload} detail={`${Math.round(upload * 100)}%`} />
      )}
      <ol className="flex flex-wrap gap-2 text-sm" aria-label="Progress">
        {steps.map((step, i) => {
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
      {progress && progress.total > 0 && (
        <Bar
          label={steps[current]?.label ?? 'Working'}
          fraction={progress.done / progress.total}
          detail={`${progress.done} of ${progress.total} ${progress.unit}`}
        />
      )}
    </div>
  );
}

function Bar({ label, fraction, detail }: { label: string; fraction: number; detail: string }) {
  const value = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs text-slate-600">
        <span>{label}</span>
        <span className="tabular-nums">{detail}</span>
      </div>
      <div
        className="h-2 rounded-full bg-slate-200"
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={value}
      >
        <div className="h-2 rounded-full bg-slate-900" style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}
