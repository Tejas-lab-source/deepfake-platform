import { useState } from 'react';
import type { Frame, FrameRef, MlResult } from '../api/types';
import AuthedImage from './AuthedImage';

const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
const time = (t: number | null | undefined): string => (t == null ? '—' : `${t.toFixed(2)} s`);

/**
 * One column per sampled frame: height = the frame's head logit as a probability, the decision
 * threshold as a line, frames without a face drawn as gaps (they are excluded, never "real"),
 * the most and least suspicious frames marked, and links to the explained frames' heatmaps.
 */
export default function FrameTimeline({
  analysisId,
  ml,
  threshold,
}: {
  analysisId: string;
  ml: MlResult;
  threshold: number;
}) {
  const frames: Frame[] = ml.frames ?? [];
  const [selected, setSelected] = useState<number | null>(null);
  if (frames.length === 0) {
    return <p className="text-sm text-slate-600">No frame results are available for this video.</p>;
  }
  const most: FrameRef | null = ml.mostSuspicious ?? null;
  const least: FrameRef | null = ml.leastSuspicious ?? null;
  const chosen = frames.find((f) => f.index === selected) ?? null;
  const explained = frames.filter((f) => f.explained);

  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-600">
        {ml.framesSampled ?? frames.length} frames sampled, {ml.framesScored ?? 0} with a scored
        face
        {ml.video ? `, ${ml.video.durationSec.toFixed(1)} s of video` : ''}
        {ml.aggregate ? `; verdict from the top ${ml.aggregate.k} frames` : ''}.
      </p>
      <div className="relative overflow-x-auto">
        <div
          className="relative flex h-32 min-w-full items-end gap-px border-b border-slate-300"
          role="list"
          aria-label="Frame timeline"
        >
          <div
            className="pointer-events-none absolute inset-x-0 border-t border-dashed border-slate-900"
            style={{ bottom: `${threshold * 100}%` }}
            aria-hidden="true"
          />
          {frames.map((f) => {
            const scored = f.hasFace && f.logit !== null;
            // The frame's calibrated probability (same scale as the threshold) when known.
            const p = scored ? (f.probability ?? sigmoid(f.logit as number)) : 0;
            const suspicious = scored && p >= threshold;
            const isMost = most?.index === f.index;
            const isLeast = least?.index === f.index && !isMost;
            const label = scored
              ? `Frame ${f.index} at ${time(f.timestamp)}: score ${Math.round(p * 100)}%${suspicious ? ', above threshold' : ''}${isMost ? ', most suspicious' : ''}${isLeast ? ', least suspicious' : ''}`
              : `Frame ${f.index} at ${time(f.timestamp)}: ${f.decoded === false ? 'could not be decoded' : 'no face detected'} (excluded)`;
            return (
              <button
                key={f.index}
                role="listitem"
                aria-label={label}
                title={label}
                onClick={() => setSelected(f.index)}
                className={`relative h-full min-w-3 flex-1 ${selected === f.index ? 'bg-slate-100' : ''}`}
              >
                {scored ? (
                  <span
                    data-testid="frame-bar"
                    className={`absolute inset-x-0 bottom-0 rounded-t ${suspicious ? 'bg-rose-500' : 'bg-emerald-500'} ${isMost || isLeast ? 'ring-2 ring-slate-900' : ''}`}
                    style={{ height: `${Math.max(2, p * 100)}%` }}
                  />
                ) : (
                  <span
                    data-testid="frame-gap"
                    className="absolute inset-x-0 bottom-0 h-full bg-[repeating-linear-gradient(45deg,#e2e8f0_0_4px,transparent_4px_8px)]"
                  />
                )}
              </button>
            );
          })}
        </div>
      </div>
      <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">
        <li>
          <span className="mr-1 inline-block h-2 w-3 bg-rose-500" />
          at or above the threshold
        </li>
        <li>
          <span className="mr-1 inline-block h-2 w-3 bg-emerald-500" />
          below the threshold
        </li>
        <li>
          <span className="mr-1 inline-block h-2 w-3 bg-slate-200" />
          no face / not decoded (excluded)
        </li>
        <li>
          <span className="mr-1 inline-block w-3 border-t border-dashed border-slate-900" />
          decision threshold
        </li>
      </ul>
      <div className="grid gap-2 text-sm sm:grid-cols-2">
        {most && (
          <button
            className="rounded border border-slate-200 p-2 text-left hover:bg-slate-50"
            onClick={() => setSelected(most.index)}
          >
            Most suspicious: frame {most.index} at {time(most.timestamp)}
          </button>
        )}
        {least && least.index !== most?.index && (
          <button
            className="rounded border border-slate-200 p-2 text-left hover:bg-slate-50"
            onClick={() => setSelected(least.index)}
          >
            Least suspicious: frame {least.index} at {time(least.timestamp)}
          </button>
        )}
      </div>
      {explained.length > 0 && (
        <p className="text-xs text-slate-500">
          Heatmaps were made for frames {explained.map((f) => f.index).join(' and ')}.
        </p>
      )}
      {chosen && (
        <div className="rounded border border-slate-200 p-3 text-sm" aria-live="polite">
          <p className="font-medium">
            Frame {chosen.index} at {time(chosen.timestamp)}
          </p>
          {chosen.hasFace && chosen.logit !== null ? (
            <p className="text-slate-600">
              Head logit <span className="tabular-nums">{chosen.logit.toFixed(2)}</span>
              {chosen.probability != null && (
                <>
                  , face probability{' '}
                  <span className="tabular-nums">{Math.round(chosen.probability * 100)}%</span>
                </>
              )}
              .
            </p>
          ) : (
            <p className="text-slate-600">
              No face was detected here, so this frame does not count towards the verdict.
            </p>
          )}
          {chosen.faces?.[0]?.heatmaps?.gend && (
            <div className="mt-2 flex gap-3">
              {chosen.faces[0].crops?.gend && (
                <AuthedImage
                  analysisId={analysisId}
                  imageKey={chosen.faces[0].crops.gend}
                  alt={`Frame ${chosen.index} face`}
                  className="h-28 w-28 rounded"
                />
              )}
              <AuthedImage
                analysisId={analysisId}
                imageKey={chosen.faces[0].heatmaps.gend}
                alt={`Frame ${chosen.index} heatmap`}
                className="h-28 w-28 rounded"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
