import { useState } from 'react';
import type { Face } from '../api/types';
import AuthedImage from './AuthedImage';

type Detector = 'gend' | 'sbi';
type Mode = 'blend' | 'side';

/**
 * The face crop each detector saw, with its Grad-CAM heatmap: blended (with an opacity slider)
 * or side by side. Heatmaps exist only for explained faces in the real pipeline.
 */
export default function HeatmapViewer({ analysisId, face }: { analysisId: string; face: Face }) {
  const [detector, setDetector] = useState<Detector>('gend');
  const [mode, setMode] = useState<Mode>('blend');
  const [opacity, setOpacity] = useState(0.5);

  if (!face.heatmaps) {
    return (
      <p className="text-sm text-slate-600">
        No heatmap for this face. Heatmaps come from the real model pipeline (not mock mode) and are
        made for the deciding face only.
      </p>
    );
  }
  const heat = face.heatmaps[detector];
  const crop = face.crops?.[detector] ?? null;
  const detectors: Detector[] = face.heatmaps.sbi ? ['gend', 'sbi'] : ['gend'];
  const box = 'aspect-square w-full rounded';

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2 text-sm">
        {detectors.map((d) => (
          <button
            key={d}
            aria-pressed={detector === d}
            onClick={() => setDetector(d)}
            className={`rounded border px-3 py-1 ${detector === d ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-300 bg-white'}`}
          >
            {d === 'gend' ? 'GenD (CLIP)' : 'SBI (EfficientNet)'}
          </button>
        ))}
        <span className="mx-1 hidden w-px bg-slate-200 sm:block" />
        {(['blend', 'side'] as const).map((m) => (
          <button
            key={m}
            aria-pressed={mode === m}
            onClick={() => setMode(m)}
            disabled={!crop}
            className={`rounded border px-3 py-1 disabled:opacity-40 ${mode === m ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-300 bg-white'}`}
          >
            {m === 'blend' ? 'Blend' : 'Side by side'}
          </button>
        ))}
      </div>

      {!heat ? (
        <p className="text-sm text-slate-600">No heatmap from this detector.</p>
      ) : !crop ? (
        <AuthedImage
          analysisId={analysisId}
          imageKey={heat}
          alt="Heatmap"
          className={`${box} max-w-xs`}
        />
      ) : mode === 'side' ? (
        <div className="grid max-w-lg grid-cols-2 gap-3">
          <figure>
            <AuthedImage analysisId={analysisId} imageKey={crop} alt="Face crop" className={box} />
            <figcaption className="mt-1 text-xs text-slate-500">
              Face as the detector saw it
            </figcaption>
          </figure>
          <figure>
            <AuthedImage analysisId={analysisId} imageKey={heat} alt="Heatmap" className={box} />
            <figcaption className="mt-1 text-xs text-slate-500">
              Heatmap (red = most influence)
            </figcaption>
          </figure>
        </div>
      ) : (
        <div className="max-w-xs">
          <div className="relative">
            <AuthedImage analysisId={analysisId} imageKey={crop} alt="Face crop" className={box} />
            <AuthedImage
              analysisId={analysisId}
              imageKey={heat}
              alt="Heatmap overlay"
              className={`absolute inset-0 ${box}`}
              style={{ opacity }}
            />
          </div>
          <label className="mt-2 flex items-center gap-2 text-sm">
            <span className="text-slate-600">Heatmap opacity</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              aria-label="Heatmap opacity"
              className="flex-1"
            />
            <span className="w-10 text-right tabular-nums">{Math.round(opacity * 100)}%</span>
          </label>
        </div>
      )}
      {face.heatmapIoU !== null && (
        <p className="text-xs text-slate-500">
          Overlap with the most changed beautified copy (top-region IoU):{' '}
          <span className="tabular-nums">{face.heatmapIoU.toFixed(2)}</span>. Higher means the
          detector looked at the same regions despite the filter.
        </p>
      )}
      <p className="text-xs text-slate-500">
        A heatmap shows where a model looked, not proof of where an edit is.
      </p>
    </div>
  );
}
