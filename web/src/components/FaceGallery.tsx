import { useState } from 'react';
import type { Face } from '../api/types';
import AuthedImage from './AuthedImage';
import HeatmapViewer from './HeatmapViewer';
import { pct, VERDICT_LABEL } from './verdict';

/** Every detected face, the deciding one first, with its scores and heatmaps. */
export default function FaceGallery({
  analysisId,
  faces,
  mock,
}: {
  analysisId: string;
  faces: Face[];
  mock: boolean;
}) {
  const [selected, setSelected] = useState(0);
  const face = faces[selected];
  if (!face) return <p className="text-sm text-slate-600">No face was detected.</p>;

  return (
    <div className="space-y-4">
      {faces.length > 1 && (
        <div className="flex flex-wrap gap-2" role="tablist" aria-label="Faces">
          {faces.map((f, i) => (
            <button
              key={f.box.join(',')}
              role="tab"
              aria-selected={i === selected}
              onClick={() => setSelected(i)}
              className={`flex items-center gap-2 rounded border px-2 py-1 text-sm ${i === selected ? 'border-slate-900' : 'border-slate-300'}`}
            >
              {f.crops?.gend ? (
                <AuthedImage
                  analysisId={analysisId}
                  imageKey={f.crops.gend}
                  alt={`Face ${i + 1}`}
                  className="h-10 w-10 rounded"
                />
              ) : null}
              Face {i + 1}
              {i === 0 && <span className="text-xs text-slate-500">(deciding)</span>}
            </button>
          ))}
        </div>
      )}
      <dl className="grid grid-cols-2 gap-y-1 text-sm sm:grid-cols-4">
        {face.verdict && (
          <>
            <dt className="text-slate-500">Face verdict</dt>
            <dd>{VERDICT_LABEL[face.verdict]}</dd>
          </>
        )}
        {face.probability !== undefined && (
          <>
            <dt className="text-slate-500">Face probability</dt>
            <dd className="tabular-nums">{pct(face.probability)}</dd>
          </>
        )}
        <dt className="text-slate-500">GenD head logit</dt>
        <dd className="tabular-nums">{face.scores.gendHead.toFixed(2)}</dd>
        <dt className="text-slate-500">SBI logit</dt>
        <dd className="tabular-nums">
          {face.scores.sbi === null ? 'n/a' : face.scores.sbi.toFixed(2)}
        </dd>
        <dt className="text-slate-500">Box (x, y, w, h)</dt>
        <dd className="tabular-nums">{face.box.join(', ')}</dd>
        {face.beautifiedLikelihood !== null && (
          <>
            <dt className="text-slate-500">Smoothing cue</dt>
            <dd className="tabular-nums">{face.beautifiedLikelihood.toFixed(2)}</dd>
          </>
        )}
      </dl>
      <div>
        <h4 className="mb-2 text-sm font-semibold">Score per beautified copy</h4>
        <ul className="flex flex-wrap gap-2 text-sm">
          {Object.entries(face.variantLogits).map(([name, logit]) => (
            <li key={name} className="rounded border border-slate-200 px-2 py-1">
              {name} <span className="tabular-nums">{logit.toFixed(2)}</span>
            </li>
          ))}
        </ul>
        <p className="mt-1 text-xs text-slate-500">
          Logits above 0 lean manipulated, below 0 lean not; spread{' '}
          <span className="tabular-nums">{face.spread.toFixed(2)}</span>.
          {mock && ' Mock mode: placeholder numbers, and the box is a centred placeholder.'}
        </p>
      </div>
      <HeatmapViewer analysisId={analysisId} face={face} />
    </div>
  );
}
