import { useEffect, useState } from 'react';
import { api } from '../api/client';
import { watchAnalysis } from '../api/socket';
import type { Analysis, Stage } from '../api/types';
import ProgressSteps from '../components/ProgressSteps';
import ResultView from '../components/ResultView';
import UploadDropzone from '../components/UploadDropzone';

const POLL_MS = 1500;

export default function Detect() {
  const [externalCheck, setExternalCheck] = useState(false);
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [stage, setStage] = useState<Stage | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const busy = stage !== null && stage !== 'done' && stage !== 'failed';

  async function start(file: File) {
    setError(null);
    setAnalysis(null);
    setStage('queued');
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return file.type.startsWith('image/') ? URL.createObjectURL(file) : null;
    });
    try {
      const created = await api.createAnalysis(file, externalCheck);
      setAnalysisId(created.analysisId);
    } catch (err) {
      setStage(null);
      setError(err instanceof Error ? err.message : 'Upload failed');
    }
  }

  // Follow the job: Socket.IO for live stages, polling as the fallback and for the result.
  useEffect(() => {
    if (!analysisId) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const load = async () => {
      try {
        const a = await api.getAnalysis(analysisId);
        if (stopped) return;
        setStage(a.stage);
        if (a.status === 'done') return setAnalysis(a);
        if (a.status === 'failed') return setError(a.error ?? 'The analysis failed');
      } catch (err) {
        if (!stopped) setError(err instanceof Error ? err.message : 'Could not load the result');
        return;
      }
      timer = setTimeout(load, POLL_MS);
    };

    const stopWatching = watchAnalysis(analysisId, (e) => {
      setStage(e.stage);
      if (e.stage === 'done' || e.stage === 'failed') {
        clearTimeout(timer);
        void load();
      }
    });
    void load();
    return () => {
      stopped = true;
      clearTimeout(timer);
      stopWatching();
    };
  }, [analysisId]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Check a face for manipulation</h1>
        <p className="mt-1 text-sm text-slate-600">
          Upload a face image or a short video. The result is an automated check, not proof.
        </p>
      </div>

      <UploadDropzone disabled={busy} onFile={start} />

      <label className="flex items-start gap-3 text-sm">
        <input
          type="checkbox"
          className="mt-1"
          checked={externalCheck}
          disabled={busy}
          onChange={(e) => setExternalCheck(e.target.checked)}
        />
        <span>
          <span className="font-medium">Get a second opinion from Hive</span>
          <span className="block text-slate-500">
            Sends this file to Hive, a third-party service. Leave it off to keep the file on this
            server.
          </span>
        </span>
      </label>

      {stage && stage !== 'failed' && <ProgressSteps stage={stage} />}
      {error && (
        <p
          role="alert"
          className="rounded border border-rose-300 bg-rose-50 p-3 text-sm text-rose-800"
        >
          {error}
        </p>
      )}
      {analysis && <ResultView analysis={analysis} preview={preview} />}
    </div>
  );
}
