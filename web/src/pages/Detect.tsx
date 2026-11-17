import { useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import ProgressSteps from '../components/ProgressSteps';
import ResultView from '../components/ResultView';
import { Disclaimer, ErrorState, PageHeader } from '../components/states';
import UploadDropzone from '../components/UploadDropzone';
import { CONNECTION_TEXT, useAnalysis } from '../hooks/useAnalysis';

export default function Detect() {
  const [externalCheck, setExternalCheck] = useState(false);
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [kind, setKind] = useState<'image' | 'video'>('image');
  const [upload, setUpload] = useState<number | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const watch = useAnalysis(analysisId);

  const uploading = upload !== null;
  const busy =
    uploading ||
    (analysisId !== null && watch.stage !== 'done' && watch.stage !== 'failed' && !watch.error);

  async function start(file: File) {
    setUploadError(null);
    setAnalysisId(null);
    setKind(file.type.startsWith('video/') ? 'video' : 'image');
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return file.type.startsWith('image/') ? URL.createObjectURL(file) : null;
    });
    setUpload(0);
    try {
      const created = await api.createAnalysis(file, externalCheck, setUpload);
      setAnalysisId(created.analysisId);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUpload(null);
    }
  }

  const notice = analysisId && !watch.analysis ? CONNECTION_TEXT[watch.connection] : null;

  return (
    <div className="space-y-6">
      <PageHeader title="Check a face for manipulation">
        Upload a face image or a short video. The result is an automated check, not proof.
      </PageHeader>

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

      {(uploading ||
        (analysisId && watch.stage && watch.stage !== 'failed' && !watch.analysis)) && (
        <ProgressSteps
          stage={watch.stage ?? 'queued'}
          progress={watch.progress}
          kind={kind}
          upload={upload}
        />
      )}
      {notice && <p className="text-sm text-amber-800">{notice}</p>}
      {uploadError && <ErrorState message={uploadError} />}
      {watch.error && <ErrorState message={watch.error} />}
      {watch.analysis?.status === 'done' && (
        <>
          <p className="text-sm">
            <Link className="underline" to={`/analyses/${watch.analysis.id}`}>
              Permanent link to this result
            </Link>{' '}
            (also in your History).
          </p>
          <ResultView analysis={watch.analysis} preview={preview} />
        </>
      )}
      {!analysisId && !uploading && <Disclaimer />}
    </div>
  );
}
