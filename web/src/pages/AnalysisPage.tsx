import { Link, useParams } from 'react-router-dom';
import ProgressSteps from '../components/ProgressSteps';
import ResultView from '../components/ResultView';
import { ErrorState, Loading } from '../components/states';
import { CONNECTION_TEXT, useAnalysis } from '../hooks/useAnalysis';

/** One analysis by id: live progress while it runs, then the full result. */
export default function AnalysisPage() {
  const { id = '' } = useParams();
  const watch = useAnalysis(id);
  const a = watch.analysis;

  return (
    <div className="space-y-4">
      <p className="text-sm">
        <Link className="underline" to="/history">
          ← History
        </Link>
      </p>
      {watch.error && <ErrorState message={watch.error} />}
      {!a && !watch.error && watch.stage === null && <Loading />}
      {!a && !watch.error && watch.stage && (
        <>
          <ProgressSteps stage={watch.stage} progress={watch.progress} />
          {CONNECTION_TEXT[watch.connection] && (
            <p className="text-sm text-amber-800">{CONNECTION_TEXT[watch.connection]}</p>
          )}
        </>
      )}
      {a?.status === 'done' && <ResultView analysis={a} />}
    </div>
  );
}
