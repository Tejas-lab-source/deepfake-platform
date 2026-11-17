import { useEffect, useState } from 'react';
import { api } from '../api/client';
import { watchAnalysis, type Connection } from '../api/socket';
import type { Analysis, Progress, Stage } from '../api/types';

const POLL_MS = 2000;

export interface AnalysisWatch {
  analysis: Analysis | null;
  stage: Stage | null;
  progress: Progress | null;
  error: string | null;
  connection: Connection;
}

/**
 * Follow one analysis until it is done or failed: Socket.IO for live stages and measured
 * progress, polling as the fallback (and to fetch the result). Works for a page opened long
 * after the upload too: the first poll and the socket snapshot give the current state.
 */
export function useAnalysis(analysisId: string | null): AnalysisWatch {
  const [state, setState] = useState<AnalysisWatch>({
    analysis: null,
    stage: null,
    progress: null,
    error: null,
    connection: 'connecting',
  });

  useEffect(() => {
    if (!analysisId) return;
    setState({
      analysis: null,
      stage: 'queued',
      progress: null,
      error: null,
      connection: 'connecting',
    });
    let stopped = false;
    let finished = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const load = async () => {
      clearTimeout(timer);
      try {
        const a = await api.getAnalysis(analysisId);
        if (stopped) return;
        if (a.status === 'done' || a.status === 'failed') {
          finished = true;
          setState((s) => ({
            ...s,
            analysis: a,
            stage: a.stage,
            progress: null,
            error: a.status === 'failed' ? (a.error ?? 'The analysis failed') : null,
          }));
          stopWatching();
          return;
        }
        setState((s) => ({ ...s, stage: a.stage, progress: a.progress ?? s.progress }));
      } catch (err) {
        if (stopped) return;
        setState((s) => ({
          ...s,
          error: err instanceof Error ? err.message : 'Could not load the result',
        }));
        return;
      }
      timer = setTimeout(() => void load(), POLL_MS);
    };

    const stopWatching = watchAnalysis(
      analysisId,
      (e) => {
        if (finished) return;
        setState((s) => ({ ...s, stage: e.stage, progress: e.progress ?? null }));
        if (e.stage === 'done' || e.stage === 'failed') void load();
      },
      (connection) => !stopped && setState((s) => ({ ...s, connection })),
    );
    void load();
    return () => {
      stopped = true;
      clearTimeout(timer);
      stopWatching();
    };
  }, [analysisId]);

  return state;
}

export const CONNECTION_TEXT: Record<Connection, string | null> = {
  connecting: null,
  live: null,
  reconnecting:
    'Live updates interrupted; reconnecting. Progress is still checked every few seconds.',
  offline: 'Live updates unavailable; progress is checked every few seconds instead.',
  expired: 'Your session has expired. Log in again to see live updates.',
};
