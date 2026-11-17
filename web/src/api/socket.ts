import { io } from 'socket.io-client';
import { getSession, refresh } from './client';
import type { Progress, Stage, Status } from './types';

export interface ProgressEvent {
  analysisId: string;
  stage: Stage;
  status?: Status;
  progress?: Progress | null;
  detail?: string;
  snapshot?: boolean;
}

export type Connection = 'connecting' | 'live' | 'reconnecting' | 'offline' | 'expired';

/**
 * Watch one analysis over Socket.IO. The server sends the current state on subscribe, so a
 * late or reconnected client is never behind. Reconnects re-subscribe; an expired token is
 * refreshed once and the connection retried. When live updates are unavailable, callers keep
 * polling (Detect does).
 */
export function watchAnalysis(
  analysisId: string,
  onProgress: (e: ProgressEvent) => void,
  onConnection: (c: Connection) => void = () => undefined,
): () => void {
  let refreshed = false;
  const socket = io({
    // A function, so every (re)connection uses the current access token.
    auth: (cb) => cb({ token: getSession()?.accessToken }),
    reconnectionAttempts: 8,
    reconnectionDelayMax: 10_000,
  });
  onConnection('connecting');
  socket.on('connect', () => {
    onConnection('live');
    socket.emit('subscribe', analysisId, (ok: boolean) => {
      if (!ok) onConnection('offline');
    });
  });
  socket.on('progress', (e: ProgressEvent) => {
    if (e.analysisId === analysisId) onProgress(e);
  });
  socket.on('connect_error', (err: Error & { data?: { reason?: string } }) => {
    if (err.message === 'unauthorized') {
      if (err.data?.reason === 'expired' && !refreshed) {
        refreshed = true;
        void refresh().then((ok) => (ok ? socket.connect() : onConnection('expired')));
      } else {
        onConnection('expired');
      }
      return;
    }
    onConnection('reconnecting');
  });
  socket.io.on('reconnect_failed', () => onConnection('offline'));
  socket.on('disconnect', (reason) => {
    if (reason !== 'io client disconnect') onConnection('reconnecting');
  });
  return () => {
    socket.disconnect();
  };
}
