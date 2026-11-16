import { io } from 'socket.io-client';
import { getSession } from './client';
import type { Stage } from './types';

export interface ProgressEvent {
  analysisId: string;
  stage: Stage;
  detail?: string;
}

/** Watch one analysis over Socket.IO. Returns a function that stops watching. */
export function watchAnalysis(analysisId: string, onProgress: (e: ProgressEvent) => void) {
  const socket = io({ auth: { token: getSession()?.accessToken }, reconnectionAttempts: 5 });
  socket.on('connect', () => socket.emit('subscribe', analysisId));
  socket.on('progress', (e: ProgressEvent) => {
    if (e.analysisId === analysisId) onProgress(e);
  });
  return () => {
    socket.disconnect();
  };
}
