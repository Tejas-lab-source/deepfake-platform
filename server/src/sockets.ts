/**
 * Live progress over Socket.IO.
 *
 *   handshake   auth: { token } (a JWT access token). Refused with connect_error
 *               "unauthorized" (data.reason: missing | invalid | expired).
 *   subscribe   (analysisId, ack) -> ack(true) only for the caller's own analysis; the socket
 *               then joins that analysis's room and immediately receives a `progress` snapshot
 *               of the current state, so a late subscriber never misses where the job is.
 *   progress    { analysisId, stage, status, progress?: {done,total,unit}, detail?, snapshot? }
 *   unsubscribe (analysisId) leaves the room.
 *
 * Rooms are per analysis and joining needs ownership, so a user only ever receives events for
 * their own jobs.
 */
import type { Server as HttpServer } from 'node:http';
import { Server } from 'socket.io';
import type { AnalysisRepo } from './data/types.js';
import { logger } from './logger.js';
import { checkAccessToken, type AuthUser } from './middleware/auth.js';
import type { ProgressEvent } from './queue/process.js';

let io: Server | null = null;

const room = (analysisId: string): string => `analysis:${analysisId}`;
const OBJECT_ID = /^[0-9a-f]{24}$/i;
/** Rooms one socket may be in at a time (a user watches a handful of jobs at most). */
const MAX_ROOMS = 20;

export interface SocketDeps {
  analyses: AnalysisRepo;
  corsOrigins: string[];
}

export function attachSockets(server: HttpServer, deps: SocketDeps): Server {
  io = new Server(server, {
    // Same origin unless origins are configured: the web container proxies /socket.io.
    cors: deps.corsOrigins.length > 0 ? { origin: deps.corsOrigins } : undefined,
    maxHttpBufferSize: 10_000,
  });

  io.use((socket, next) => {
    const check = checkAccessToken(socket.handshake.auth?.token);
    if (!check.ok) {
      const err = new Error('unauthorized') as Error & { data?: unknown };
      err.data = { reason: check.reason };
      return next(err);
    }
    socket.data.user = check.user;
    next();
  });

  io.on('connection', (socket) => {
    const user = socket.data.user as AuthUser;

    socket.on('subscribe', async (analysisId: unknown, ack?: unknown) => {
      const reply = typeof ack === 'function' ? (ack as (ok: boolean) => void) : () => undefined;
      try {
        if (typeof analysisId !== 'string' || !OBJECT_ID.test(analysisId)) return reply(false);
        if (socket.rooms.size > MAX_ROOMS) return reply(false);
        const analysis = await deps.analyses.findOwned(analysisId, user.id);
        if (!analysis) return reply(false);
        await socket.join(room(analysisId));
        reply(true);
        const snapshot: ProgressEvent & { snapshot: true } = {
          analysisId,
          stage: analysis.stage,
          status: analysis.status,
          progress: analysis.progress,
          ...(analysis.error ? { detail: analysis.error } : {}),
          snapshot: true,
        };
        socket.emit('progress', snapshot);
      } catch (err) {
        logger.warn({ err }, 'socket subscribe failed');
        reply(false);
      }
    });

    socket.on('unsubscribe', (analysisId: unknown) => {
      if (typeof analysisId === 'string' && OBJECT_ID.test(analysisId)) {
        void socket.leave(room(analysisId));
      }
    });
  });
  return io;
}

/** Send a progress event to the sockets watching this analysis (its owner's). */
export function emitProgress(event: ProgressEvent): void {
  io?.to(room(event.analysisId)).emit('progress', event);
}
