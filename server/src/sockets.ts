import type { Server as HttpServer } from 'node:http';
import mongoose from 'mongoose';
import { Server } from 'socket.io';
import { logger } from './logger.js';
import { verifyAccessToken, type AuthUser } from './middleware/auth.js';
import { Analysis } from './models/index.js';
import type { Stage } from './types.js';

let io: Server | null = null;

const room = (analysisId: string): string => `analysis:${analysisId}`;

export function attachSockets(server: HttpServer): Server {
  io = new Server(server, { cors: { origin: true } });

  io.use((socket, next) => {
    const token: unknown = socket.handshake.auth.token;
    const user = typeof token === 'string' ? verifyAccessToken(token) : null;
    if (!user) return next(new Error('unauthorized'));
    socket.data.user = user;
    next();
  });

  io.on('connection', (socket) => {
    const user = socket.data.user as AuthUser;
    socket.on('subscribe', async (analysisId: unknown, ack?: (ok: boolean) => void) => {
      try {
        const valid = typeof analysisId === 'string' && mongoose.isValidObjectId(analysisId);
        const owned = valid && (await Analysis.exists({ _id: analysisId, ownerId: user.id }));
        if (owned) await socket.join(room(analysisId));
        ack?.(Boolean(owned));
      } catch (err) {
        logger.warn({ err }, 'socket subscribe failed');
        ack?.(false);
      }
    });
  });
  return io;
}

export interface ProgressEvent {
  analysisId: string;
  stage: Stage;
  detail?: string;
}

/** Send a progress event to everyone watching this analysis. */
export function emitProgress(event: ProgressEvent): void {
  io?.to(room(event.analysisId)).emit('progress', event);
}
