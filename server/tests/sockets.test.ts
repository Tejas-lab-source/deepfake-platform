/**
 * Socket.IO progress, end to end over a real local server and client: JWT handshake,
 * ownership, late-subscriber snapshots and event isolation between users.
 */
import { createServer, type Server as HttpServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import jwt from 'jsonwebtoken';
import { io as connect, type Socket } from 'socket.io-client';
import { afterEach, describe, expect, it } from 'vitest';
import { config } from '../src/config.js';
import { signTokens } from '../src/middleware/auth.js';
import { attachSockets, emitProgress } from '../src/sockets.js';
import { memoryRepos } from './helpers/memoryRepos.js';

const open: Array<{ close: () => void }> = [];
afterEach(() => {
  while (open.length) open.pop()!.close();
});

async function start() {
  const repos = memoryRepos();
  const http: HttpServer = createServer();
  const io = attachSockets(http, { analyses: repos.analyses, corsOrigins: [] });
  await new Promise<void>((r) => http.listen(0, '127.0.0.1', r));
  open.push({ close: () => void io.close() });
  const url = `http://127.0.0.1:${(http.address() as AddressInfo).port}`;
  const user = async () => {
    const u = await repos.users.create({ email: `${Math.random()}@x.test`, passwordHash: 'x' });
    return { id: u.id, token: signTokens({ id: u.id, role: 'user' }).accessToken };
  };
  const analysis = async (ownerId: string) =>
    repos.analyses.create({
      ownerId,
      mediaId: 'f'.repeat(24),
      mediaSha256: 'x',
      kind: 'video',
      externalCheckRequested: false,
    });
  const client = (token: unknown): Socket => {
    const s = connect(url, { auth: { token }, transports: ['websocket'], reconnection: false });
    open.push({ close: () => s.close() });
    return s;
  };
  return { repos, user, analysis, client };
}

const connected = (s: Socket) =>
  new Promise<void>((resolve, reject) => {
    s.on('connect', () => resolve());
    s.on('connect_error', reject);
  });

const refused = (s: Socket) =>
  new Promise<{ message: string; data: unknown }>((resolve) => {
    s.on('connect_error', (err: Error & { data?: unknown }) =>
      resolve({ message: err.message, data: err.data }),
    );
  });

const subscribe = (s: Socket, id: string) =>
  new Promise<boolean>((resolve) => s.emit('subscribe', id, resolve));

describe('socket handshake', () => {
  it('refuses missing, invalid and expired tokens with a clear reason', async () => {
    const t = await start();
    expect(await refused(t.client(undefined))).toEqual({
      message: 'unauthorized',
      data: { reason: 'missing' },
    });
    expect((await refused(t.client('not-a-jwt'))).data).toEqual({ reason: 'invalid' });
    const expired = jwt.sign(
      { sub: 'a'.repeat(24), role: 'user', typ: 'access', exp: Math.floor(Date.now() / 1000) - 60 },
      config.JWT_ACCESS_SECRET,
    );
    expect((await refused(t.client(expired))).data).toEqual({ reason: 'expired' });
    const refresh = signTokens({ id: 'a'.repeat(24), role: 'user' }).refreshToken;
    expect((await refused(t.client(refresh))).data).toEqual({ reason: 'invalid' });
  });
});

describe('progress rooms', () => {
  it('lets owners subscribe and sends late subscribers the current state at once', async () => {
    const t = await start();
    const me = await t.user();
    const a = await t.analysis(me.id);
    await t.repos.analyses.update(a.id, {
      status: 'processing',
      stage: 'scoring',
      progress: { done: 12, total: 32, unit: 'frames' },
    });
    const s = t.client(me.token);
    await connected(s);
    const events: unknown[] = [];
    s.on('progress', (e) => events.push(e));
    expect(await subscribe(s, a.id)).toBe(true);
    await new Promise((r) => setTimeout(r, 50));
    expect(events[0]).toEqual({
      analysisId: a.id,
      stage: 'scoring',
      status: 'processing',
      progress: { done: 12, total: 32, unit: 'frames' },
      snapshot: true,
    });

    emitProgress({
      analysisId: a.id,
      stage: 'scoring',
      status: 'processing',
      progress: { done: 13, total: 32, unit: 'frames' },
    });
    await new Promise((r) => setTimeout(r, 50));
    expect(events[1]).toMatchObject({ progress: { done: 13 } });
  });

  it("never delivers another user's events, and refuses their rooms", async () => {
    const t = await start();
    const me = await t.user();
    const other = await t.user();
    const theirs = await t.analysis(other.id);
    const mine = await t.analysis(me.id);
    const s = t.client(me.token);
    await connected(s);
    const events: Array<{ analysisId: string }> = [];
    s.on('progress', (e) => events.push(e));
    expect(await subscribe(s, theirs.id)).toBe(false);
    expect(await subscribe(s, 'not-an-id')).toBe(false);
    expect(await subscribe(s, '0'.repeat(24))).toBe(false);
    expect(await subscribe(s, mine.id)).toBe(true);
    emitProgress({ analysisId: theirs.id, stage: 'done', status: 'done' });
    emitProgress({ analysisId: mine.id, stage: 'done', status: 'done' });
    await new Promise((r) => setTimeout(r, 80));
    expect(events.map((e) => e.analysisId)).toEqual([mine.id, mine.id]); // snapshot + done
  });

  it('stops delivering after unsubscribe', async () => {
    const t = await start();
    const me = await t.user();
    const a = await t.analysis(me.id);
    const s = t.client(me.token);
    await connected(s);
    const events: unknown[] = [];
    s.on('progress', (e) => events.push(e));
    await subscribe(s, a.id);
    s.emit('unsubscribe', a.id);
    await new Promise((r) => setTimeout(r, 50));
    emitProgress({ analysisId: a.id, stage: 'done', status: 'done' });
    await new Promise((r) => setTimeout(r, 50));
    expect(events).toHaveLength(1); // only the snapshot
  });
});
