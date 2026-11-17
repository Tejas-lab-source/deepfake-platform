/**
 * End-to-end smoke test against a running stack (mock mode), through the same public entry
 * point a browser uses. No external service is contacted: Hive runs in mock mode, Groq is off.
 *
 *   docker compose up -d --build
 *   cd server && npx tsx scripts/smoke.ts            # SMOKE_BASE=http://localhost:8080
 *
 * Checks: health, auth, image upload with live Socket.IO progress, result, heatmap route,
 * Ask, PDF, video upload with frame progress and timestamps, error handling, deletion.
 * Exits non-zero if any check fails.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { io as connect } from 'socket.io-client';

const BASE = process.env.SMOKE_BASE ?? 'http://localhost:8080';
const API = `${BASE}/api/v1`;
const HERE = path.dirname(fileURLToPath(import.meta.url));
/** "real": the stack runs the real models (MOCK_MODE=false); media should contain a face. */
const REAL = process.env.SMOKE_EXPECT === 'real';
const IMAGE = readFileSync(
  process.env.SMOKE_IMAGE ?? path.resolve(HERE, '../../ml-service/tests/fixtures/sample.png'),
);
const VIDEO = readFileSync(
  process.env.SMOKE_VIDEO ?? path.resolve(HERE, 'fixtures/synthetic-clip.mp4'),
);
const VIDEO_FRAMES = Number(process.env.SMOKE_VIDEO_FRAMES ?? 30);

const results: Array<{ name: string; ok: boolean; detail?: string }> = [];
function check(name: string, ok: boolean, detail?: string): void {
  results.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
}

interface Session {
  accessToken: string;
  user: { id: string };
}

async function call(
  method: string,
  p: string,
  token?: string,
  body?: unknown,
): Promise<{ status: number; json: Record<string, unknown> | null; res: Response }> {
  const res = await fetch(`${API}${p}`, {
    method,
    headers: {
      ...(token ? { authorization: `Bearer ${token}` } : {}),
      ...(body !== undefined ? { 'content-type': 'application/json' } : {}),
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const type = res.headers.get('content-type') ?? '';
  const json = type.includes('json') ? ((await res.json()) as Record<string, unknown>) : null;
  return { status: res.status, json, res };
}

async function register(): Promise<Session> {
  const email = `smoke-${Date.now()}-${Math.random().toString(36).slice(2)}@example.test`;
  const r = await call('POST', '/auth/register', undefined, {
    email,
    password: 'smoke-test-password',
  });
  if (r.status !== 201) throw new Error(`register failed: ${r.status}`);
  return r.json as unknown as Session;
}

async function upload(
  token: string,
  bytes: Buffer,
  name: string,
  type: string,
  externalCheck = false,
) {
  const form = new FormData();
  form.append('externalCheck', String(externalCheck));
  form.append('file', new Blob([new Uint8Array(bytes)], { type }), name);
  const res = await fetch(`${API}/analyses`, {
    method: 'POST',
    headers: { authorization: `Bearer ${token}` },
    body: form,
  });
  return { status: res.status, json: (await res.json()) as Record<string, unknown> };
}

interface Progress {
  stage: string;
  status?: string;
  progress?: { done: number; total: number; unit: string } | null;
  snapshot?: boolean;
}

/** Follow one analysis over Socket.IO until done/failed (or timeout). */
function follow(token: string, analysisId: string, timeoutMs = 120_000): Promise<Progress[]> {
  return new Promise((resolve, reject) => {
    const events: Progress[] = [];
    const socket = connect(BASE, {
      auth: { token },
      transports: ['websocket'],
      reconnection: false,
    });
    const timer = setTimeout(() => {
      socket.close();
      reject(
        new Error(
          `no completion within ${timeoutMs} ms; got ${events.map((e) => e.stage).join(',')}`,
        ),
      );
    }, timeoutMs);
    socket.on('connect', () =>
      socket.emit(
        'subscribe',
        analysisId,
        (ok: boolean) => ok || reject(new Error('subscribe refused')),
      ),
    );
    socket.on('connect_error', (e) => reject(e));
    socket.on('progress', (e: Progress & { analysisId: string }) => {
      if (e.analysisId !== analysisId) return;
      events.push(e);
      if (e.stage === 'done' || e.stage === 'failed') {
        clearTimeout(timer);
        socket.close();
        resolve(events);
      }
    });
  });
}

async function main(): Promise<void> {
  console.log(`Smoke test against ${BASE}\n`);
  const health = await call('GET', '/health');
  check(
    'health',
    health.status === 200 && health.json?.status === 'ok',
    `mockMode=${String(health.json?.mockMode)}`,
  );
  check(
    REAL ? 'real mode (models loaded)' : 'mock mode is on (no models, no external calls)',
    health.json?.mockMode === !REAL,
  );

  // ---- auth
  const anon = await call('GET', '/analyses');
  check('unauthenticated requests are refused', anon.status === 401);
  const me = await register();
  const other = await register();
  check('register returns tokens', typeof me.accessToken === 'string');

  // ---- image
  const img = await upload(me.accessToken, IMAGE, 'portrait.png', 'image/png', true);
  check('image upload accepted', img.status === 202, `status ${img.status}`);
  const imageId = String(img.json.analysisId);
  const imgEvents = await follow(me.accessToken, imageId);
  const stages = imgEvents.map((e) => e.stage);
  check(
    'socket snapshot then live stages',
    imgEvents[0]?.snapshot === true && stages.at(-1) === 'done',
    stages.join(' > '),
  );
  const result = await call('GET', `/analyses/${imageId}`, me.accessToken);
  const final = result.json?.final as { verdict?: string } | null;
  check(
    'image result',
    result.status === 200 && result.json?.status === 'done' && !!final?.verdict,
    `verdict ${final?.verdict}`,
  );
  const external = result.json?.external as { provider?: string; status?: string } | null;
  check(
    'second check ran with the mock verifier only (consent given, no real Hive call)',
    external?.status === 'ok' && external.provider === 'hive-mock',
    `provider ${external?.provider}`,
  );
  const explanation = result.json?.explanation as { fallbackUsed?: boolean } | null;
  check('explanation from the template (Groq off)', explanation?.fallbackUsed === true);
  const ml = result.json?.ml as {
    faces?: Array<{ heatmaps: { gend: string } | null }>;
  } | null;
  const heatKey = ml?.faces?.[0]?.heatmaps?.gend ?? null;
  if (REAL) {
    const name = String(heatKey).replace(/^derived\//, '');
    const h = await fetch(`${API}/analyses/${imageId}/heatmaps/${name}`, {
      headers: { authorization: `Bearer ${me.accessToken}` },
    });
    const png = Buffer.from(await h.arrayBuffer());
    check(
      'real heatmap served to its owner as a PNG',
      h.status === 200 && png.subarray(1, 4).toString() === 'PNG',
      `${png.length} bytes`,
    );
    const theirs = await call('GET', `/analyses/${imageId}/heatmaps/${name}`, other.accessToken);
    check('another user cannot fetch the heatmap', theirs.status === 404);
  } else {
    const heat = await call(
      'GET',
      `/analyses/${imageId}/heatmaps/${'0'.repeat(32)}.png`,
      me.accessToken,
    );
    check(
      'heatmap route: none in mock mode, unknown names refused',
      heatKey === null && heat.status === 404,
      `status ${heat.status}`,
    );
  }
  const traversal = await call(
    'GET',
    `/analyses/${imageId}/heatmaps/..%2F..%2Fetc%2Fpasswd`,
    me.accessToken,
  );
  check('heatmap path traversal refused', traversal.status === 400);

  // ---- ask, pdf
  const ask = await call('POST', `/analyses/${imageId}/ask`, me.accessToken, {
    question: 'Why was this flagged?',
  });
  check(
    'ask answers from the template',
    ask.status === 200 && ask.json?.source === 'template',
    String(ask.json?.answer).slice(0, 70),
  );
  const off = await call('POST', `/analyses/${imageId}/ask`, me.accessToken, {
    question: 'What is the capital of France?',
  });
  check('ask refuses unrelated questions', off.json?.intent === 'unsupported');
  const pdf = await fetch(`${API}/analyses/${imageId}/report.pdf`, {
    headers: { authorization: `Bearer ${me.accessToken}` },
  });
  const pdfBytes = Buffer.from(await pdf.arrayBuffer());
  check(
    'PDF report downloads',
    pdf.status === 200 &&
      pdf.headers.get('content-type') === 'application/pdf' &&
      pdfBytes.subarray(0, 5).toString() === '%PDF-',
    `${pdfBytes.length} bytes`,
  );

  // ---- video
  const vid = await upload(me.accessToken, VIDEO, 'clip.mp4', 'video/mp4');
  check('video upload accepted', vid.status === 202, `status ${vid.status}`);
  const videoId = String(vid.json.analysisId);
  const vidEvents = await follow(me.accessToken, videoId);
  const vStages = [...new Set(vidEvents.map((e) => e.stage))];
  check('video progress stages', vStages.includes('done'), vStages.join(' > '));
  const v = await call('GET', `/analyses/${videoId}`, me.accessToken);
  if (REAL) {
    const real = v.json?.ml as { framesScored?: number; mostSuspicious?: unknown; mock?: boolean };
    check(
      'real video: faces scored, most suspicious frame found',
      !real.mock && (real.framesScored ?? 0) > 0 && real.mostSuspicious != null,
      `${real.framesScored} frames scored`,
    );
  }
  const vml = v.json?.ml as {
    frames?: Array<{ index: number; timestamp: number | null; hasFace: boolean }>;
    video?: { frameCount: number; fps: number };
    framesSampled?: number;
  } | null;
  const framesOk =
    !!vml?.frames?.length &&
    vml.video?.frameCount === VIDEO_FRAMES &&
    vml.frames.every((f) => f.timestamp === Math.round((f.index / vml.video!.fps) * 1000) / 1000);
  check(
    'video frames with real timestamps',
    framesOk,
    `${vml?.framesSampled} sampled of ${vml?.video?.frameCount}`,
  );
  const vpdf = await fetch(`${API}/analyses/${videoId}/report.pdf`, {
    headers: { authorization: `Bearer ${me.accessToken}` },
  });
  check('video PDF report', vpdf.status === 200);

  // ---- errors
  const bad = await upload(me.accessToken, Buffer.from('not an image'), 'x.png', 'image/png');
  check('non-media upload refused by content', bad.status === 415);
  const notMine = await call('GET', `/analyses/${imageId}`, other.accessToken);
  check("another user's analysis is not found", notMine.status === 404);
  const longQ = await call('POST', `/analyses/${imageId}/ask`, me.accessToken, {
    question: 'x'.repeat(600),
  });
  check('over-long question refused', longQ.status === 400);
  const cmp = await call('POST', `/analyses/${imageId}/compare`, me.accessToken, {
    family: 'F1',
    strength: 0.5,
  });
  if (REAL) {
    check(
      'Filter Lab compares clean vs filtered with the real models',
      cmp.status === 201 && typeof cmp.json?.flipped === 'boolean',
      `flipped=${String(cmp.json?.flipped)}, IoU=${String(cmp.json?.heatmapIoU)}`,
    );
  } else {
    check(
      'Filter Lab says the real pipeline is needed (mock mode)',
      cmp.status === 501,
      String(cmp.json?.detail),
    );
  }
  const otherSocket = await new Promise<boolean>((resolve) => {
    const s = connect(BASE, {
      auth: { token: other.accessToken },
      transports: ['websocket'],
      reconnection: false,
    });
    s.on('connect', () => s.emit('subscribe', imageId, (ok: boolean) => (s.close(), resolve(ok))));
    s.on('connect_error', () => resolve(false));
  });
  check("another user cannot subscribe to the analysis's events", otherSocket === false);

  // ---- delete
  const del = await call('DELETE', `/analyses/${imageId}`, me.accessToken);
  const gone = await call('GET', `/analyses/${imageId}`, me.accessToken);
  check('delete removes the analysis', del.status === 204 && gone.status === 404);
  await call('DELETE', `/analyses/${videoId}`, me.accessToken);

  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  process.exitCode = failed.length === 0 ? 0 : 1;
}

main().catch((err: unknown) => {
  console.error('Smoke test aborted:', err instanceof Error ? err.message : err);
  process.exitCode = 1;
});
