/**
 * M7 live check: one Hive request and one Groq request, with synthetic data only.
 *
 *   Dry run (default, no network):  npx tsx scripts/liveCheck.ts
 *   Live (needs approval):          M7_LIVE=1 npx tsx --env-file=live-check/.env.m7 \
 *                                     scripts/liveCheck.ts --live [--allow-repeat]
 *
 * What is sent:
 *   Hive  a generated abstract image (no person, no face), uploaded as "upload" (image/png)
 *   Groq  the system prompt and the whitelisted result JSON built from
 *         live-check/synthetic-result.json, exactly as the app builds it
 * Nothing else: no user image, file name, path, personal data or analysis.
 *
 * Safeguards in live mode:
 *   - Destinations: only the official Hive and Groq URLs over HTTPS on the default port.
 *     Custom URLs are refused before anything starts, every request is checked again, and
 *     redirects are refused, never followed, so credentials cannot reach another host.
 *   - Budget: one unit is reserved in the app's Redis daily Hive budget (same code and
 *     limits) before the Hive request. Denied or Redis unavailable: nothing is sent.
 *     Reservations are never refunded.
 *   - One request per service: no retries, one Groq model.
 *   - Repeat runs: refused while an earlier live report or run marker exists, unless
 *     --allow-repeat is given. Earlier reports are never overwritten or deleted.
 */
import { mkdtempSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { deflateSync } from 'node:zlib';
import { Redis } from 'ioredis';
import { DailyBudget } from '../src/budget.js';
import { loadConfig } from '../src/config.js';
import { buildResultJson, EXPLANATION_SCHEMA, SYSTEM_PROMPT } from '../src/explain/prompt.js';
import { validateExplanation } from '../src/explain/validate.js';
import type { ExplanationInput } from '../src/explainTemplate.js';
import type { FetchLike } from '../src/http.js';
import { GroqClient } from '../src/providers/groq.js';
import { HiveVerifier } from '../src/providers/hive.js';
import { scrubDeep, registerSecrets } from '../src/secrets.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const LIVE_CHECK_DIR = path.resolve(HERE, '../live-check');
export const HIVE_URL =
  'https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection';
export const GROQ_BASE_URL = 'https://api.groq.com/openai/v1';
/** The only URLs a live run may contact, exactly. */
export const APPROVED_URLS = new Set([HIVE_URL, `${GROQ_BASE_URL}/chat/completions`]);
/** Identity the live check uses in the Redis Hive budget (the demo pool). */
export const M7_BUDGET_USER = 'm7-live-check';

// ---- synthetic data --------------------------------------------------------------------

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function crc32(buf: Buffer): number {
  let c = 0xffffffff;
  for (const byte of buf) c = CRC_TABLE[(c ^ byte) & 0xff]! ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Buffer): Buffer {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
}

/** A deterministic abstract RGB image: colour gradients and rings. No person, no face. */
export function syntheticPng(size = 256): Buffer {
  const raw = Buffer.alloc((size * 3 + 1) * size);
  for (let y = 0; y < size; y++) {
    const row = y * (size * 3 + 1);
    raw[row] = 0; // filter: none
    for (let x = 0; x < size; x++) {
      const d = Math.hypot(x - size / 2, y - size / 2);
      const ring = Math.floor(d / 16) % 2 === 0 ? 40 : 0;
      const i = row + 1 + x * 3;
      raw[i] = Math.min(255, Math.round((x / size) * 200) + ring);
      raw[i + 1] = Math.min(255, Math.round((y / size) * 200) + ring);
      raw[i + 2] = Math.min(255, 120 + ring);
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // colour type: RGB
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

export function syntheticResult(): { input: ExplanationInput; extras: Record<string, unknown> } {
  const data = JSON.parse(readFileSync(path.join(LIVE_CHECK_DIR, 'synthetic-result.json'), 'utf8'));
  return { input: data.input as ExplanationInput, extras: data.extras as Record<string, unknown> };
}

// ---- safety gate -----------------------------------------------------------------------

export function liveAllowed(argv: string[], env: NodeJS.ProcessEnv): { ok: boolean; why: string } {
  if (!argv.includes('--live')) return { ok: false, why: 'dry run (no --live flag)' };
  if (env.M7_LIVE !== '1') return { ok: false, why: 'M7_LIVE is not 1' };
  if (env.HIVE_MODE !== 'real' && env.GROQ_MODE !== 'real') {
    return { ok: false, why: 'neither HIVE_MODE nor GROQ_MODE is real' };
  }
  return { ok: true, why: 'live run approved by flag and M7_LIVE=1' };
}

// ---- destinations ----------------------------------------------------------------------

export class DestinationRefused extends Error {}

/** Throws unless `raw` is exactly one of the approved URLs: HTTPS, the approved host, the
 * default port, no credentials, no query or fragment, the approved path. */
export function assertApprovedUrl(raw: string): URL {
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new DestinationRefused('not a valid URL');
  }
  if (url.protocol !== 'https:')
    throw new DestinationRefused(`protocol ${url.protocol} is not https`);
  if (url.port !== '') throw new DestinationRefused(`port ${url.port} is not allowed`);
  if (url.username || url.password)
    throw new DestinationRefused('URLs with credentials are not allowed');
  if (url.search || url.hash)
    throw new DestinationRefused('query strings and fragments are not allowed');
  const normalised = `${url.protocol}//${url.hostname}${url.pathname}`;
  if (!APPROVED_URLS.has(normalised)) {
    throw new DestinationRefused(`${url.hostname}${url.pathname} is not an approved destination`);
  }
  return url;
}

/** Live mode only accepts the official endpoints: a custom HIVE_API_URL or GROQ_BASE_URL is
 * refused before anything starts (empty means the official default). */
export function liveDestinations(env: NodeJS.ProcessEnv): { hiveUrl: string; groqBase: string } {
  const hive = (env.HIVE_API_URL ?? '').trim();
  const groq = (env.GROQ_BASE_URL ?? '').trim().replace(/\/$/, '');
  if (hive && hive !== HIVE_URL)
    throw new DestinationRefused('custom HIVE_API_URL is not allowed in live mode');
  if (groq && groq !== GROQ_BASE_URL)
    throw new DestinationRefused('custom GROQ_BASE_URL is not allowed in live mode');
  assertApprovedUrl(HIVE_URL);
  assertApprovedUrl(`${GROQ_BASE_URL}/chat/completions`);
  return { hiveUrl: HIVE_URL, groqBase: GROQ_BASE_URL };
}

export interface GuardEvent {
  url: string;
  outcome: 'allowed' | 'refused_destination' | 'refused_redirect';
  detail?: string;
}

/** Wraps a fetch so a request (and its authorization header) only ever leaves for an approved
 * URL, and redirects are never followed: a 3xx answer is refused. */
export function guardedFetch(inner: FetchLike, events: GuardEvent[]): FetchLike {
  return async (url, init) => {
    try {
      assertApprovedUrl(url);
    } catch (err) {
      events.push({
        url: safeUrl(url),
        outcome: 'refused_destination',
        detail: (err as Error).message,
      });
      throw err;
    }
    const res = await inner(url, { ...init, redirect: 'manual' });
    if (res.status >= 300 && res.status < 400) {
      const location = res.headers.get('location') ?? '';
      const target = safeUrl(location);
      events.push({
        url: safeUrl(url),
        outcome: 'refused_redirect',
        detail: `redirect to ${target}`,
      });
      await res.body?.cancel().catch(() => undefined);
      throw new DestinationRefused(`redirect to ${target} refused`);
    }
    events.push({ url: safeUrl(url), outcome: 'allowed' });
    return res;
  };
}

/** Host and path only: never query strings, which can carry tokens. */
function safeUrl(raw: string): string {
  try {
    const u = new URL(raw);
    return `${u.protocol}//${u.host}${u.pathname}`;
  } catch {
    return '(unparseable URL)';
  }
}

// ---- repeat-run protection -------------------------------------------------------------

export class RepeatRunRefused extends Error {}

const LIVE_ARTIFACT = /^live-(report|run)-.+\.(json|marker)$/;

export function previousLiveRuns(outDir: string): string[] {
  try {
    return readdirSync(outDir)
      .filter((f) => LIVE_ARTIFACT.test(f))
      .sort();
  } catch {
    return [];
  }
}

/** Write a file that must not exist yet ('wx'): earlier reports are never overwritten. */
function writeNew(file: string, text: string): void {
  writeFileSync(file, text, { flag: 'wx' });
}

// ---- the check -------------------------------------------------------------------------

export interface RecordedRequest {
  service: 'hive' | 'groq';
  url: string;
  method: string;
  headerNames: string[];
  formFields?: Array<{ name: string; fileName: string; type: string; bytes: number }>;
  jsonBody?: unknown;
}

/** A fetch that records what would be sent and answers with a simulated documented
 * response. It never opens a network connection. */
export function dryRunFetch(recorded: RecordedRequest[]): FetchLike {
  const hiveExample = readFileSync(
    path.resolve(HERE, '../tests/fixtures/hive-v3-image.json'),
    'utf8',
  );
  return async (url, init) => {
    const service = url.includes('groq') ? 'groq' : 'hive';
    const headers = init?.headers as Record<string, string>;
    const entry: RecordedRequest = {
      service,
      url,
      method: init?.method ?? 'GET',
      headerNames: Object.keys(headers ?? {}).sort(),
    };
    if (init?.body instanceof FormData) {
      entry.formFields = [...init.body.entries()].map(([name, value]) => ({
        name,
        fileName: value instanceof File ? value.name : '',
        type: value instanceof File ? value.type : 'text',
        bytes: value instanceof File ? value.size : String(value).length,
      }));
    } else if (typeof init?.body === 'string') {
      entry.jsonBody = JSON.parse(init.body);
    }
    recorded.push(entry);
    if (service === 'hive') return new Response(hiveExample, { status: 200 });
    const content = JSON.stringify({
      summary: 'SIMULATED: the automated check found signs of face manipulation.',
      why: 'SIMULATED dry-run reply. Our pipeline scored 87% against a 50% threshold; the second check scored 12% against 90%, so the two checks disagree.',
      caveats: ['This is an automated check and is not proof.'],
    });
    return new Response(
      JSON.stringify({
        model: 'simulated',
        choices: [{ finish_reason: 'stop', message: { content } }],
      }),
      { status: 200 },
    );
  };
}

export interface RunOptions {
  live: boolean;
  env: NodeJS.ProcessEnv;
  /** The network, for a live run. Tests inject a fake; it is always wrapped by the guard. */
  fetchImpl?: FetchLike;
  outDir?: string;
  allowRepeat?: boolean;
  /** The Redis Hive budget for a live run. Default: the app's limits on REDIS_URL. */
  budget?: DailyBudget;
  now?: () => Date;
}

export async function runCheck(opts: RunOptions): Promise<Record<string, unknown>> {
  const { env } = opts;
  registerSecrets([env.HIVE_API_KEY ?? '', env.GROQ_API_KEY ?? '']);
  const out = opts.outDir ?? path.join(LIVE_CHECK_DIR, 'out');
  const stamp = (opts.now ?? (() => new Date()))().toISOString().replace(/[:.]/g, '-');
  const recorded: RecordedRequest[] = [];
  const guardEvents: GuardEvent[] = [];
  let ownedRedis: Redis | null = null;

  // Live-only checks, all before any network or Redis use.
  let destinations = { hiveUrl: HIVE_URL, groqBase: GROQ_BASE_URL };
  if (opts.live) {
    const earlier = previousLiveRuns(out);
    if (earlier.length > 0 && !opts.allowRepeat) {
      throw new RepeatRunRefused(
        `an earlier live run exists (${earlier.join(', ')}); pass --allow-repeat to run again`,
      );
    }
    destinations = liveDestinations(env);
  }
  const network = opts.live ? (opts.fetchImpl ?? ((u, i) => fetch(u, i))) : dryRunFetch(recorded);
  const fetchImpl = guardedFetch(network, guardEvents);

  mkdirSync(out, { recursive: true });
  if (opts.live) {
    // The marker exists before any request, so even a crashed run blocks an unplanned repeat.
    writeNew(path.join(out, `live-run-${stamp}.marker`), `started ${stamp}\n`);
  }

  // A fresh temporary folder holds the generated image; nothing from the app is read.
  const work = mkdtempSync(path.join(tmpdir(), 'm7-'));
  const image = path.join(work, 'synthetic.png');
  writeFileSync(image, syntheticPng());

  const report: Record<string, unknown> = {
    mode: opts.live ? 'LIVE' : 'DRY RUN (no network, simulated responses)',
    startedAt: stamp,
  };

  try {
    if (!opts.live || env.HIVE_MODE === 'real') {
      report.hive = await hiveCheck(opts, destinations.hiveUrl, image, fetchImpl, (r) => {
        ownedRedis = r;
      });
    }

    if (!opts.live || env.GROQ_MODE === 'real') {
      const { input, extras } = syntheticResult();
      const facts = buildResultJson(input, extras);
      const groq = new GroqClient({
        apiKey: env.GROQ_API_KEY || (opts.live ? '' : 'dry-run-placeholder'),
        baseUrl: destinations.groqBase,
        models: [env.GROQ_MODEL_EXPLAIN || 'openai/gpt-oss-120b'], // one model only
        timeoutMs: Number(env.GROQ_TIMEOUT_MS || 20000),
        retries: 0, // one HTTP attempt only
        maxCompletionTokens: Number(env.GROQ_MAX_COMPLETION_TOKENS || 1024),
        reasoningEffort: (env.GROQ_REASONING_EFFORT as 'low' | 'medium' | 'high') || 'low',
        temperature: Number(env.GROQ_TEMPERATURE || 0.2),
        fetchImpl,
      });
      const r = await groq.completeJson({
        system: SYSTEM_PROMPT,
        user: `Result JSON:\n${JSON.stringify(facts)}`,
        schemaName: 'explanation',
        schema: EXPLANATION_SCHEMA as unknown as Record<string, unknown>,
      });
      report.groq = r.ok
        ? {
            status: 'ok',
            model: r.model,
            attempts: r.attempts,
            usage: r.usage,
            validation: validateExplanation(r.content, facts),
          }
        : { status: 'error', reason: r.reason, model: r.model, attempts: r.attempts };
      report.groqContent = r.ok ? r.content : null;
    }
  } finally {
    const redis = ownedRedis as Redis | null;
    redis?.disconnect();
  }

  report.destinationGuard = guardEvents;
  if (!opts.live) report.requestsThatWouldBeSent = recorded;
  const safe = scrubDeep(report) as Record<string, unknown>;
  const text = JSON.stringify(safe, null, 2);
  if (opts.live) writeNew(path.join(out, `live-report-${stamp}.json`), text);
  else writeFileSync(path.join(out, 'dry-run-report.json'), text);
  return safe;
}

/** The Hive part. In a live run, exactly one unit is reserved in the app's Redis Hive budget
 * before the single HTTP attempt; HiveVerifier is then called without its own reservation
 * hook, so nothing is reserved twice. Denied or Redis unavailable: nothing is sent. The
 * reservation is never refunded, whatever the outcome. */
async function hiveCheck(
  opts: RunOptions,
  url: string,
  image: string,
  fetchImpl: FetchLike,
  ownRedis: (r: Redis) => void,
): Promise<Record<string, unknown>> {
  const { env } = opts;
  let budget = opts.budget ?? null;
  const result: Record<string, unknown> = {};

  if (opts.live) {
    if (!budget) {
      const cfg = loadConfig(env);
      const redis = new Redis(env.REDIS_URL || 'redis://localhost:6379', {
        maxRetriesPerRequest: 1,
        connectTimeout: 2000,
        enableOfflineQueue: false,
        lazyConnect: true,
      });
      redis.on('error', () => undefined); // reported below; never printed with secrets
      ownRedis(redis);
      budget = new DailyBudget(redis, 'hive', {
        daily: cfg.HIVE_DAILY_LIMIT,
        reserve: cfg.HIVE_DEMO_RESERVE,
        perUser: cfg.HIVE_USER_DAILY_LIMIT,
      });
      try {
        await redis.connect();
      } catch {
        return {
          status: 'not_sent',
          reason: 'budget_unavailable: Redis is not reachable',
          requestsSent: 0,
        };
      }
    }
    let reservation;
    try {
      reservation = await budget.reserve({ userId: M7_BUDGET_USER, demo: true });
    } catch {
      return {
        status: 'not_sent',
        reason: 'budget_unavailable: reservation failed',
        requestsSent: 0,
      };
    }
    result.reservation = reservation;
    if (!reservation.allowed) {
      return {
        ...result,
        status: 'not_sent',
        reason: `budget:${reservation.reason}`,
        requestsSent: 0,
      };
    }
  }

  const hive = new HiveVerifier({
    apiKey: env.HIVE_API_KEY || (opts.live ? '' : 'dry-run-placeholder'),
    url,
    timeoutMs: Number(env.HIVE_TIMEOUT_MS || 15000),
    retries: 0, // one HTTP attempt only
    fetchImpl,
  });
  // No reserveAttempt here: the single unit was reserved above.
  const r = await hive.verify(image, { mime: 'image/png' });
  if (r.rateLimited && budget) await budget.markExhausted().catch(() => undefined);
  Object.assign(result, {
    status: r.status,
    reason: r.reason ?? null,
    deepfakeScore: r.deepfakeScore,
    aiGeneratedScore: r.aiGeneratedScore,
    frames: r.frames?.length ?? 0,
    raw: r.raw,
  });
  if (budget) {
    const usage = await budget.usage(M7_BUDGET_USER).catch(() => null);
    result.budgetAfter = usage && {
      day: usage.day,
      used: usage.used,
      demoUsed: usage.demoUsed,
      exhausted: usage.exhausted,
      limits: usage.limits,
      pool: 'demo reserve (identity m7-live-check)',
    };
  }
  return result;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const gate = liveAllowed(argv, process.env);
  if (argv.includes('--live') && !gate.ok) {
    console.error(`Refusing to make live calls: ${gate.why}.`);
    process.exitCode = 2;
    return;
  }
  try {
    const report = await runCheck({
      live: gate.ok,
      env: process.env,
      allowRepeat: argv.includes('--allow-repeat'),
    });
    console.log(JSON.stringify(report, null, 2));
  } catch (err) {
    if (err instanceof RepeatRunRefused || err instanceof DestinationRefused) {
      console.error(`Refusing to make live calls: ${err.message}.`);
      process.exitCode = 2;
      return;
    }
    throw err;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  void main();
}
