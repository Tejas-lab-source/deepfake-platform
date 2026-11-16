/**
 * M7 live check: one Hive request and one Groq request, with synthetic data only.
 *
 *   Dry run (default, no network):  npx tsx scripts/liveCheck.ts
 *   Live (needs approval):          M7_LIVE=1 npx tsx --env-file=live-check/.env.m7 \
 *                                     scripts/liveCheck.ts --live
 *
 * What is sent:
 *   Hive  a generated abstract image (no person, no face), uploaded as "upload" (image/png)
 *   Groq  the system prompt and the whitelisted result JSON built from
 *         live-check/synthetic-result.json, exactly as the app builds it
 * Nothing else: no user image, file name, path, personal data or analysis.
 *
 * Both calls go through the app's own clients, parser and validation. At most one billed
 * request per service (no retries, one Groq model). Outputs, scrubbed of secrets, go to
 * live-check/out/ (gitignored). Keys are never printed.
 */
import { mkdtempSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { deflateSync } from 'node:zlib';
import { buildResultJson, EXPLANATION_SCHEMA, SYSTEM_PROMPT } from '../src/explain/prompt.js';
import { validateExplanation } from '../src/explain/validate.js';
import type { ExplanationInput } from '../src/explainTemplate.js';
import type { FetchLike } from '../src/http.js';
import { GroqClient } from '../src/providers/groq.js';
import { HiveVerifier } from '../src/providers/hive.js';
import { scrubDeep, registerSecrets } from '../src/secrets.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const LIVE_CHECK_DIR = path.resolve(HERE, '../live-check');
const DEFAULT_HIVE_URL =
  'https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection';
const DEFAULT_GROQ_URL = 'https://api.groq.com/openai/v1';

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

export async function runCheck(opts: {
  live: boolean;
  env: NodeJS.ProcessEnv;
  fetchImpl?: FetchLike;
  outDir?: string;
}): Promise<Record<string, unknown>> {
  const { env } = opts;
  registerSecrets([env.HIVE_API_KEY ?? '', env.GROQ_API_KEY ?? '']);
  const recorded: RecordedRequest[] = [];
  const fetchImpl = opts.live ? opts.fetchImpl : dryRunFetch(recorded);

  // A fresh temporary folder holds the generated image; nothing from the app is read.
  const work = mkdtempSync(path.join(tmpdir(), 'm7-'));
  const image = path.join(work, 'synthetic.png');
  writeFileSync(image, syntheticPng());

  const report: Record<string, unknown> = {
    mode: opts.live ? 'LIVE' : 'DRY RUN (no network, simulated responses)',
  };

  if (!opts.live || env.HIVE_MODE === 'real') {
    const hive = new HiveVerifier({
      apiKey: env.HIVE_API_KEY || (opts.live ? '' : 'dry-run-placeholder'),
      url: env.HIVE_API_URL || DEFAULT_HIVE_URL,
      timeoutMs: Number(env.HIVE_TIMEOUT_MS || 15000),
      retries: 0,
      fetchImpl,
    });
    const r = await hive.verify(image, { mime: 'image/png' });
    report.hive = {
      status: r.status,
      reason: r.reason ?? null,
      deepfakeScore: r.deepfakeScore,
      aiGeneratedScore: r.aiGeneratedScore,
      frames: r.frames?.length ?? 0,
    };
    report.hiveRaw = r.raw;
  }

  if (!opts.live || env.GROQ_MODE === 'real') {
    const { input, extras } = syntheticResult();
    const facts = buildResultJson(input, extras);
    const groq = new GroqClient({
      apiKey: env.GROQ_API_KEY || (opts.live ? '' : 'dry-run-placeholder'),
      baseUrl: env.GROQ_BASE_URL || DEFAULT_GROQ_URL,
      models: [env.GROQ_MODEL_EXPLAIN || 'openai/gpt-oss-120b'],
      timeoutMs: Number(env.GROQ_TIMEOUT_MS || 20000),
      retries: 0,
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

  if (!opts.live) report.requestsThatWouldBeSent = recorded;
  const out = opts.outDir ?? path.join(LIVE_CHECK_DIR, 'out');
  mkdirSync(out, { recursive: true });
  const safe = scrubDeep(report) as Record<string, unknown>;
  writeFileSync(
    path.join(out, opts.live ? 'live-report.json' : 'dry-run-report.json'),
    JSON.stringify(safe, null, 2),
  );
  return safe;
}

async function main(): Promise<void> {
  const gate = liveAllowed(process.argv.slice(2), process.env);
  if (process.argv.includes('--live') && !gate.ok) {
    console.error(`Refusing to make live calls: ${gate.why}.`);
    process.exitCode = 2;
    return;
  }
  const report = await runCheck({ live: gate.ok, env: process.env });
  console.log(JSON.stringify(report, null, 2));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  void main();
}
