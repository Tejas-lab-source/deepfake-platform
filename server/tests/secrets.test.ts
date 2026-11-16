import { Writable } from 'node:stream';
import { describe, expect, it } from 'vitest';
import { createLogger } from '../src/logger.js';
import { REDACTED, registerSecrets, scrub, scrubDeep } from '../src/secrets.js';

const HIVE_KEY = 'hive-test-key-0123456789abcdef';
const GROQ_KEY = 'gsk_testkey0123456789ABCDEFxyz';
registerSecrets([HIVE_KEY, GROQ_KEY]);

function capture(): { lines: string[]; stream: Writable } {
  const lines: string[] = [];
  const stream = new Writable({
    write(chunk, _enc, cb) {
      lines.push(String(chunk));
      cb();
    },
  });
  return { lines, stream };
}

describe('secret scrubbing', () => {
  it('removes registered secrets and credential-shaped tokens from text', () => {
    const text = `key=${HIVE_KEY} header "Bearer abcdefghijklmnop" groq ${GROQ_KEY} other gsk_another1234567890`;
    const out = scrub(text);
    expect(out).not.toContain(HIVE_KEY);
    expect(out).not.toContain(GROQ_KEY);
    expect(out).not.toContain('abcdefghijklmnop');
    expect(out).not.toContain('gsk_another1234567890');
    expect(out).toContain('Bearer [REDACTED]');
    expect(scrub('nothing secret here')).toBe('nothing secret here');
  });

  it('scrubs nested objects, drops credential-named fields and survives cycles', () => {
    const obj: Record<string, unknown> = {
      message: `failed with ${HIVE_KEY}`,
      headers: { Authorization: `Bearer ${GROQ_KEY}`, accept: 'json' },
      nested: [{ api_key: 'x', note: GROQ_KEY }],
      err: new Error(`boom ${HIVE_KEY}`),
    };
    obj.self = obj;
    const out = JSON.stringify(scrubDeep(obj));
    expect(out).not.toContain(HIVE_KEY);
    expect(out).not.toContain(GROQ_KEY);
    expect(out).toContain('"accept":"json"');
    expect(out).toContain(REDACTED);
    expect(out).toContain('[CIRCULAR]');
  });

  it('never lets a key reach the log, whatever field it is in', () => {
    const { lines, stream } = capture();
    const log = createLogger(stream, 'info');
    log.info({ headers: { authorization: `Bearer ${HIVE_KEY}` } }, 'request');
    log.error({ err: new Error(`upstream said ${GROQ_KEY}`) }, 'failed');
    log.warn(`message with ${HIVE_KEY} inline`);
    log.info({ config: { HIVE_API_KEY: HIVE_KEY, GROQ_API_KEY: GROQ_KEY } }, 'cfg');
    log.info({ detail: { deep: [{ text: `token ${GROQ_KEY}` }] } }, 'deep');
    const all = lines.join('');
    expect(lines).toHaveLength(5);
    expect(all).not.toContain(HIVE_KEY);
    expect(all).not.toContain(GROQ_KEY);
    expect(all).toContain('upstream said');
  });
});
