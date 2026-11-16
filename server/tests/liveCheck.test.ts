import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { inflateSync } from 'node:zlib';
import { describe, expect, it, vi } from 'vitest';
import { liveAllowed, runCheck, syntheticPng, type RecordedRequest } from '../scripts/liveCheck.js';

describe('M7 live-check preparation (no network)', () => {
  it('generates a valid, deterministic, abstract PNG', () => {
    const a = syntheticPng(64);
    expect(a.subarray(0, 8).toString('hex')).toBe('89504e470d0a1a0a');
    expect(a.readUInt32BE(16)).toBe(64); // width
    expect(a.readUInt32BE(20)).toBe(64); // height
    expect(a.equals(syntheticPng(64))).toBe(true);
    const idatStart = a.indexOf('IDAT') + 4;
    const idatLen = a.readUInt32BE(idatStart - 8);
    expect(inflateSync(a.subarray(idatStart, idatStart + idatLen)).length).toBe((64 * 3 + 1) * 64);
  });

  it('refuses live calls without both --live and M7_LIVE=1', () => {
    const real = { HIVE_MODE: 'real', GROQ_MODE: 'real' };
    expect(liveAllowed([], { ...real, M7_LIVE: '1' }).ok).toBe(false);
    expect(liveAllowed(['--live'], { ...real, M7_LIVE: '0' }).ok).toBe(false);
    expect(liveAllowed(['--live'], { M7_LIVE: '1', HIVE_MODE: 'mock', GROQ_MODE: 'off' }).ok).toBe(
      false,
    );
    expect(liveAllowed(['--live'], { ...real, M7_LIVE: '1' }).ok).toBe(true);
  });

  it('dry run makes no network call and sends only synthetic, minimal data', async () => {
    const networkFetch = vi.spyOn(globalThis, 'fetch');
    const out = mkdtempSync(path.join(tmpdir(), 'm7-test-'));
    const report = await runCheck({
      live: false,
      env: { GROQ_API_KEY: 'gsk_dryRunPlaceholder0123456789' },
      outDir: out,
    });
    expect(networkFetch).not.toHaveBeenCalled();
    networkFetch.mockRestore();

    const requests = report.requestsThatWouldBeSent as RecordedRequest[];
    expect(requests.map((r) => r.service)).toEqual(['hive', 'groq']);
    const [hive, groq] = requests as [RecordedRequest, RecordedRequest];

    // Hive: one multipart field, neutral name, PNG only, and the official endpoint.
    expect(hive.url).toBe(
      'https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection',
    );
    expect(hive.formFields).toEqual([
      { name: 'media', fileName: 'upload', type: 'image/png', bytes: expect.any(Number) },
    ]);
    expect(hive.headerNames).toEqual(['accept', 'authorization']);

    // Groq: system prompt + whitelisted synthetic facts only.
    expect(groq.url).toBe('https://api.groq.com/openai/v1/chat/completions');
    const messages = (groq.jsonBody as { messages: Array<{ role: string; content: string }> })
      .messages;
    expect(messages.map((m) => m.role)).toEqual(['system', 'user']);
    const facts = JSON.parse(messages[1]!.content.replace(/^Result JSON:\n/, ''));
    expect(Object.keys(facts).sort()).toEqual([
      'checksDisagree',
      'finalVerdict',
      'mockMode',
      'modelCardVersion',
      'ourPipeline',
      'ruleApplied',
      'secondCheck',
      'verificationBadge',
    ]);
    const sent = JSON.stringify(requests);
    for (const forbidden of [tmpdir(), 'synthetic.png', '/data/uploads', '@', 'sha256', 'base64']) {
      expect(sent).not.toContain(forbidden);
    }
    // Parsing and validation ran on the simulated documented responses.
    expect(report.hive).toMatchObject({ status: 'ok', deepfakeScore: 0.03 });
    expect(report.groq).toMatchObject({ status: 'ok', validation: { ok: true } });
    // The placeholder key never appears in the report.
    expect(JSON.stringify(report)).not.toContain('gsk_dryRunPlaceholder0123456789');
  });
});
