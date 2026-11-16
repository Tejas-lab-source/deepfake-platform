import { describe, expect, it } from 'vitest';
import { configWarnings, loadConfig, secretValues } from '../src/config.js';

const base = { NODE_ENV: 'test' };

describe('config: Phase 4 settings', () => {
  it('derives provider modes from MOCK_MODE, and each can be set on its own', () => {
    expect(loadConfig({ ...base, MOCK_MODE: 'true' })).toMatchObject({
      HIVE_MODE: 'mock',
      GROQ_MODE: 'mock',
    });
    expect(loadConfig({ ...base, MOCK_MODE: 'false' })).toMatchObject({
      HIVE_MODE: 'real',
      GROQ_MODE: 'real',
    });
    const mixed = loadConfig({ ...base, MOCK_MODE: 'true', HIVE_MODE: 'off', GROQ_MODE: 'real' });
    expect(mixed).toMatchObject({ HIVE_MODE: 'off', GROQ_MODE: 'real' });
    expect(() => loadConfig({ ...base, HIVE_MODE: 'live' })).toThrow();
  });

  it('has the approved budget and Groq defaults', () => {
    const cfg = loadConfig(base);
    expect(cfg).toMatchObject({
      HIVE_API_URL:
        'https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection',
      HIVE_TIMEOUT_MS: 15_000,
      HIVE_DAILY_LIMIT: 100,
      HIVE_DEMO_RESERVE: 20,
      HIVE_USER_DAILY_LIMIT: 10,
      HIVE_RETRIES: 0,
      GROQ_BASE_URL: 'https://api.groq.com/openai/v1',
      GROQ_MODEL_EXPLAIN: 'openai/gpt-oss-120b',
      GROQ_MODEL_FALLBACKS: ['openai/gpt-oss-20b'],
      GROQ_REASONING_EFFORT: 'low',
      GROQ_TEMPERATURE: 0.2,
      GROQ_MAX_COMPLETION_TOKENS: 1024,
    });
    expect(loadConfig({ ...base, GROQ_MODEL_FALLBACKS: ' a , ,b ' }).GROQ_MODEL_FALLBACKS).toEqual([
      'a',
      'b',
    ]);
  });

  it('refuses a demo reserve larger than the daily limit', () => {
    expect(() => loadConfig({ ...base, HIVE_DAILY_LIMIT: '10', HIVE_DEMO_RESERVE: '11' })).toThrow(
      /HIVE_DEMO_RESERVE/,
    );
  });

  it('warns about real mode without keys and about billed retries, without printing keys', () => {
    const cfg = loadConfig({ ...base, HIVE_MODE: 'real', GROQ_MODE: 'real', HIVE_RETRIES: '1' });
    const warnings = configWarnings(cfg);
    expect(warnings).toHaveLength(3);
    const withKeys = loadConfig({
      ...base,
      HIVE_MODE: 'real',
      GROQ_MODE: 'real',
      HIVE_API_KEY: 'hive-secret-value-123',
      GROQ_API_KEY: 'gsk_secretvalue1234567',
    });
    expect(configWarnings(withKeys)).toEqual([]);
    expect(JSON.stringify(configWarnings(withKeys))).not.toContain('secret');
  });

  it('lists the values that must be scrubbed', () => {
    const cfg = loadConfig({
      ...base,
      HIVE_API_KEY: 'hive-secret-value-123',
      GROQ_API_KEY: 'short',
    });
    const secrets = secretValues(cfg);
    expect(secrets).toContain('hive-secret-value-123');
    expect(secrets).not.toContain('short'); // too short to scrub safely
    expect(secrets).toContain(cfg.JWT_ACCESS_SECRET);
  });

  it('treats empty variables as unset, so defaults apply instead of 0 or a crash', () => {
    const cfg = loadConfig({
      ...base,
      HIVE_MODE: '',
      GROQ_MODE: '',
      HIVE_DAILY_LIMIT: '',
      HIVE_DEMO_RESERVE: '',
      GROQ_TEMPERATURE: '',
      HIVE_API_KEY: '',
    });
    expect(cfg).toMatchObject({
      HIVE_MODE: 'mock',
      GROQ_MODE: 'mock',
      HIVE_DAILY_LIMIT: 100,
      HIVE_DEMO_RESERVE: 20,
      GROQ_TEMPERATURE: 0.2,
      HIVE_API_KEY: '',
    });
    expect(() =>
      loadConfig({ NODE_ENV: 'production', JWT_ACCESS_SECRET: '', INTERNAL_SECRET: '' }),
    ).toThrow(/development default/);
  });
});
