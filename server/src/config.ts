import { z } from 'zod';

const bool = z.enum(['true', 'false']).transform((v) => v === 'true');
const mode = z.enum(['mock', 'real', 'off']);
export type ProviderMode = z.infer<typeof mode>;

const DEV_DEFAULT = /change-me$/;

const schema = z.object({
  NODE_ENV: z.string().default('development'),
  PORT: z.coerce.number().int().default(4000),
  LOG_LEVEL: z.string().default('info'),
  MOCK_MODE: bool.default('true'),

  MONGO_URL: z.string().default('mongodb://localhost:27017/deepfake'),
  REDIS_URL: z.string().default('redis://localhost:6379'),
  ML_SERVICE_URL: z.string().default('http://localhost:8000'),
  ML_TIMEOUT_MS: z.coerce.number().int().default(300_000),
  INTERNAL_SECRET: z.string().min(8).default('dev-internal-secret-change-me'),

  JWT_ACCESS_SECRET: z.string().min(8).default('dev-access-secret-change-me'),
  JWT_REFRESH_SECRET: z.string().min(8).default('dev-refresh-secret-change-me'),
  ACCESS_TTL_SEC: z.coerce
    .number()
    .int()
    .default(15 * 60),
  REFRESH_TTL_SEC: z.coerce
    .number()
    .int()
    .default(7 * 24 * 3600),
  BCRYPT_ROUNDS: z.coerce.number().int().default(10),

  UPLOAD_DIR: z.string().default('./uploads'),
  MAX_IMAGE_MB: z.coerce.number().default(10),
  MAX_VIDEO_MB: z.coerce.number().default(100),
  MEDIA_TTL_HOURS: z.coerce.number().default(24),
  K: z.coerce.number().int().min(0).default(4),

  // External second check (Hive v3). Mode defaults from MOCK_MODE: mock when true, real when
  // false. "off" disables it. A real call also needs the user's per-upload consent.
  HIVE_MODE: mode.optional(),
  HIVE_API_KEY: z.string().default(''),
  HIVE_API_URL: z
    .string()
    .url()
    .default('https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection'),
  HIVE_THRESHOLD: z.coerce.number().min(0).max(1).default(0.9),
  HIVE_TIMEOUT_MS: z.coerce.number().int().positive().default(15_000),
  // Hive v3's documented default limit is 100 requests a day. Never assume it is free.
  HIVE_DAILY_LIMIT: z.coerce.number().int().min(0).default(100),
  // Part of the daily limit that only the demo pool (admin role) may use.
  HIVE_DEMO_RESERVE: z.coerce.number().int().min(0).default(20),
  HIVE_USER_DAILY_LIMIT: z.coerce.number().int().min(0).default(10),
  // Every attempt is a billable request, so no retries unless deliberately configured.
  HIVE_RETRIES: z.coerce.number().int().min(0).max(3).default(0),

  // Explanations (Groq, OpenAI-compatible). Mode defaults from MOCK_MODE like HIVE_MODE.
  GROQ_MODE: mode.optional(),
  GROQ_API_KEY: z.string().default(''),
  GROQ_BASE_URL: z.string().url().default('https://api.groq.com/openai/v1'),
  GROQ_MODEL_EXPLAIN: z.string().min(1).default('openai/gpt-oss-120b'),
  GROQ_MODEL_CHAT: z.string().min(1).default('openai/gpt-oss-20b'),
  // Tried in order when the primary model is unavailable (Groq retires models at short notice).
  GROQ_MODEL_FALLBACKS: z.string().default('openai/gpt-oss-20b'),
  GROQ_TIMEOUT_MS: z.coerce.number().int().positive().default(20_000),
  GROQ_RETRIES: z.coerce.number().int().min(0).max(5).default(2),
  GROQ_DAILY_LIMIT: z.coerce.number().int().min(0).default(900),
  GROQ_MAX_COMPLETION_TOKENS: z.coerce.number().int().positive().default(1024),
  GROQ_REASONING_EFFORT: z.enum(['low', 'medium', 'high']).default('low'),
  GROQ_TEMPERATURE: z.coerce.number().min(0).max(2).default(0.2),

  WORKER_CONCURRENCY: z.coerce.number().int().default(2),
});

type Parsed = z.infer<typeof schema>;

export type Config = Omit<Parsed, 'HIVE_MODE' | 'GROQ_MODE' | 'GROQ_MODEL_FALLBACKS'> & {
  HIVE_MODE: ProviderMode;
  GROQ_MODE: ProviderMode;
  GROQ_MODEL_FALLBACKS: string[];
};

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  // An empty variable means "not set": Docker Compose passes unset `${VAR:-}` values as "",
  // which would otherwise become 0 for numbers and fail for enums.
  const present = Object.fromEntries(Object.entries(env).filter(([, v]) => v !== ''));
  const parsed = schema.parse(present);
  const fallbackMode: ProviderMode = parsed.MOCK_MODE ? 'mock' : 'real';
  const cfg: Config = {
    ...parsed,
    HIVE_MODE: parsed.HIVE_MODE ?? fallbackMode,
    GROQ_MODE: parsed.GROQ_MODE ?? fallbackMode,
    GROQ_MODEL_FALLBACKS: parsed.GROQ_MODEL_FALLBACKS.split(',')
      .map((m) => m.trim())
      .filter((m) => m.length > 0),
  };
  if (cfg.HIVE_DEMO_RESERVE > cfg.HIVE_DAILY_LIMIT) {
    throw new Error('HIVE_DEMO_RESERVE cannot exceed HIVE_DAILY_LIMIT');
  }
  if (cfg.NODE_ENV === 'production') {
    for (const key of ['INTERNAL_SECRET', 'JWT_ACCESS_SECRET', 'JWT_REFRESH_SECRET'] as const) {
      if (DEV_DEFAULT.test(cfg[key])) {
        throw new Error(`${key} still has its development default; set it in .env`);
      }
    }
  }
  return cfg;
}

/** Non-fatal configuration problems, logged at startup. Never contains a secret value. */
export function configWarnings(cfg: Config): string[] {
  const warnings: string[] = [];
  if (cfg.HIVE_MODE === 'real' && !cfg.HIVE_API_KEY) {
    warnings.push('HIVE_MODE=real but HIVE_API_KEY is empty: second checks will be not_configured');
  }
  if (cfg.GROQ_MODE === 'real' && !cfg.GROQ_API_KEY) {
    warnings.push('GROQ_MODE=real but GROQ_API_KEY is empty: explanations will use the template');
  }
  if (cfg.HIVE_MODE === 'real' && cfg.HIVE_RETRIES > 0) {
    warnings.push(`HIVE_RETRIES=${cfg.HIVE_RETRIES}: every retry is a separately billed request`);
  }
  return warnings;
}

/** Values that must never appear in logs, responses or stored error messages. */
export function secretValues(cfg: Config): string[] {
  return [
    cfg.HIVE_API_KEY,
    cfg.GROQ_API_KEY,
    cfg.INTERNAL_SECRET,
    cfg.JWT_ACCESS_SECRET,
    cfg.JWT_REFRESH_SECRET,
  ].filter((v) => v.length >= 8);
}

export const config = loadConfig();
