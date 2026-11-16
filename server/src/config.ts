import { z } from 'zod';

const bool = z.enum(['true', 'false']).transform((v) => v === 'true');

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

  HIVE_API_KEY: z.string().default(''),
  HIVE_THRESHOLD: z.coerce.number().min(0).max(1).default(0.9),
  HIVE_TIMEOUT_MS: z.coerce.number().int().default(15_000),

  GROQ_API_KEY: z.string().default(''),
  GROQ_MODEL_EXPLAIN: z.string().default('openai/gpt-oss-120b'),
  GROQ_MODEL_CHAT: z.string().default('openai/gpt-oss-20b'),

  WORKER_CONCURRENCY: z.coerce.number().int().default(2),
});

export type Config = z.infer<typeof schema>;

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const cfg = schema.parse(env);
  if (cfg.NODE_ENV === 'production') {
    for (const key of ['INTERNAL_SECRET', 'JWT_ACCESS_SECRET', 'JWT_REFRESH_SECRET'] as const) {
      if (DEV_DEFAULT.test(cfg[key])) {
        throw new Error(`${key} still has its development default; set it in .env`);
      }
    }
  }
  return cfg;
}

export const config = loadConfig();
