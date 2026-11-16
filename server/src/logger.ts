import { destination as pinoDestination, pino, type DestinationStream, type Logger } from 'pino';
import { config } from './config.js';
import { scrub } from './secrets.js';

// Field paths that may carry credentials. pino replaces their values with [REDACTED].
export const REDACT_PATHS = [
  'req.headers.authorization',
  'req.headers.cookie',
  'headers.authorization',
  'headers.Authorization',
  '*.headers.authorization',
  '*.headers.Authorization',
  '*.authorization',
  '*.Authorization',
  'apiKey',
  '*.apiKey',
  'HIVE_API_KEY',
  'GROQ_API_KEY',
  '*.HIVE_API_KEY',
  '*.GROQ_API_KEY',
];

/** Build a logger. Every line is scrubbed of configured secrets and credential-shaped tokens
 * just before it is written, so a key in an error message or stack trace never reaches the
 * log, whatever field it is in. */
export function createLogger(destination?: DestinationStream, level = config.LOG_LEVEL): Logger {
  const target: DestinationStream = destination ?? pinoDestination(1);
  const scrubbing: DestinationStream = {
    write(line: string) {
      target.write(scrub(line));
    },
  };
  return pino({ level, redact: { paths: REDACT_PATHS, censor: '[REDACTED]' } }, scrubbing);
}

export const logger = createLogger();
