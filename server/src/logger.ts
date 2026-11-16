import { pino } from 'pino';
import { config } from './config.js';

export const logger = pino({
  level: config.LOG_LEVEL,
  // Never write tokens to the logs.
  redact: ['req.headers.authorization', 'req.headers.cookie'],
});
