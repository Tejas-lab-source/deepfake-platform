import cors from 'cors';
import express from 'express';
import { pinoHttp } from 'pino-http';
import { config } from './config.js';
import { logger } from './logger.js';
import { notFound, problemHandler } from './middleware/problem.js';
import { analysesRouter } from './routes/analyses.js';
import { authRouter } from './routes/auth.js';

export function createApp(): express.Express {
  const app = express();
  app.disable('x-powered-by');
  app.use(cors());
  app.use(pinoHttp({ logger, autoLogging: { ignore: (req) => req.url === '/api/v1/health' } }));
  app.use(express.json({ limit: '100kb' }));

  const api = express.Router();
  api.get('/health', (_req, res) => {
    res.json({ status: 'ok', mockMode: config.MOCK_MODE });
  });
  api.use('/auth', authRouter);
  api.use('/analyses', analysesRouter);

  app.use('/api/v1', api);
  app.use(notFound);
  app.use(problemHandler);
  return app;
}
