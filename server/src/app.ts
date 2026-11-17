import cors from 'cors';
import express, { type RequestHandler } from 'express';
import { pinoHttp } from 'pino-http';
import { config } from './config.js';
import type { AppDeps } from './deps.js';
import { logger } from './logger.js';
import { notFound, problemHandler } from './middleware/problem.js';
import { analysesRouter } from './routes/analyses.js';
import { authRouter } from './routes/auth.js';
import { metaRouter } from './routes/meta.js';

/** Conservative headers for a JSON API that also serves PNG and PDF downloads. */
const securityHeaders: RequestHandler = (_req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  res.setHeader('Content-Security-Policy', "default-src 'none'; frame-ancestors 'none'");
  res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  next();
};

const noStore: RequestHandler = (_req, res, next) => {
  res.setHeader('Cache-Control', 'no-store');
  next();
};

/** Small JSON bodies everywhere, except the evaluation import, which parses its own. */
const jsonBody: RequestHandler = (req, res, next) =>
  req.path === '/api/v1/eval-runs' ? next() : express.json({ limit: '100kb' })(req, res, next);

export function createApp(deps: AppDeps): express.Express {
  const app = express();
  app.disable('x-powered-by');
  app.set('trust proxy', config.TRUST_PROXY);
  app.use(securityHeaders);
  // Same-origin by default: the web container proxies the API, so the browser needs no CORS.
  if (deps.corsOrigins.length > 0) {
    app.use(cors({ origin: deps.corsOrigins, credentials: false, maxAge: 600 }));
  }
  app.use(
    pinoHttp({
      logger,
      autoLogging: { ignore: (req) => req.url === '/api/v1/health' },
      // Never log tokens, cookies or query strings.
      serializers: {
        req: (req: { method: string; url: string; id: unknown }) => ({
          id: req.id,
          method: req.method,
          path: req.url.split('?')[0],
        }),
        res: (res: { statusCode: number }) => ({ statusCode: res.statusCode }),
      },
    }),
  );
  app.use(jsonBody);

  const api = express.Router();
  api.use(noStore);
  api.get('/health', (_req, res) => {
    res.json({ status: 'ok', mockMode: deps.mockMode });
  });
  api.use('/auth', authRouter(deps));
  api.use('/analyses', analysesRouter(deps));
  api.use('/', metaRouter(deps));

  app.use('/api/v1', api);
  app.use(notFound);
  app.use(problemHandler);
  return app;
}
