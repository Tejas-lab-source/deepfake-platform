import type { ErrorRequestHandler, RequestHandler } from 'express';
import multer from 'multer';
import { ZodError } from 'zod';
import { logger } from '../logger.js';

/** An error that maps to an RFC 7807 problem+json response. */
export class HttpProblem extends Error {
  constructor(
    readonly status: number,
    readonly title: string,
    detail?: string,
  ) {
    super(detail ?? title);
  }
}

export const notFound: RequestHandler = (_req, _res, next) => {
  next(new HttpProblem(404, 'Not Found'));
};

export const problemHandler: ErrorRequestHandler = (err, req, res, _next) => {
  let problem: HttpProblem;
  if (err instanceof HttpProblem) {
    problem = err;
  } else if (err instanceof ZodError) {
    const detail = err.issues.map((i) => `${i.path.join('.') || 'body'}: ${i.message}`).join('; ');
    problem = new HttpProblem(400, 'Invalid request', detail);
  } else if (err instanceof multer.MulterError) {
    problem =
      err.code === 'LIMIT_FILE_SIZE'
        ? new HttpProblem(413, 'File too large', err.message)
        : new HttpProblem(400, 'Invalid upload', err.message);
  } else {
    logger.error({ err, path: req.path }, 'unhandled error');
    problem = new HttpProblem(500, 'Internal Server Error');
  }
  res.status(problem.status).type('application/problem+json').json({
    type: 'about:blank',
    title: problem.title,
    status: problem.status,
    detail: problem.message,
    instance: req.originalUrl,
  });
};
