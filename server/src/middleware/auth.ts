import type { RequestHandler } from 'express';
import jwt from 'jsonwebtoken';
import { config } from '../config.js';
import { HttpProblem } from './problem.js';

export type Role = 'user' | 'admin';
export interface AuthUser {
  id: string;
  role: Role;
}

declare module 'express-serve-static-core' {
  interface Request {
    user?: AuthUser;
  }
}

interface Claims {
  sub: string;
  role: Role;
  typ: 'access' | 'refresh';
}

export function signTokens(user: AuthUser): { accessToken: string; refreshToken: string } {
  const claims = { sub: user.id, role: user.role };
  return {
    accessToken: jwt.sign({ ...claims, typ: 'access' }, config.JWT_ACCESS_SECRET, {
      expiresIn: config.ACCESS_TTL_SEC,
    }),
    refreshToken: jwt.sign({ ...claims, typ: 'refresh' }, config.JWT_REFRESH_SECRET, {
      expiresIn: config.REFRESH_TTL_SEC,
    }),
  };
}

function verify(token: string, secret: string, typ: Claims['typ']): AuthUser | null {
  try {
    const claims = jwt.verify(token, secret, { algorithms: ['HS256'] }) as Claims;
    return claims.typ === typ ? { id: claims.sub, role: claims.role } : null;
  } catch {
    return null;
  }
}

export const verifyAccessToken = (token: string): AuthUser | null =>
  verify(token, config.JWT_ACCESS_SECRET, 'access');

/** Like verifyAccessToken, but says why a token was refused (for clear client errors). */
export function checkAccessToken(
  token: unknown,
): { ok: true; user: AuthUser } | { ok: false; reason: 'missing' | 'expired' | 'invalid' } {
  if (typeof token !== 'string' || token.length === 0) return { ok: false, reason: 'missing' };
  try {
    const claims = jwt.verify(token, config.JWT_ACCESS_SECRET, { algorithms: ['HS256'] }) as Claims;
    if (claims.typ !== 'access') return { ok: false, reason: 'invalid' };
    return { ok: true, user: { id: claims.sub, role: claims.role } };
  } catch (err) {
    return { ok: false, reason: err instanceof jwt.TokenExpiredError ? 'expired' : 'invalid' };
  }
}
export const verifyRefreshToken = (token: string): AuthUser | null =>
  verify(token, config.JWT_REFRESH_SECRET, 'refresh');

export const requireAuth: RequestHandler = (req, _res, next) => {
  const header = req.headers.authorization;
  const check = checkAccessToken(header?.startsWith('Bearer ') ? header.slice(7) : undefined);
  if (!check.ok) {
    const detail =
      check.reason === 'expired' ? 'Access token expired' : 'Missing or invalid access token';
    return next(new HttpProblem(401, 'Unauthorized', detail));
  }
  req.user = check.user;
  next();
};
