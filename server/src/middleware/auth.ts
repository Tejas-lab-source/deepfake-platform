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
export const verifyRefreshToken = (token: string): AuthUser | null =>
  verify(token, config.JWT_REFRESH_SECRET, 'refresh');

export const requireAuth: RequestHandler = (req, _res, next) => {
  const header = req.headers.authorization;
  const user = header?.startsWith('Bearer ') ? verifyAccessToken(header.slice(7)) : null;
  if (!user) return next(new HttpProblem(401, 'Unauthorized', 'Missing or invalid access token'));
  req.user = user;
  next();
};
