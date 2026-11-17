import bcrypt from 'bcryptjs';
import { Router } from 'express';
import { z } from 'zod';
import { config } from '../config.js';
import type { UserRecord } from '../data/types.js';
import type { AppDeps } from '../deps.js';
import { signTokens, verifyRefreshToken } from '../middleware/auth.js';
import { HttpProblem } from '../middleware/problem.js';
import { rateLimit } from '../rateLimit.js';

const credentials = z.object({
  email: z.string().trim().toLowerCase().email().max(254),
  password: z.string().min(8).max(128),
});

function session(user: UserRecord) {
  return {
    user: { id: user.id, email: user.email, role: user.role },
    ...signTokens({ id: user.id, role: user.role }),
  };
}

// Compared against when the email is unknown, so a login takes the same time either way.
const DUMMY_HASH = bcrypt.hashSync('timing-equaliser', 4);

export function authRouter(deps: AppDeps): Router {
  const { users } = deps.repos;
  const router = Router();
  router.use(rateLimit(deps.rateLimiters.auth, (req) => `ip:${req.ip ?? 'unknown'}`));

  router.post('/register', async (req, res) => {
    const { email, password } = credentials.parse(req.body);
    if (await users.findByEmail(email)) {
      throw new HttpProblem(409, 'Email already registered');
    }
    const passwordHash = await bcrypt.hash(password, config.BCRYPT_ROUNDS);
    const user = await users.create({ email, passwordHash });
    res.status(201).json(session(user));
  });

  router.post('/login', async (req, res) => {
    const { email, password } = credentials.parse(req.body);
    const user = await users.findByEmail(email);
    const ok = await bcrypt.compare(password, user?.passwordHash ?? DUMMY_HASH);
    if (!user || !ok) {
      throw new HttpProblem(401, 'Unauthorized', 'Wrong email or password');
    }
    res.json(session(user));
  });

  router.post('/refresh', async (req, res) => {
    const { refreshToken } = z
      .object({ refreshToken: z.string().min(1).max(4096) })
      .parse(req.body);
    const claims = verifyRefreshToken(refreshToken);
    const user = claims ? await users.findById(claims.id) : null;
    if (!user) throw new HttpProblem(401, 'Unauthorized', 'Invalid or expired refresh token');
    res.json(session(user));
  });

  return router;
}
