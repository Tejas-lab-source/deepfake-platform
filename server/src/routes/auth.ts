import bcrypt from 'bcryptjs';
import { Router } from 'express';
import { z } from 'zod';
import { config } from '../config.js';
import { signTokens, verifyRefreshToken, type Role } from '../middleware/auth.js';
import { HttpProblem } from '../middleware/problem.js';
import { User } from '../models/index.js';

const credentials = z.object({
  email: z.string().trim().toLowerCase().email().max(254),
  password: z.string().min(8).max(128),
});

function session(user: { _id: unknown; email: string; role: string }) {
  const id = String(user._id);
  const role = user.role as Role;
  return { user: { id, email: user.email, role }, ...signTokens({ id, role }) };
}

export const authRouter = Router();

authRouter.post('/register', async (req, res) => {
  const { email, password } = credentials.parse(req.body);
  if (await User.exists({ email })) {
    throw new HttpProblem(409, 'Email already registered');
  }
  const passwordHash = await bcrypt.hash(password, config.BCRYPT_ROUNDS);
  const user = await User.create({ email, passwordHash });
  res.status(201).json(session(user));
});

authRouter.post('/login', async (req, res) => {
  const { email, password } = credentials.parse(req.body);
  const user = await User.findOne({ email });
  if (!user || !(await bcrypt.compare(password, user.passwordHash))) {
    throw new HttpProblem(401, 'Unauthorized', 'Wrong email or password');
  }
  res.json(session(user));
});

authRouter.post('/refresh', async (req, res) => {
  const { refreshToken } = z.object({ refreshToken: z.string().min(1) }).parse(req.body);
  const claims = verifyRefreshToken(refreshToken);
  const user = claims ? await User.findById(claims.id) : null;
  if (!user) throw new HttpProblem(401, 'Unauthorized', 'Invalid or expired refresh token');
  res.json(session(user));
});
