import type { Analysis, Session } from './types';

const BASE = '/api/v1';
const STORAGE_KEY = 'deepfake.session';

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

let session: Session | null = load();
const listeners = new Set<(s: Session | null) => void>();

function load(): Session | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}

export function getSession(): Session | null {
  return session;
}

export function setSession(next: Session | null): void {
  session = next;
  try {
    if (next) localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    // storage unavailable: the session lasts until the tab closes
  }
  listeners.forEach((fn) => fn(next));
}

export function onSessionChange(fn: (s: Session | null) => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

async function parse<T>(res: Response): Promise<T> {
  const text = await res.text();
  const body: unknown = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const problem = body as { detail?: string; title?: string } | null;
    throw new ApiError(res.status, problem?.detail ?? problem?.title ?? `HTTP ${res.status}`);
  }
  return body as T;
}

let refreshing: Promise<boolean> | null = null;

function refresh(): Promise<boolean> {
  refreshing ??= (async () => {
    try {
      if (!session) return false;
      const res = await fetch(`${BASE}/auth/refresh`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ refreshToken: session.refreshToken }),
      });
      setSession(res.ok ? ((await res.json()) as Session) : null);
      return res.ok;
    } catch {
      return false;
    } finally {
      refreshing = null;
    }
  })();
  return refreshing;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const send = (): Promise<Response> =>
    fetch(BASE + path, {
      ...init,
      headers: {
        ...init.headers,
        ...(session ? { authorization: `Bearer ${session.accessToken}` } : {}),
      },
    });
  let res = await send();
  if (res.status === 401 && session && (await refresh())) res = await send();
  return parse<T>(res);
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body),
});

export const api = {
  async authenticate(mode: 'login' | 'register', email: string, password: string): Promise<void> {
    setSession(
      await parse<Session>(await fetch(`${BASE}/auth/${mode}`, json({ email, password }))),
    );
  },
  health: () => request<{ status: string; mockMode: boolean }>('/health'),
  createAnalysis(file: File, externalCheck: boolean) {
    const form = new FormData();
    form.append('file', file);
    form.append('externalCheck', String(externalCheck));
    return request<{ analysisId: string; jobId: string }>('/analyses', {
      method: 'POST',
      body: form,
    });
  },
  getAnalysis: (id: string) => request<Analysis>(`/analyses/${id}`),
};
