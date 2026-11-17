import type {
  Analysis,
  AnalysisListItem,
  AskResponse,
  Comparison,
  EvalReport,
  ModelCard,
  Page,
  Session,
} from './types';

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

function problemMessage(status: number, body: unknown): string {
  const problem = body as { detail?: string; title?: string } | null;
  return problem?.detail ?? problem?.title ?? `HTTP ${status}`;
}

async function parse<T>(res: Response): Promise<T> {
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }
  if (!res.ok) throw new ApiError(res.status, problemMessage(res.status, body));
  return body as T;
}

let refreshing: Promise<boolean> | null = null;

/** Exchange the refresh token for new tokens; signs out when that is no longer possible. */
export function refresh(): Promise<boolean> {
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

async function send(path: string, init: RequestInit = {}): Promise<Response> {
  const go = (): Promise<Response> =>
    fetch(BASE + path, {
      ...init,
      headers: {
        ...init.headers,
        ...(session ? { authorization: `Bearer ${session.accessToken}` } : {}),
      },
    });
  let res = await go();
  if (res.status === 401 && session && (await refresh())) res = await go();
  return res;
}

const request = async <T>(path: string, init: RequestInit = {}): Promise<T> =>
  parse<T>(await send(path, init));

async function blob(path: string): Promise<Blob> {
  const res = await send(path);
  if (!res.ok) await parse(res);
  return res.blob();
}

const json = (body: unknown, method = 'POST'): RequestInit => ({
  method,
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body),
});

/** Multipart upload with progress (fetch cannot report upload progress). */
function uploadWithProgress(
  form: FormData,
  onProgress?: (fraction: number) => void,
): Promise<{ analysisId: string; jobId: string; duplicate?: boolean }> {
  const attempt = (): Promise<{ status: number; body: unknown }> =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${BASE}/analyses`);
      if (session) xhr.setRequestHeader('authorization', `Bearer ${session.accessToken}`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        let body: unknown = null;
        try {
          body = xhr.responseText ? JSON.parse(xhr.responseText) : null;
        } catch {
          body = null;
        }
        resolve({ status: xhr.status, body });
      };
      xhr.onerror = () =>
        reject(new ApiError(0, 'Network error: the upload did not reach the server.'));
      xhr.send(form);
    });
  return (async () => {
    let r = await attempt();
    if (r.status === 401 && session && (await refresh())) r = await attempt();
    if (r.status < 200 || r.status >= 300)
      throw new ApiError(r.status, problemMessage(r.status, r.body));
    return r.body as { analysisId: string; jobId: string; duplicate?: boolean };
  })();
}

export const api = {
  async authenticate(mode: 'login' | 'register', email: string, password: string): Promise<void> {
    setSession(
      await parse<Session>(await fetch(`${BASE}/auth/${mode}`, json({ email, password }))),
    );
  },
  health: () => request<{ status: string; mockMode: boolean }>('/health'),
  createAnalysis(file: File, externalCheck: boolean, onProgress?: (fraction: number) => void) {
    const form = new FormData();
    form.append('externalCheck', String(externalCheck));
    form.append('file', file);
    return uploadWithProgress(form, onProgress);
  },
  getAnalysis: (id: string) => request<Analysis>(`/analyses/${id}`),
  listAnalyses: (page: number, limit = 10) =>
    request<Page<AnalysisListItem>>(`/analyses?page=${page}&limit=${limit}`),
  async deleteAnalysis(id: string): Promise<void> {
    const res = await send(`/analyses/${id}`, { method: 'DELETE' });
    if (!res.ok) await parse(res);
  },
  /** A derived image (heatmap, crop) of an analysis, as a Blob (needs the auth header). */
  image: (id: string, key: string) =>
    blob(`/analyses/${id}/heatmaps/${encodeURIComponent(key.replace(/^derived\//, ''))}`),
  report: (id: string) => blob(`/analyses/${id}/report.pdf`),
  ask: (id: string, question: string) =>
    request<AskResponse>(`/analyses/${id}/ask`, json({ question })),
  compare: (id: string, family: string, strength: number) =>
    request<Comparison>(`/analyses/${id}/compare`, json({ family, strength })),
  comparisons: (id: string) => request<{ items: Comparison[] }>(`/analyses/${id}/comparisons`),
  modelCard: () => request<ModelCard>('/model-card'),
  evalRuns: () => request<{ report: EvalReport | null }>('/eval-runs'),
};

/** Save a Blob as a file download. */
export function saveBlob(data: Blob, filename: string): void {
  const url = URL.createObjectURL(data);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
