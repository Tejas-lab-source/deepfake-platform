import { useEffect, useState } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { api, getSession, onSessionChange, setSession } from './api/client';
import Detect from './pages/Detect';
import Login from './pages/Login';

export default function App() {
  const [session, setLocal] = useState(getSession());
  const [mockMode, setMockMode] = useState(false);

  useEffect(() => onSessionChange(setLocal), []);
  useEffect(() => {
    api
      .health()
      .then((h) => setMockMode(h.mockMode))
      .catch(() => undefined);
  }, []);

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
          <span className="font-semibold">Deepfake check</span>
          {session && (
            <div className="flex items-center gap-3 text-sm text-slate-600">
              <span className="hidden sm:inline">{session.user.email}</span>
              <button
                className="rounded border border-slate-300 px-3 py-1 hover:bg-slate-100"
                onClick={() => setSession(null)}
              >
                Log out
              </button>
            </div>
          )}
        </div>
      </header>
      {mockMode && (
        <div className="bg-amber-100 px-4 py-2 text-center text-sm text-amber-900">
          Mock mode: scores are placeholders derived from the file hash, not from a model.
        </div>
      )}
      <main className="mx-auto max-w-5xl px-4 py-8">
        <Routes>
          <Route path="/login" element={session ? <Navigate to="/" replace /> : <Login />} />
          <Route path="/" element={session ? <Detect /> : <Navigate to="/login" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
