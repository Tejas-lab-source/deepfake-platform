import { useEffect, useState } from 'react';
import { Navigate, NavLink, Route, Routes } from 'react-router-dom';
import { api, getSession, onSessionChange, setSession } from './api/client';
import AnalysisPage from './pages/AnalysisPage';
import Detect from './pages/Detect';
import FilterLab from './pages/FilterLab';
import History from './pages/History';
import Login from './pages/Login';
import ModelCard from './pages/ModelCard';
import Results from './pages/Results';

export const NAV: Array<[string, string]> = [
  ['/', 'Detect'],
  ['/history', 'History'],
  ['/filter-lab', 'Filter Lab'],
  ['/results', 'Results'],
  ['/model-card', 'Model card'],
];

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

  const guard = (page: JSX.Element) => (session ? page : <Navigate to="/login" replace />);

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-2 px-4 py-3">
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
        {session && (
          <nav aria-label="Main" className="mx-auto max-w-5xl overflow-x-auto px-4">
            <ul className="flex gap-1 text-sm">
              {NAV.map(([to, label]) => (
                <li key={to}>
                  <NavLink
                    to={to}
                    end={to === '/'}
                    className={({ isActive }) =>
                      `block whitespace-nowrap border-b-2 px-3 py-2 ${isActive ? 'border-slate-900 font-medium text-slate-900' : 'border-transparent text-slate-600 hover:text-slate-900'}`
                    }
                  >
                    {label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
        )}
      </header>
      {mockMode && (
        <div className="bg-amber-100 px-4 py-2 text-center text-sm text-amber-900">
          Mock mode: scores are placeholders derived from the file hash, not from a model.
        </div>
      )}
      <main className="mx-auto max-w-5xl px-4 py-8">
        <Routes>
          <Route path="/login" element={session ? <Navigate to="/" replace /> : <Login />} />
          <Route path="/" element={guard(<Detect />)} />
          <Route path="/history" element={guard(<History />)} />
          <Route path="/analyses/:id" element={guard(<AnalysisPage />)} />
          <Route path="/filter-lab" element={guard(<FilterLab />)} />
          <Route path="/results" element={guard(<Results />)} />
          <Route path="/model-card" element={guard(<ModelCard />)} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
