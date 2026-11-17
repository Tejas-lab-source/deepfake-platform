import { useState, type FormEvent } from 'react';
import { api } from '../api/client';

interface Turn {
  question: string;
  answer: string | null;
  error: string | null;
  source?: string;
}

const SUGGESTIONS = [
  'Why was this flagged?',
  'How confident is it?',
  'What did the second check say?',
  'Can I trust this result?',
];
const MAX = 500;

/** "Ask about this result": answers come only from this analysis's approved facts. */
export default function AskPanel({ analysisId }: { analysisId: string }) {
  const [question, setQuestion] = useState('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);

  async function ask(q: string) {
    const text = q.trim();
    if (text.length < 3 || busy) return;
    setBusy(true);
    setQuestion('');
    setTurns((t) => [...t, { question: text, answer: null, error: null }]);
    try {
      const r = await api.ask(analysisId, text);
      setTurns((t) =>
        t.map((x, i) => (i === t.length - 1 ? { ...x, answer: r.answer, source: r.source } : x)),
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Could not get an answer';
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, error: message } : x)));
    } finally {
      setBusy(false);
    }
  }

  const submit = (e: FormEvent) => {
    e.preventDefault();
    void ask(question);
  };

  return (
    <div className="space-y-3">
      <ul className="space-y-3" aria-live="polite">
        {turns.map((t, i) => (
          <li key={i} className="space-y-1 text-sm">
            <p className="font-medium">You: {t.question}</p>
            {t.answer && <p className="rounded bg-slate-50 p-2 text-slate-800">{t.answer}</p>}
            {t.error && (
              <p role="alert" className="text-rose-700">
                {t.error}
              </p>
            )}
            {!t.answer && !t.error && <p className="text-slate-500">Thinking…</p>}
          </li>
        ))}
      </ul>
      {turns.length === 0 && (
        <div className="flex flex-wrap gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => void ask(s)}
              className="rounded-full border border-slate-300 px-3 py-1 text-sm hover:bg-slate-100"
            >
              {s}
            </button>
          ))}
        </div>
      )}
      <form onSubmit={submit} className="flex flex-col gap-2 sm:flex-row">
        <label className="sr-only" htmlFor="ask-input">
          Question about this result
        </label>
        <input
          id="ask-input"
          value={question}
          maxLength={MAX}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about this result…"
          className="flex-1 rounded border border-slate-300 px-3 py-2 text-sm"
        />
        <button
          type="submit"
          disabled={busy || question.trim().length < 3}
          className="rounded bg-slate-900 px-4 py-2 text-sm text-white disabled:opacity-50"
        >
          Ask
        </button>
      </form>
      <p className="text-xs text-slate-500">
        Answers use only this result's verdict, scores and checks. Automated, not proof.
      </p>
    </div>
  );
}
