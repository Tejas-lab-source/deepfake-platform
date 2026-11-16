import type { Badge, Verdict } from '../api/types';

/** REAL is always worded as "No manipulation found". */
export const VERDICT_LABEL: Record<Verdict, string> = {
  FAKE: 'Likely manipulated',
  REAL: 'No manipulation found',
  UNCERTAIN: 'Uncertain',
};

export const VERDICT_STYLE: Record<Verdict, { panel: string; text: string; bar: string }> = {
  FAKE: { panel: 'border-rose-300 bg-rose-50', text: 'text-rose-800', bar: 'bg-rose-500' },
  REAL: {
    panel: 'border-emerald-300 bg-emerald-50',
    text: 'text-emerald-800',
    bar: 'bg-emerald-500',
  },
  UNCERTAIN: { panel: 'border-amber-300 bg-amber-50', text: 'text-amber-800', bar: 'bg-amber-500' },
};

export const BADGE_LABEL: Record<Badge, string> = {
  verified_by_2: 'Confirmed by both checks',
  second_check_disagrees: 'Second check disagrees',
  checks_disagree: 'The two checks disagree',
  hive_hint: 'Second check shown as a hint',
  single_check: 'Single check only',
};

export const pct = (x: number): string => `${Math.round(x * 100)}%`;
