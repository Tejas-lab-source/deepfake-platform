import { describe, expect, it } from 'vitest';
import { reconcile, type ExternalStatus } from '../src/reconcile.js';
import type { Verdict } from '../src/types.js';

const T = 0.9;
const run = (
  verdict: Verdict,
  confident: boolean,
  score: number | null,
  status: ExternalStatus = 'ok',
) =>
  reconcile({
    ours: { verdict, confident },
    external: { status, deepfakeScore: score },
    hiveThreshold: T,
  });

describe('reconcile', () => {
  it('row 1: FAKE + FAKE -> FAKE, verified_by_2', () => {
    expect(run('FAKE', true, 0.95)).toMatchObject({ verdict: 'FAKE', badge: 'verified_by_2' });
    expect(run('FAKE', false, 0.95)).toMatchObject({ verdict: 'FAKE', badge: 'verified_by_2' });
  });

  it('row 2: REAL + REAL -> REAL, verified_by_2', () => {
    expect(run('REAL', true, 0.1)).toMatchObject({ verdict: 'REAL', badge: 'verified_by_2' });
    expect(run('REAL', false, 0.1)).toMatchObject({ verdict: 'REAL', badge: 'verified_by_2' });
  });

  it('row 3: confident FAKE + REAL -> FAKE, second_check_disagrees', () => {
    expect(run('FAKE', true, 0.1)).toMatchObject({
      verdict: 'FAKE',
      badge: 'second_check_disagrees',
      externalVerdict: 'REAL',
    });
  });

  it('row 4: confident REAL + FAKE -> REAL, second_check_disagrees', () => {
    expect(run('REAL', true, 0.95)).toMatchObject({
      verdict: 'REAL',
      badge: 'second_check_disagrees',
      externalVerdict: 'FAKE',
    });
  });

  it('row 5: not confident + disagreement -> UNCERTAIN, checks_disagree', () => {
    expect(run('FAKE', false, 0.1)).toMatchObject({
      verdict: 'UNCERTAIN',
      badge: 'checks_disagree',
    });
    expect(run('REAL', false, 0.95)).toMatchObject({
      verdict: 'UNCERTAIN',
      badge: 'checks_disagree',
    });
  });

  it('row 6: UNCERTAIN + anything -> UNCERTAIN, hive_hint', () => {
    expect(run('UNCERTAIN', false, 0.95)).toMatchObject({
      verdict: 'UNCERTAIN',
      badge: 'hive_hint',
    });
    expect(run('UNCERTAIN', false, 0.1)).toMatchObject({
      verdict: 'UNCERTAIN',
      badge: 'hive_hint',
    });
  });

  it.each(['error', 'timeout', 'quota', 'opted_out'] as const)(
    'row 7: external %s -> ours, single_check',
    (status) => {
      for (const verdict of ['FAKE', 'REAL', 'UNCERTAIN'] as const) {
        expect(run(verdict, verdict !== 'UNCERTAIN', null, status)).toMatchObject({
          verdict,
          badge: 'single_check',
          externalVerdict: null,
        });
      }
    },
  );

  it('treats an ok status with no score as unavailable', () => {
    expect(run('FAKE', true, null, 'ok')).toMatchObject({ verdict: 'FAKE', badge: 'single_check' });
  });

  it('applies the threshold inclusively and honours a custom one', () => {
    expect(run('FAKE', true, 0.9).externalVerdict).toBe('FAKE');
    expect(run('FAKE', true, 0.8999).externalVerdict).toBe('REAL');
    const custom = reconcile({
      ours: { verdict: 'FAKE', confident: true },
      external: { status: 'ok', deepfakeScore: 0.6 },
      hiveThreshold: 0.5,
    });
    expect(custom.badge).toBe('verified_by_2');
  });

  it('never flips our verdict to the opposite class', () => {
    for (const verdict of ['FAKE', 'REAL'] as const) {
      for (const confident of [true, false]) {
        for (const score of [0, 0.5, 0.9, 1]) {
          const out = run(verdict, confident, score).verdict;
          expect([verdict, 'UNCERTAIN']).toContain(out);
        }
      }
    }
  });
});
