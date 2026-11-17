import { useCallback, useEffect, useRef, useState } from 'react';

export type AsyncState<T> =
  | { status: 'loading'; data: null; error: null }
  | { status: 'error'; data: null; error: string }
  | { status: 'success'; data: T; error: null };

/** Load data on mount (and when `deps` change), with loading/error/success and a reload. */
export function useAsync<T>(
  load: () => Promise<T>,
  deps: unknown[],
): AsyncState<T> & {
  reload: () => void;
} {
  const [state, setState] = useState<AsyncState<T>>({ status: 'loading', data: null, error: null });
  const [nonce, setNonce] = useState(0);
  const loadRef = useRef(load);
  loadRef.current = load;

  useEffect(() => {
    let cancelled = false;
    setState({ status: 'loading', data: null, error: null });
    loadRef
      .current()
      .then((data) => !cancelled && setState({ status: 'success', data, error: null }))
      .catch(
        (err: unknown) =>
          !cancelled &&
          setState({
            status: 'error',
            data: null,
            error: err instanceof Error ? err.message : 'Something went wrong',
          }),
      );
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { ...state, reload };
}
