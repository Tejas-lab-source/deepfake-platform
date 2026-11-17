import { useEffect, useState, type CSSProperties } from 'react';
import { api } from '../api/client';

interface Props {
  analysisId: string;
  /** A derived image key, e.g. "derived/<hex>.png". */
  imageKey: string;
  alt: string;
  className?: string;
  style?: CSSProperties;
}

/** An analysis image fetched with the access token (plain <img src> cannot send it). */
export default function AuthedImage({ analysisId, imageKey, alt, className, style }: Props) {
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let url: string | null = null;
    let cancelled = false;
    setSrc(null);
    setFailed(false);
    api
      .image(analysisId, imageKey)
      .then((b) => {
        if (cancelled) return;
        url = URL.createObjectURL(b);
        setSrc(url);
      })
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [analysisId, imageKey]);

  if (failed) {
    return (
      <div
        className={`grid place-items-center bg-slate-100 text-xs text-slate-500 ${className ?? ''}`}
      >
        Image unavailable
      </div>
    );
  }
  if (!src) {
    return <div aria-busy="true" className={`animate-pulse bg-slate-100 ${className ?? ''}`} />;
  }
  return <img src={src} alt={alt} className={className} style={style} />;
}
