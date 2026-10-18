"use client";
import { useCallback, useEffect, useState } from "react";

export const SPRING = { type: "spring", stiffness: 300, damping: 30 };

export function Glass({ className = "", children, ...rest }) {
  return (
    <div
      className={`rounded-lg border border-white/10 bg-white/[0.02] backdrop-blur-xl transition-shadow hover:shadow-[0_0_0_1px_rgba(139,92,246,0.35),0_0_24px_-6px_rgba(34,211,238,0.35)] ${className}`}
      {...rest}
    >
      {children}
    </div>
  );
}

export function Eyebrow({ children }) {
  return <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-zinc-500">{children}</p>;
}

export function Skeleton({ className = "" }) {
  return (
    <div className={`relative overflow-hidden rounded-md bg-white/[0.04] ${className}`}>
      <div className="absolute inset-0 -translate-x-full animate-[shimmer_1.6s_infinite] bg-gradient-to-r from-transparent via-white/[0.08] to-transparent" />
    </div>
  );
}

export function ErrorState({ message, onRetry }) {
  return (
    <div className="flex items-center justify-between rounded-md border border-red-500/30 bg-red-500/5 px-4 py-3">
      <span className="font-mono text-xs text-red-400">{message}</span>
      {onRetry && (
        <button
          onClick={onRetry}
          className="rounded-md border border-white/10 px-3 py-1 font-mono text-xs text-zinc-200 hover:bg-white/5"
        >
          Retry
        </button>
      )}
    </div>
  );
}

/** GET a JSON endpoint with loading / error / reload state. */
export function useApi(path) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const load = useCallback(() => {
    setState((s) => ({ ...s, loading: true, error: null }));
    fetch(path)
      .then((r) => {
        if (!r.ok) throw new Error(`${path} → HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => setState({ data, error: null, loading: false }))
      .catch((e) => setState({ data: null, error: e.message, loading: false }));
  }, [path]);
  useEffect(load, [load]);
  return { ...state, reload: load };
}

/** Synthetic face scene used as the demo frame (no raw PII is shipped). */
export function FaceScene({ blur = 0, exposure = 1, className = "" }) {
  const id = `blur-${String(blur).replace(".", "_")}`;
  return (
    <svg viewBox="0 0 320 240" className={className} preserveAspectRatio="xMidYMid slice">
      <defs>
        <filter id={id}>
          <feGaussianBlur stdDeviation={blur} />
        </filter>
        <radialGradient id="scene-bg" cx="50%" cy="45%" r="70%">
          <stop offset="0" stopColor="#27272a" />
          <stop offset="1" stopColor="#09090b" />
        </radialGradient>
      </defs>
      <g filter={blur ? `url(#${id})` : undefined} style={{ filter: blur ? undefined : `brightness(${exposure})` }}>
        <rect width="320" height="240" fill="url(#scene-bg)" />
        <ellipse cx="172" cy="150" rx="48" ry="62" fill="#a1a1aa" />
        <ellipse cx="152" cy="118" rx="7" ry="4" fill="#18181b" />
        <ellipse cx="196" cy="116" rx="7" ry="4" fill="#18181b" />
        <path d="M172 126 l-6 22 h12z" fill="#71717a" />
        <path d="M152 176 q20 12 40 -1" stroke="#27272a" strokeWidth="3" fill="none" />
      </g>
    </svg>
  );
}
