"use client";
import { useCallback, useEffect, useRef, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";
import { ChevronsLeftRight } from "lucide-react";

/**
 * Before/After split view. The divider position is written straight to the DOM from the pointer
 * handler (clip-path + left), so dragging never waits on a React render and stays at display rate.
 */
export default function DiffSlider({
  left,
  right,
  initial = 50,
  className,
}: {
  left: ReactNode;
  right: ReactNode;
  initial?: number;
  className?: string;
}) {
  const root = useRef<HTMLDivElement>(null);
  const clip = useRef<HTMLDivElement>(null);
  const bar = useRef<HTMLDivElement>(null);
  const pos = useRef(initial);

  const apply = useCallback((p: number) => {
    pos.current = Math.min(100, Math.max(0, p));
    if (clip.current) clip.current.style.clipPath = `inset(0 ${100 - pos.current}% 0 0)`;
    if (bar.current) bar.current.style.left = `${pos.current}%`;
    root.current?.setAttribute("aria-valuenow", String(Math.round(pos.current)));
  }, []);

  useEffect(() => apply(initial), [apply, initial]);

  const fromEvent = (e: PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    apply(((e.clientX - r.left) / r.width) * 100);
  };

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "ArrowLeft") apply(pos.current - 2);
    if (e.key === "ArrowRight") apply(pos.current + 2);
    if (e.key === "Home") apply(0);
    if (e.key === "End") apply(100);
  };

  return (
    <div
      ref={root}
      role="slider"
      tabIndex={0}
      aria-label="Compare Model V1 and Model V2"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={initial}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        fromEvent(e);
      }}
      onPointerMove={(e) => {
        if (e.currentTarget.hasPointerCapture(e.pointerId)) fromEvent(e);
      }}
      onKeyDown={onKey}
      className={`relative cursor-ew-resize touch-none select-none overflow-hidden rounded-md border border-white/[0.08] bg-black outline-none focus-visible:ring-1 focus-visible:ring-cyan-400 ${className ?? ""}`}
    >
      <div className="absolute inset-0">{right}</div>
      <div ref={clip} className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - initial}% 0 0)` }}>
        {left}
      </div>
      <div ref={bar} className="pointer-events-none absolute inset-y-0 w-px bg-white" style={{ left: `${initial}%` }}>
        <span className="absolute left-0 top-1/2 flex h-7 w-7 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white text-black shadow-lg">
          <ChevronsLeftRight size={14} />
        </span>
      </div>
    </div>
  );
}
