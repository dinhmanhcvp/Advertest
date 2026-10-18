"use client";
import type { ComponentPropsWithoutRef, ReactNode } from "react";

export const SPRING = { type: "spring", stiffness: 300, damping: 30 } as const;

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

/** Bento surface: translucent glass, hairline border, inner top highlight, gradient glow on hover. */
export function Card({
  className,
  active,
  children,
  ...rest
}: { active?: boolean; children: ReactNode } & ComponentPropsWithoutRef<"div">) {
  return (
    <div
      data-active={active ? "true" : undefined}
      className={cx(
        "glow-border rounded-lg border border-white/[0.08] bg-white/[0.02] backdrop-blur-xl",
        "shadow-[inset_0_1px_0_0_rgba(255,255,255,0.1)]",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}

export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <p className={cx("font-mono text-[10px] uppercase tracking-[0.18em] text-zinc-500", className)}>{children}</p>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx("animate-pulse rounded-md bg-white/[0.04]", className)} />;
}

type Tone = "fail" | "physics" | "track" | "cure" | "neutral";
const TONES: Record<Tone, string> = {
  fail: "border-red-500/30 bg-red-500/10 text-red-400",
  physics: "border-amber-500/30 bg-amber-500/10 text-amber-400",
  track: "border-cyan-400/30 bg-cyan-400/10 text-cyan-400",
  cure: "border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
  neutral: "border-white/10 bg-white/[0.03] text-zinc-400",
};

export function Badge({ tone = "neutral", children, className }: { tone?: Tone; children: ReactNode; className?: string }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
        TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function PrimaryButton({ className, children, ...rest }: ComponentPropsWithoutRef<"button">) {
  return (
    <button
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-md bg-white px-4 py-2 text-sm font-medium text-black",
        "transition hover:bg-zinc-200 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40",
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

export function SectionTitle({ eyebrow, title, aside }: { eyebrow: string; title: string; aside?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div>
        <Eyebrow>{eyebrow}</Eyebrow>
        <h2 className="mt-1 text-xl font-semibold tracking-tight text-white">{title}</h2>
      </div>
      {aside}
    </div>
  );
}

export function Sparkline({ values, color = "#06b6d4", className }: { values: number[]; color?: string; className?: string }) {
  const w = 120;
  const h = 28;
  const pts = values.map((v, i) => `${((i / (values.length - 1)) * w).toFixed(1)},${(h - 2 - v * (h - 4)).toFixed(1)}`);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className={className} preserveAspectRatio="none" aria-hidden>
      <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
      <line x1="0" x2={w} y1={h - 2 - 0.4 * (h - 4)} y2={h - 2 - 0.4 * (h - 4)} stroke="#ef4444" strokeDasharray="2 3" strokeOpacity="0.5" />
    </svg>
  );
}
