"use client";

import { motion } from "framer-motion";
import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { ATTACK_BY_ID, FAMILY_COLOR, INSIGHTS } from "@/lib/loop/catalog";
import { H, renderSample, W } from "@/lib/loop/render";
import { cn } from "@/lib/utils";

export const pct = (v, d = 1) => (v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`);
export const num = (v, d = 3) => (v == null || Number.isNaN(v) ? "—" : v.toFixed(d));

export function Panel({ className, children, hover = false, ...rest }) {
  return (
    <div className={cn("st-panel", hover && "st-panel-hover", className)} {...rest}>
      {children}
    </div>
  );
}

export function StepHeader({ index, kicker, title, desc, right }) {
  return (
    <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
      <div className="max-w-2xl">
        <div className="mb-3 flex items-center gap-2">
          <span className="mono flex h-5 min-w-5 items-center justify-center rounded-md border border-white/15 bg-white/5 px-1.5 text-[10.5px] font-semibold text-white/80">
            {String(index).padStart(2, "0")}
          </span>
          <span className="st-label">{kicker}</span>
        </div>
        <h2 className="st-h1">{title}</h2>
        {desc && <p className="st-dim mt-2.5 max-w-xl text-[14px] leading-relaxed">{desc}</p>}
      </div>
      {right && <div className="flex shrink-0 items-center gap-2">{right}</div>}
    </div>
  );
}

export function Delta({ value, unit = "pp", good = "up", digits = 1, className }) {
  if (value == null || Number.isNaN(value)) return <span className="st-faint mono">—</span>;
  const v = unit === "pp" ? value * 100 : value;
  const up = v > 0.0001;
  const flat = Math.abs(v) < 0.0001;
  const positive = good === "up" ? up : !up;
  const Icon = flat ? Minus : up ? ArrowUpRight : ArrowDownRight;
  return (
    <span
      className={cn(
        "mono inline-flex items-center gap-0.5 text-[12px] font-semibold",
        flat ? "text-white/40" : positive ? "text-emerald-400" : "text-rose-400",
        className,
      )}
    >
      <Icon className="h-3.5 w-3.5" />
      {up ? "+" : ""}
      {v.toFixed(digits)}
      {unit === "pp" ? "pp" : ""}
    </span>
  );
}

export function Stat({ label, value, sub, delta, tone, className, big = false }) {
  const toneClass = tone === "bad" ? "text-rose-300" : tone === "ok" ? "text-emerald-300" : tone === "warn" ? "text-amber-300" : "text-white";
  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <span className="st-label">{label}</span>
      <div className="flex items-baseline gap-2">
        <span className={cn("mono font-semibold tracking-tight", big ? "text-[34px]" : "text-[22px]", toneClass)}>{value}</span>
        {delta}
      </div>
      {sub && <span className="st-faint text-[11.5px]">{sub}</span>}
    </div>
  );
}

export function Meter({ value, color = "#7c7cff", height = 6, className, track = "rgba(255,255,255,0.07)", delay = 0 }) {
  return (
    <div className={cn("w-full overflow-hidden rounded-full", className)} style={{ height, background: track }}>
      <motion.div
        className="h-full rounded-full"
        style={{ background: color }}
        initial={{ width: 0 }}
        animate={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
        transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1], delay }}
      />
    </div>
  );
}

export function Ring({ value, size = 150, stroke = 10, color = "#7c7cff", children }) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} stroke="rgba(255,255,255,0.07)" strokeWidth={stroke} fill="none" />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          fill="none"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: c * (1 - Math.max(0, Math.min(1, value))) }}
          transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1] }}
          style={{ filter: `drop-shadow(0 0 8px ${color}88)` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">{children}</div>
    </div>
  );
}

export function TagBadge({ tag, small = false, className }) {
  const meta = INSIGHTS[tag];
  if (!meta) return null;
  return (
    <span
      className={cn("inline-flex items-center gap-1.5 rounded-md border font-semibold", small ? "h-5 px-1.5 text-[10.5px]" : "h-6 px-2 text-[11.5px]", className)}
      style={{ color: meta.color, borderColor: `${meta.color}44`, background: `${meta.color}14` }}
    >
      <span className="st-dot" />
      {meta.short}
    </span>
  );
}

export function AttackChip({ id, severity, compact = false }) {
  const def = ATTACK_BY_ID[id];
  if (!def) return null;
  const color = FAMILY_COLOR[def.family];
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-md border px-1.5 text-[11px] font-semibold"
      style={{ height: compact ? 20 : 24, color, borderColor: `${color}40`, background: `${color}12` }}
    >
      {def.name}
      {severity != null && (
        <span className="mono flex items-center gap-[2px]">
          {[1, 2, 3, 4, 5].map((i) => (
            <span key={i} className="block h-2.5 w-[3px] rounded-[1px]" style={{ background: i <= severity ? color : `${color}33` }} />
          ))}
        </span>
      )}
    </span>
  );
}

export function SourceBadge({ source }) {
  const map = {
    rl: { label: "RL-learned", color: "#34d399", dot: true },
    heuristic: { label: "Heuristic", color: "#9b9da5" },
    explore: { label: "Explore", color: "#fbbf24" },
    custom: { label: "Custom", color: "#7c7cff" },
  };
  const m = map[source] || map.heuristic;
  return (
    <span className="inline-flex h-5 items-center gap-1.5 rounded-md border px-1.5 text-[10.5px] font-semibold" style={{ color: m.color, borderColor: `${m.color}44`, background: `${m.color}12` }}>
      <span className={cn("st-dot", source === "rl" && "st-pulse")} />
      {m.label}
    </span>
  );
}

/** A canvas thumbnail rendered from a sample + optional attack combo. */
export function SampleCanvas({ sample, attacks, pred, clean = false, overlay = true, className, label, fill = false }) {
  const ref = useRef(null);
  const [ready, setReady] = useState(false);
  const attackKey = attacks ? attacks.map((a) => `${a.id}:${a.severity}`).join(",") : "";
  const predKey = pred ? `${pred.err}:${pred.score.toFixed(3)}:${pred.conf.toFixed(3)}` : "";
  // biome-ignore lint/correctness/useExhaustiveDependencies: attackKey/predKey are stable serialisations of attacks/pred
  useEffect(() => {
    if (!ref.current || !sample) return;
    const id = requestAnimationFrame(() => {
      renderSample(ref.current, sample, attacks || [], { pred, clean, overlay });
      setReady(true);
    });
    return () => cancelAnimationFrame(id);
  }, [sample?.id, sample?.seed, attackKey, predKey, clean, overlay]);
  return (
    <div className={cn("relative overflow-hidden bg-[#0a0b0e]", className)} style={fill ? undefined : { aspectRatio: `${W} / ${H}` }}>
      <canvas ref={ref} width={W} height={H} className={cn("h-full w-full transition-opacity duration-300", ready ? "opacity-100" : "opacity-0")} />
      {label && (
        <span className="mono absolute left-2 top-2 rounded bg-black/65 px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-white/90 backdrop-blur">
          {label}
        </span>
      )}
    </div>
  );
}

export function Spark({ values, color = "#7c7cff", width = 90, height = 26 }) {
  if (!values?.length) return <span className="st-faint mono text-[11px]">no data</span>;
  const max = Math.max(...values, 0.01);
  const min = Math.min(...values, 0);
  const pts = values.map((v, i) => {
    const x = values.length === 1 ? width / 2 : (i / (values.length - 1)) * (width - 4) + 2;
    const y = height - 3 - ((v - min) / (max - min || 1)) * (height - 6);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return (
    <svg width={width} height={height} aria-hidden="true">
      <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round" />
      {values.length > 0 && <circle cx={pts[pts.length - 1].split(",")[0]} cy={pts[pts.length - 1].split(",")[1]} r="2.2" fill={color} />}
    </svg>
  );
}

export function EmptyState({ icon: Icon, title, desc, action }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-white/10 px-6 py-14 text-center">
      {Icon && (
        <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-white/10 bg-white/[0.03]">
          <Icon className="h-5 w-5 text-white/50" />
        </div>
      )}
      <div className="text-[14px] font-semibold text-white/90">{title}</div>
      {desc && <p className="st-dim max-w-sm text-[12.5px] leading-relaxed">{desc}</p>}
      {action}
    </div>
  );
}
