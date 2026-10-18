"use client";
import { motion } from "framer-motion";
import { ChevronRight } from "lucide-react";
import { cx, SPRING } from "@/components/ui/primitives";
import type { DataSource } from "@/lib/types";

export const STAGES = [
  { n: "01", label: "TRIAGE & ISOLATE" },
  { n: "02", label: "ADVERSARIAL STUDIO" },
  { n: "03", label: "TEMPORAL INSPECTOR" },
  { n: "04", label: "PROOF OF CURE" },
] as const;

export default function TopNav({
  stage,
  onStage,
  demo,
  onDemo,
  source,
}: {
  stage: number;
  onStage: (i: number) => void;
  demo: boolean;
  onDemo: (v: boolean) => void;
  source: DataSource;
}) {
  return (
    <header className="sticky top-0 z-40 border-b border-white/[0.08] bg-black/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 md:px-6">
        <div className="flex items-center gap-3">
          <span className="font-mono text-[13px] font-semibold tracking-[0.12em] text-white">
            ADVERTEST <span className="text-zinc-600">{"//"}</span> DATA ENGINE
          </span>
          <span className="rounded border border-white/15 bg-gradient-to-b from-white/15 to-white/[0.02] px-1.5 py-0.5 font-mono text-[10px] text-zinc-200">
            v2.4 EGO-PII
          </span>
        </div>

        <nav aria-label="Pipeline stages" className="order-3 flex w-full items-center gap-1 overflow-x-auto [scrollbar-width:none] lg:order-none lg:w-auto lg:flex-1 lg:justify-center">
          {STAGES.map((s, i) => (
            <div key={s.n} className="flex items-center gap-1">
              <button
                onClick={() => onStage(i)}
                aria-current={stage === i ? "step" : undefined}
                className={cx(
                  "relative whitespace-nowrap rounded-md px-3 py-1.5 font-mono text-[11px] tracking-wide transition-colors",
                  stage === i ? "text-black" : i < stage ? "text-zinc-300 hover:bg-white/5" : "text-zinc-500 hover:bg-white/5 hover:text-zinc-300",
                )}
              >
                {stage === i && <motion.span layoutId="stage-pill" transition={SPRING} className="absolute inset-0 rounded-md bg-white" />}
                <span className="relative">
                  <span className={stage === i ? "text-zinc-500" : "text-zinc-600"}>{s.n}.</span> {s.label}
                </span>
              </button>
              {i < STAGES.length - 1 && <ChevronRight size={12} className="text-zinc-700" />}
            </div>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-4">
          <div className="flex items-center gap-2 font-mono text-[11px] text-zinc-400">
            <span className="relative flex h-2 w-2">
              <span className={cx("absolute inline-flex h-full w-full animate-ping2 rounded-full", source === "api" ? "bg-cyan-400" : "bg-amber-400")} />
              <span className={cx("relative inline-flex h-2 w-2 rounded-full", source === "api" ? "bg-cyan-400" : "bg-amber-400")} />
            </span>
            {source === "api" ? "DATA BLOCK: CONNECTED" : "DATA BLOCK: LOCAL CACHE"}
          </div>
          <button
            role="switch"
            aria-checked={demo}
            onClick={() => onDemo(!demo)}
            className="flex items-center gap-2 rounded-md border border-white/10 px-2.5 py-1.5 font-mono text-[11px] text-zinc-300 hover:bg-white/5"
          >
            DEMO ARTIFACTS: <span className={demo ? "text-emerald-400" : "text-zinc-500"}>{demo ? "ACTIVE" : "OFF"}</span>
            <span className={cx("relative h-4 w-7 rounded-full transition-colors", demo ? "bg-emerald-500" : "bg-zinc-700")}>
              <motion.span layout transition={SPRING} className="absolute top-0.5 h-3 w-3 rounded-full bg-black" style={{ left: demo ? 14 : 2 }} />
            </span>
          </button>
        </div>
      </div>
    </header>
  );
}
