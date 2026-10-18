"use client";
import { useState } from "react";
import { motion } from "framer-motion";
import { Eyebrow, FaceScene, Glass, SPRING } from "./ui";

// Face centre drifts right/down at constant velocity; YOLO misses frame t (index 2) under motion blur.
const FRAMES = [-2, -1, 0, 1, 2].map((dt, i) => ({
  label: dt === 0 ? "t" : dt > 0 ? `t+${dt}` : `t${dt}`,
  box: [118 + i * 9, 84 + i * 3, 104, 120],
  blur: dt === 0 ? 3.2 : dt === 1 ? 0.8 : 0,
  detected: dt !== 0,
  conf: dt === 0 ? 0 : dt === 1 ? 0.52 : 0.9,
}));

/** Constant-velocity extrapolation from t-2, t-1 (mirrors advertest/eval/temporal_tracker.py). */
export function predictBox(prev2, prev1) {
  return prev1.map((v, i) => v + (v - prev2[i]));
}

export default function TemporalInspector() {
  const [tracker, setTracker] = useState(false);
  const [active, setActive] = useState(2);
  const f = FRAMES[active];
  const predicted = predictBox(FRAMES[0].box, FRAMES[1].box);
  const missed = !f.detected;
  const recovered = missed && tracker;

  return (
    <Glass className="p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <Eyebrow>Module 04 · Temporal Inspector</Eyebrow>
          <h2 className="mt-1 text-lg font-semibold tracking-tight text-white">Temporal Persistence Tracker</h2>
        </div>
        <button
          role="switch"
          aria-checked={tracker}
          onClick={() => setTracker((v) => !v)}
          className="flex items-center gap-3 rounded-md border border-white/10 px-3 py-2"
        >
          <span className="font-mono text-xs text-zinc-300">Enable Temporal Tracker</span>
          <span className={`relative h-5 w-9 rounded-full transition-colors ${tracker ? "bg-cyan-400" : "bg-zinc-700"}`}>
            <motion.span
              layout
              transition={SPRING}
              className="absolute top-0.5 h-4 w-4 rounded-full bg-black"
              style={{ left: tracker ? 18 : 2 }}
            />
          </span>
        </button>
      </div>

      <div className="relative mt-4 aspect-[16/7] overflow-hidden rounded-md border border-white/10 bg-black">
        <FaceScene blur={f.blur} className="h-full w-full" />
        <svg viewBox="0 0 320 240" className="absolute inset-0 h-full w-full" preserveAspectRatio="xMidYMid slice">
          {f.detected && (
            <rect x={f.box[0]} y={f.box[1]} width={f.box[2]} height={f.box[3]} fill="none" stroke="#10b981" strokeWidth="2" />
          )}
          {recovered && (
            <motion.rect
              key={`rec-${tracker}`}
              initial={{ x: FRAMES[1].box[0], y: FRAMES[1].box[1] }}
              animate={{ x: predicted[0], y: predicted[1] }}
              transition={SPRING}
              width={predicted[2]}
              height={predicted[3]}
              fill="none"
              stroke="#22d3ee"
              strokeWidth="2"
              strokeDasharray="5 4"
            />
          )}
        </svg>
        <span
          className={`absolute left-3 top-3 rounded px-2 py-1 font-mono text-[11px] ${
            recovered
              ? "bg-cyan-400/10 text-cyan-400"
              : missed
                ? "bg-red-500/10 text-red-400"
                : "bg-emerald-500/10 text-emerald-400"
          }`}
        >
          {recovered ? "[Box Recovered via Velocity]" : missed ? "[Detection Failed]" : `Detected · conf ${f.conf.toFixed(2)}`}
        </span>
      </div>

      <div className="mt-3 grid grid-cols-5 gap-2">
        {FRAMES.map((fr, i) => (
          <button
            key={fr.label}
            onClick={() => setActive(i)}
            className={`overflow-hidden rounded-md border text-left transition ${
              i === active ? "border-white/60" : "border-white/10 hover:border-white/30"
            }`}
          >
            <FaceScene blur={fr.blur} className="aspect-[4/3] w-full" />
            <span
              className={`block px-2 py-1 font-mono text-[10px] ${
                !fr.detected ? (tracker ? "text-cyan-400" : "text-red-400") : "text-zinc-400"
              }`}
            >
              {fr.label} · {fr.detected ? fr.conf.toFixed(2) : tracker ? "tracked" : "miss"}
            </span>
          </button>
        ))}
      </div>
    </Glass>
  );
}
