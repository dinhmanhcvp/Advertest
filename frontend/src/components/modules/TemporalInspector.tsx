"use client";
import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Crosshair, ScanLine } from "lucide-react";
import CVFrame from "@/components/ui/CVFrame";
import { Badge, Card, cx, Eyebrow, SectionTitle, Sparkline, SPRING } from "@/components/ui/primitives";
import { predictBox } from "@/lib/geometry";
import { landmarksFor } from "@/lib/taxonomy";
import type { Box, TrackFrame, WorkspaceData } from "@/lib/types";

const CONF_THRESHOLD = 0.4;
const TRACKED_CONF = 0.78;
const TRACK_ID = "#04";

const center = (b: Box): [number, number] => [b[0] + b[2] / 2, b[1] + b[3] / 2];

export default function TemporalInspector({ data }: { data: WorkspaceData }) {
  const frames: TrackFrame[] = data.track;
  const lostIdx = frames.findIndex((f) => !f.detected);
  const [idx, setIdx] = useState(lostIdx >= 0 ? lostIdx : 2);
  const [tracker, setTracker] = useState(false);

  const frame = frames[idx];
  const prev1 = frames[Math.max(0, idx - 1)];
  const prev2 = frames[Math.max(0, idx - 2)];
  const predicted = useMemo(() => predictBox(prev2.box, prev1.box), [prev1, prev2]);
  const lost = !frame.detected;
  const recovered = lost && tracker;

  const series = (on: boolean) => frames.map((f) => (!f.detected && on ? TRACKED_CONF : f.conf));
  const effective = series(tracker);

  const [pcx, pcy] = center(predicted);
  const [qcx, qcy] = center(prev1.box);
  const shownBox = recovered ? predicted : frame.box;

  return (
    <section className="space-y-4">
      <SectionTitle
        eyebrow="Stage 03 · Temporal Persistence Inspector"
        title="Rescuing detections the blur destroyed"
        aside={
          <button
            role="switch"
            aria-checked={tracker}
            onClick={() => setTracker((v) => !v)}
            className={cx(
              "flex items-center gap-3 rounded-lg border px-4 py-2.5 transition-colors",
              tracker ? "border-cyan-400/50 bg-cyan-400/10 shadow-[0_0_28px_-6px_rgba(6,182,212,0.7)]" : "border-white/10 bg-white/[0.02] hover:bg-white/[0.05]",
            )}
          >
            <span className={cx("font-mono text-xs tracking-wider", tracker ? "text-cyan-300" : "text-zinc-300")}>ENABLE TEMPORAL KALMAN TRACKER</span>
            <span className={cx("relative h-5 w-9 rounded-full transition-colors", tracker ? "bg-cyan-400" : "bg-zinc-700")}>
              <motion.span layout transition={SPRING} className="absolute top-0.5 h-4 w-4 rounded-full bg-black" style={{ left: tracker ? 18 : 2 }} />
            </span>
          </button>
        }
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,620px)_minmax(0,1fr)]">
      <Card className="overflow-hidden p-0">
        <div className="relative aspect-[4/3] bg-black">
          <CVFrame box={frame.box} blur={frame.blur} smear={frame.blur ? [[-frame.blur * 1.6, 0], [frame.blur * 1.6, 0]] : undefined} hud={`SEQ-EGO4D-0412 · ${frame.label.toUpperCase()}`} className="h-full w-full">
            {/* normally detected frames */}
            {frame.detected && (
              <g>
                <rect x={frame.box[0]} y={frame.box[1]} width={frame.box[2]} height={frame.box[3]} fill="none" stroke="#10b981" strokeWidth="0.9" />
                {landmarksFor(frame.box).map(([x, y], i) => (
                  <circle key={i} cx={x} cy={y} r="1" fill="#10b981" />
                ))}
              </g>
            )}

            <AnimatePresence>
              {recovered && (
                <motion.g key="lock" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  {/* velocity vector t-1 → t */}
                  <defs>
                    <marker id="velArrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto">
                      <path d="M0 0 L6 3 L0 6z" fill="#06b6d4" />
                    </marker>
                  </defs>
                  <motion.line
                    x1={qcx}
                    y1={qcy}
                    initial={{ x2: qcx, y2: qcy }}
                    animate={{ x2: pcx, y2: pcy }}
                    transition={SPRING}
                    stroke="#06b6d4"
                    strokeWidth="0.9"
                    markerEnd="url(#velArrow)"
                  />
                  <motion.rect
                    initial={{ attrX: prev1.box[0], attrY: prev1.box[1], width: prev1.box[2], height: prev1.box[3] }}
                    animate={{ attrX: shownBox[0], attrY: shownBox[1], width: shownBox[2], height: shownBox[3] }}
                    transition={SPRING}
                    fill="rgba(6,182,212,0.06)"
                    stroke="#06b6d4"
                    strokeWidth="1"
                    strokeDasharray="2.4 1.8"
                  />
                  {landmarksFor(predicted).map(([x, y], i) => (
                    <motion.circle
                      key={i}
                      initial={{ cx: x - (pcx - qcx), cy: y - (pcy - qcy), r: 0 }}
                      animate={{ cx: x, cy: y, r: 1.1 }}
                      transition={{ ...SPRING, delay: 0.05 * i }}
                      fill="#06b6d4"
                      style={{ filter: "drop-shadow(0 0 2px #06b6d4)" }}
                    />
                  ))}
                  <text x={shownBox[0]} y={shownBox[1] - 1.8} fill="#22d3ee" fontSize="3.4" fontFamily="var(--font-jb-mono), monospace">
                    TRACK {TRACK_ID} · x:{shownBox[0]} y:{shownBox[1]}
                  </text>
                </motion.g>
              )}
            </AnimatePresence>
          </CVFrame>

          <div className="absolute left-3 top-3">
            <AnimatePresence mode="wait">
              {recovered ? (
                <motion.div key="r" initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={SPRING}>
                  <Badge tone="track" className="bg-black/70 backdrop-blur">
                    <Crosshair size={11} /> Target locked via IMU velocity interpolation — track id {TRACK_ID}
                  </Badge>
                </motion.div>
              ) : lost ? (
                <motion.div key="l" initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={SPRING}>
                  <Badge tone="fail" className="bg-black/70 backdrop-blur">
                    <ScanLine size={11} /> V1 detection lost — conf: {frame.conf.toFixed(2)} (&lt; {CONF_THRESHOLD.toFixed(2)} threshold)
                  </Badge>
                </motion.div>
              ) : (
                <motion.div key="d" initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={SPRING}>
                  <Badge tone="cure" className="bg-black/70 backdrop-blur">
                    Detected — conf: {frame.conf.toFixed(2)}
                  </Badge>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

        </div>
      </Card>
      <div className="space-y-4">
        <Card className="p-5">
          <Eyebrow>Tracker state · {frame.label}</Eyebrow>
          <dl className="mt-3 space-y-2 font-mono text-xs">
            {[
              ["track id", recovered || frame.detected ? TRACK_ID : "—"],
              ["status", recovered ? "PREDICTED (no detection)" : lost ? "LOST" : "DETECTED"],
              ["velocity v̂ (u/frame)", `(${(prev1.box[0] - prev2.box[0]).toFixed(1)}, ${(prev1.box[1] - prev2.box[1]).toFixed(1)})`],
              ["x̂ₜ = xₜ₋₁ + v̂", `(${predicted[0].toFixed(1)}, ${predicted[1].toFixed(1)})`],
              ["conf (effective)", effective[idx].toFixed(2)],
              ["threshold", CONF_THRESHOLD.toFixed(2)],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between gap-3 border-b border-white/5 pb-1.5">
                <dt className="text-zinc-500">{k}</dt>
                <dd className="text-zinc-100">{v}</dd>
              </div>
            ))}
          </dl>
        </Card>
        <Card className="p-5">
          <Eyebrow>Confidence across t-2 … t+2</Eyebrow>
          <Sparkline values={effective} color={tracker ? "#06b6d4" : "#ef4444"} className="mt-2 h-14 w-full" />
          <p className="mt-3 text-xs leading-relaxed text-zinc-400">
            Frame t loses the face to head-turn motion blur. The tracker extrapolates the box from the last two detections
            (constant-velocity prediction) and persists the identity until the detector recovers. Dashed red line = the{" "}
            <span className="font-mono">{CONF_THRESHOLD.toFixed(2)}</span> detection threshold.
          </p>
        </Card>
      </div>
      </div>

      {/* Filmstrip */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {frames.map((f, i) => {
          const isLost = !f.detected;
          const conf = effective[i];
          return (
            <button
              key={f.label}
              onClick={() => setIdx(i)}
              data-active={i === idx}
              className={cx(
                "glow-border overflow-hidden rounded-lg border bg-white/[0.02] text-left transition",
                i === idx ? "border-white/50" : "border-white/[0.08] hover:border-white/25",
              )}
            >
              <div className="relative aspect-[4/3]">
                <CVFrame box={f.box} blur={f.blur} smear={f.blur ? [[-f.blur * 1.6, 0], [f.blur * 1.6, 0]] : undefined} hud={f.label} className="h-full w-full" />
                {isLost && (
                  <span className={cx("absolute left-1.5 top-1.5 rounded px-1.5 py-0.5 font-mono text-[9px]", tracker ? "bg-cyan-400/20 text-cyan-300" : "bg-red-500/20 text-red-400")}>
                    {tracker ? "TRACKED" : "EXTREME BLUR"}
                  </span>
                )}
              </div>
              <div className="px-3 py-2">
                <Eyebrow className={isLost ? (tracker ? "!text-cyan-400" : "!text-red-400") : ""}>
                  {f.label}
                  {isLost ? " [extreme blur]" : ""}
                </Eyebrow>
                <p className={cx("mt-1 font-mono text-sm", conf < CONF_THRESHOLD ? "text-red-400" : isLost && tracker ? "text-cyan-400" : "text-emerald-400")}>
                  {conf.toFixed(2)}
                </p>
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}
