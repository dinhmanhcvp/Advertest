"use client";
import { useCallback, useRef, useState } from "react";
import { FaceScene } from "./ui";

function Box({ box, color, label, dashed }) {
  const [x, y, w, h] = box;
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} fill="none" stroke={color} strokeWidth="2" strokeDasharray={dashed ? "5 4" : undefined} />
      <text x={x} y={y - 4} fill={color} fontSize="9" fontFamily="var(--font-jb-mono), monospace">{label}</text>
    </g>
  );
}

/** Before/After split view; pointer-captured drag drives a clip-path, so it stays on the compositor. */
export default function DiffSlider({ v1, v2, showBoxes = true, showLandmarks = true }) {
  const ref = useRef(null);
  const [pos, setPos] = useState(50);

  const move = useCallback((clientX) => {
    const r = ref.current.getBoundingClientRect();
    setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)));
  }, []);

  const down = (e) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    move(e.clientX);
  };
  const dragMove = (e) => {
    if (e.currentTarget.hasPointerCapture(e.pointerId)) move(e.clientX);
  };
  const key = (e) => {
    if (e.key === "ArrowLeft") setPos((p) => Math.max(0, p - 2));
    if (e.key === "ArrowRight") setPos((p) => Math.min(100, p + 2));
  };

  const overlay = (m, color, label, dashed) => (
    <svg viewBox="0 0 320 240" className="absolute inset-0 h-full w-full" preserveAspectRatio="xMidYMid slice">
      {showBoxes && <Box box={m.bbox} color={color} label={label} dashed={dashed} />}
      {showLandmarks && m.landmarks?.map(([lx, ly], i) => <circle key={i} cx={lx} cy={ly} r="2.6" fill="#22d3ee" />)}
    </svg>
  );

  return (
    <div
      ref={ref}
      role="slider"
      tabIndex={0}
      aria-label="Compare Model V1 and Model V2"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pos)}
      onPointerDown={down}
      onPointerMove={dragMove}
      onKeyDown={key}
      className="relative aspect-[4/3] w-full cursor-ew-resize touch-none select-none overflow-hidden rounded-md border border-white/10 bg-black"
    >
      {/* V2 (right / after) */}
      <div className="absolute inset-0">
        <FaceScene exposure={1.05} className="h-full w-full" />
        {overlay(v2, "#10b981", `V2 ${v2.confidence.toFixed(2)}`)}
      </div>
      {/* V1 (left / before) clipped */}
      <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        <FaceScene blur={2.4} className="h-full w-full" />
        {overlay(v1, "#ef4444", `V1 ${v1.confidence.toFixed(2)}`, true)}
      </div>
      <div className="pointer-events-none absolute inset-y-0 w-px bg-white" style={{ left: `${pos}%` }}>
        <span className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white px-1.5 py-1 font-mono text-[10px] text-black">⇄</span>
      </div>
      <span className="absolute left-2 top-2 rounded bg-black/70 px-1.5 py-0.5 font-mono text-[10px] text-red-400">V1 · baseline</span>
      <span className="absolute right-2 top-2 rounded bg-black/70 px-1.5 py-0.5 font-mono text-[10px] text-emerald-400">V2 · cured</span>
    </div>
  );
}
