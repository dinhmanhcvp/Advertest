"use client";
import { useEffect, useMemo, useRef, useState, type PointerEvent, type WheelEvent } from "react";
import { motion } from "framer-motion";
import { Crosshair, ImagePlus, Minus, Plus, ShieldCheck } from "lucide-react";
import CVFrame from "@/components/ui/CVFrame";
import TerminalWindow from "@/components/ui/TerminalWindow";
import { Badge, Card, cx, Eyebrow, SectionTitle, SPRING } from "@/components/ui/primitives";
import { boxIntegrity } from "@/lib/geometry";
import { makeSample } from "@/lib/mockData";
import { imuPath, landmarksFor, LANDMARK_NAMES, TAGS, toPath } from "@/lib/taxonomy";
import { FRAME_H, FRAME_W, type Box, type Pt, type StudioSample, type WorkspaceData } from "@/lib/types";

type Layer = "synth" | "bbox" | "landmarks" | "imu";
const LAYERS: { key: Layer; label: string }[] = [
  { key: "synth", label: "Raw vs. Synthesized" },
  { key: "bbox", label: "BBox Overlay" },
  { key: "landmarks", label: "5-Point Landmarks" },
  { key: "imu", label: "IMU Vector Path" },
];

function effectsFor(sample: StudioSample, severity: number, hz: number, pan: number) {
  const k = severity / 5;
  const fx = {
    blur: 0,
    smear: undefined as Pt[] | undefined,
    fisheye: 0,
    exposure: 1,
    grain: 0,
    occlusion: false,
  };
  switch (sample.tag) {
    case "error_motion_blur":
      fx.blur = 1 + k * 4;
      fx.smear = imuPath(hz, severity, pan).map(([x, y]) => [x * 0.55, y * 0.55] as Pt);
      fx.fisheye = sample.audit.applied_chain.includes("OpticalDistortion") ? 0.25 * k : 0;
      break;
    case "error_fisheye":
      fx.fisheye = 0.25 + k * 0.7;
      break;
    case "error_sensor_noise":
      fx.exposure = 1 + k * 0.9;
      fx.grain = 0.3 + k * 0.7;
      break;
    case "error_occlusion":
      fx.occlusion = true;
      break;
  }
  return fx;
}

export default function AdversarialStudio({ data, focusId }: { data: WorkspaceData; focusId: string | null }) {
  const samples = useMemo(() => {
    if (!focusId) return data.samples;
    const existing = data.samples.find((s) => s.id === focusId || s.id === `atk_${focusId}`);
    if (existing) return data.samples;
    const hn = data.hardNegatives.find((h) => h.id === focusId);
    return hn ? [makeSample(`atk_${hn.id}`, hn.tag), ...data.samples] : data.samples;
  }, [data, focusId]);

  const [activeId, setActiveId] = useState(samples[0].id);
  useEffect(() => {
    if (!focusId) return;
    const hit = samples.find((s) => s.id === focusId || s.id === `atk_${focusId}`);
    if (hit) setActiveId(hit.id);
  }, [focusId, samples]);

  const sample = samples.find((s) => s.id === activeId) ?? samples[0];

  const [severity, setSeverity] = useState(sample.severity);
  const [hz, setHz] = useState(sample.hz);
  const [pan, setPan] = useState(sample.pan);
  useEffect(() => {
    setSeverity(sample.severity);
    setHz(Math.min(5, Math.max(1.5, sample.hz)));
    setPan(sample.pan);
  }, [sample]);

  const [layers, setLayers] = useState<Record<Layer, boolean>>({ synth: true, bbox: true, landmarks: true, imu: true });
  const [hover, setHover] = useState<number | null>(null);
  const [view, setView] = useState({ x: 0, y: 0, k: 1 });
  const drag = useRef<{ x: number; y: number } | null>(null);
  const [panning, setPanning] = useState(false);
  const [upload, setUpload] = useState<string | null>(null);
  useEffect(() => () => { if (upload) URL.revokeObjectURL(upload); }, [upload]);

  const fx = effectsFor(sample, severity, hz, pan);
  const active = layers.synth ? fx : { blur: 0, smear: undefined, fisheye: 0, exposure: 1, grain: 0, occlusion: false };
  const integrity = boxIntegrity(sample.box, active.fisheye);
  const landmarks = sample.landmarks.length ? sample.landmarks : landmarksFor(sample.box);
  const [bx, by, bw, bh] = sample.box;
  const fcx = bx + bw / 2;
  const fcy = by + bh / 2;
  const curve = imuPath(hz, severity, pan);

  const audit = {
    id: sample.id,
    audit_trail: {
      insight_source: sample.audit.insight_source,
      applied_chain: sample.audit.applied_chain,
      severity_level: severity,
      imu_hz: Number(hz.toFixed(1)),
      head_pan_deg: pan,
      kernel_px: 5 + severity * 10,
      exclusion_matrix: "PASS",
      bbox_integrity: Number(integrity.toFixed(2)),
    },
  };

  const zoom = (d: number) => setView((v) => ({ ...v, k: Math.min(4, Math.max(1, v.k + d)), ...(v.k + d <= 1 ? { x: 0, y: 0 } : {}) }));
  const onWheel = (e: WheelEvent) => zoom(e.deltaY < 0 ? 0.2 : -0.2);
  const onDown = (e: PointerEvent) => {
    if (view.k > 1) {
      drag.current = { x: e.clientX - view.x, y: e.clientY - view.y };
      setPanning(true);
    }
  };
  const endPan = () => {
    drag.current = null;
    setPanning(false);
  };
  const onMove = (e: PointerEvent) => {
    if (drag.current) setView((v) => ({ ...v, x: e.clientX - drag.current!.x, y: e.clientY - drag.current!.y }));
  };

  const trajW = 240;
  const trajH = 84;
  const ext = Math.max(...curve.map(([x, y]) => Math.max(Math.abs(x), Math.abs(y))), 1);
  const sc = (trajH / 2 - 8) / ext;

  return (
    <section className="space-y-4">
      <SectionTitle
        eyebrow="Stage 02 · Adversarial Studio"
        title="Constraint-driven attack synthesis"
        aside={<Badge tone="physics">non-linear human motion · not straight-line blur</Badge>}
      />

      <div className="grid gap-4 xl:grid-cols-[300px_minmax(0,1fr)_350px]">
        {/* LEFT DOCK */}
        <Card className="space-y-5 p-4">
          <div>
            <Eyebrow>Isolated samples</Eyebrow>
            <ul className="mt-2 max-h-40 space-y-1 overflow-auto pr-1">
              {samples.map((s) => (
                <li key={s.id}>
                  <button
                    onClick={() => setActiveId(s.id)}
                    className={cx(
                      "flex w-full items-center justify-between rounded-md border px-2.5 py-1.5 text-left font-mono text-[11px] transition",
                      s.id === sample.id ? "border-white/30 bg-white/[0.06] text-white" : "border-transparent text-zinc-400 hover:bg-white/[0.04]",
                    )}
                  >
                    {s.id}
                    <span className={TAGS[s.tag].text}>{s.source === "WIDER FACE" ? "WIDER" : "EGO4D"}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>

          <div className="space-y-4">
            <Eyebrow>Biomechanical IMU blur</Eyebrow>
            {[
              { label: "Severity Level", v: severity, set: setSeverity, min: 1, max: 5, step: 1, fmt: (v: number) => `${v} / 5`, tone: "text-amber-400" },
              { label: "Walking Bounce Frequency", v: hz, set: setHz, min: 1.5, max: 5, step: 0.1, fmt: (v: number) => `${v.toFixed(1)} Hz`, tone: "text-cyan-400" },
              { label: "Head Panning Angle", v: pan, set: setPan, min: 0, max: 45, step: 1, fmt: (v: number) => `${v}°`, tone: "text-violet-400" },
            ].map((c) => (
              <label key={c.label} className="block">
                <span className="flex items-baseline justify-between text-xs text-zinc-400">
                  {c.label}
                  <span className={`font-mono ${c.tone}`}>{c.fmt(c.v)}</span>
                </span>
                <input
                  type="range"
                  min={c.min}
                  max={c.max}
                  step={c.step}
                  value={c.v}
                  onChange={(e) => c.set(Number(e.target.value))}
                  className="mt-2 w-full"
                />
              </label>
            ))}
          </div>

          <div>
            <Eyebrow>Live trajectory · oscilloscope</Eyebrow>
            <svg viewBox={`0 0 ${trajW} ${trajH}`} className="mt-2 w-full rounded-md border border-white/[0.08] bg-black/60" role="img" aria-label="Biomechanical blur kernel trajectory">
              {Array.from({ length: 11 }, (_, i) => (
                <line key={i} x1={i * 24} x2={i * 24} y1="0" y2={trajH} stroke="#fff" strokeOpacity="0.05" />
              ))}
              <line x1="0" x2={trajW} y1={trajH / 2} y2={trajH / 2} stroke="#fff" strokeOpacity="0.1" />
              <line x1="12" x2={trajW - 12} y1={trajH / 2} y2={trajH / 2} stroke="#71717a" strokeDasharray="3 3" />
              <motion.path
                key={`${hz}-${severity}-${pan}`}
                d={toPath(curve, sc * 1.9, sc * 1.9, trajW / 2, trajH / 2)}
                fill="none"
                stroke="#f59e0b"
                strokeWidth="1.75"
                strokeLinecap="round"
                initial={{ pathLength: 0.6, opacity: 0.4 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={SPRING}
              />
            </svg>
            <p className="mt-1 flex justify-between font-mono text-[10px] text-zinc-500">
              <span className="text-amber-400">— biomechanical</span>
              <span>-- linear blur</span>
            </p>
          </div>

          <div className="space-y-1.5">
            <Badge tone="cure" className="w-full justify-start normal-case tracking-normal">
              <ShieldCheck size={11} /> GUARDRAIL PASS: SunFlare + LowLight Mutually Excluded
            </Badge>
            <Badge tone={integrity >= 0.5 ? "cure" : "fail"} className="w-full justify-start normal-case tracking-normal">
              <ShieldCheck size={11} /> BBOX INTEGRITY: {(integrity * 100).toFixed(0)}% Inside Frame
              {integrity < 0.5 && " — SAMPLE DISCARDED"}
            </Badge>
          </div>
        </Card>

        {/* CENTER STAGE */}
        <Card className="relative overflow-hidden p-0">
          <div className="absolute left-1/2 top-3 z-20 flex max-w-[calc(100%-1.5rem)] -translate-x-1/2 flex-wrap justify-center gap-1 rounded-lg border border-white/10 bg-black/60 p-1 backdrop-blur-xl">
            {LAYERS.map((l) => (
              <button
                key={l.key}
                aria-pressed={layers[l.key]}
                onClick={() => setLayers((p) => ({ ...p, [l.key]: !p[l.key] }))}
                className={cx(
                  "rounded-md px-2.5 py-1 font-mono text-[11px] transition",
                  layers[l.key] ? "bg-white text-black" : "text-zinc-400 hover:bg-white/10",
                )}
              >
                {l.label}
              </button>
            ))}
            <label className="flex cursor-pointer items-center gap-1 rounded-md px-2.5 py-1 font-mono text-[11px] text-zinc-400 hover:bg-white/10">
              <ImagePlus size={12} /> Upload
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) setUpload(URL.createObjectURL(f));
                }}
              />
            </label>
          </div>

          <div
            className="relative aspect-[4/3] cursor-grab overflow-hidden bg-black active:cursor-grabbing"
            onWheel={onWheel}
            onPointerDown={onDown}
            onPointerMove={onMove}
            onPointerUp={endPan}
            onPointerLeave={endPan}
          >
            <div
              className="h-full w-full"
              style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.k})`, transformOrigin: "center", transition: panning ? "none" : "transform 120ms" }}
            >
              <CVFrame
                box={sample.box}
                blur={active.blur}
                smear={active.smear}
                fisheye={active.fisheye}
                exposure={active.exposure}
                grain={active.grain}
                occlusion={active.occlusion}
                imageUrl={upload}
                hud={`${sample.id} · ${sample.mount.toUpperCase()}`}
                className="h-full w-full"
              >
                {layers.bbox && (
                  <g>
                    <motion.rect
                      fill="none"
                      stroke="#10b981"
                      strokeWidth="0.9"
                      initial={false}
                      animate={{ attrX: bx, attrY: by, width: bw, height: bh }}
                      transition={SPRING}
                    />
                    <text x={bx} y={by - 1.6} fill="#10b981" fontSize="3.4" fontFamily="var(--font-jb-mono), monospace">
                      GT x:{bx} y:{by} w:{bw} h:{bh}
                    </text>
                  </g>
                )}
                {layers.imu && (
                  <g>
                    <defs>
                      <marker id="imuArrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto">
                        <path d="M0 0 L6 3 L0 6z" fill="#f59e0b" />
                      </marker>
                    </defs>
                    <path
                      d={toPath(curve, 1.6, 1.6, fcx, fcy)}
                      fill="none"
                      stroke="#f59e0b"
                      strokeWidth="1"
                      strokeLinecap="round"
                      markerEnd="url(#imuArrow)"
                      style={{ filter: "drop-shadow(0 0 2px #f59e0b)" }}
                    />
                  </g>
                )}
                {layers.landmarks &&
                  landmarks.map(([lx, ly], i) => (
                    <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} style={{ cursor: "crosshair" }}>
                      <circle cx={lx} cy={ly} r="3" fill="transparent" />
                      <circle cx={lx} cy={ly} r="1.1" fill="#06b6d4" style={{ filter: "drop-shadow(0 0 2px #06b6d4)" }} />
                      {hover === i && (
                        <text x={lx + 2.5} y={ly - 2} fill="#67e8f9" fontSize="3.2" fontFamily="var(--font-jb-mono), monospace">
                          {LANDMARK_NAMES[i]} ({lx.toFixed(1)}, {ly.toFixed(1)})
                        </text>
                      )}
                    </g>
                  ))}
              </CVFrame>
            </div>
          </div>

          <div className="flex items-center justify-between border-t border-white/[0.08] px-4 py-2 font-mono text-[11px] text-zinc-500">
            <span>
              {layers.synth ? "SYNTHESIZED" : "RAW"} · zoom {view.k.toFixed(1)}× · {FRAME_W}×{FRAME_H} ref
            </span>
            <span className="flex items-center gap-1">
              <button aria-label="Zoom out" onClick={() => zoom(-0.5)} className="rounded border border-white/10 p-1 hover:bg-white/10">
                <Minus size={12} />
              </button>
              <button aria-label="Zoom in" onClick={() => zoom(0.5)} className="rounded border border-white/10 p-1 hover:bg-white/10">
                <Plus size={12} />
              </button>
              <button aria-label="Reset view" onClick={() => setView({ x: 0, y: 0, k: 1 })} className="rounded border border-white/10 p-1 hover:bg-white/10">
                <Crosshair size={12} />
              </button>
            </span>
          </div>
        </Card>

        {/* RIGHT DOCK */}
        <div className="space-y-4">
          <TerminalWindow title={`audit_trail.json — ${sample.id}`} data={audit} />
          <Card className="p-4">
            <Eyebrow>Algorithmic justification</Eyebrow>
            <p className="mt-2 text-sm leading-relaxed text-zinc-300">
              {sample.tag === "error_motion_blur"
                ? `Baseline failed heavily on peripheral edges during rapid locomotion. Applied GoPro Hero10 radial intrinsics coupled with a ${hz.toFixed(1)}Hz biomechanical pitch trajectory at severity ${severity}.`
                : sample.audit.justification}
            </p>
            <p className="mt-3 flex flex-wrap gap-1.5">
              {sample.audit.applied_chain.map((c) => (
                <Badge key={c} tone="physics">
                  {c}
                </Badge>
              ))}
            </p>
          </Card>
        </div>
      </div>
    </section>
  );
}
