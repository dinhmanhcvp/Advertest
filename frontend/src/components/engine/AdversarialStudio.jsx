"use client";
import { useMemo, useState } from "react";
import { Eyebrow, ErrorState, Glass, Skeleton, useApi } from "./ui";
import TerminalWindow from "./TerminalWindow";

/** Trajectory used by the blur kernel: sine head-bob + cubic drift (mirrors the backend). */
export function imuTrajectory(hz, severity, points = 120) {
  const amp = (5 + severity * 10) * 0.4;
  const cycles = hz / 2;
  return Array.from({ length: points }, (_, i) => {
    const t = i / (points - 1);
    const x = (t - 0.5) * amp;
    const y = amp * 0.3 * Math.sin(2 * Math.PI * cycles * t) + amp * 0.15 * (t - 0.5) ** 3 * 8;
    return [x, y];
  });
}

export function trajectoryPath(hz, severity, w = 320, h = 110) {
  const pts = imuTrajectory(hz, severity);
  const amp = (5 + severity * 10) * 0.4;
  const sx = (w - 24) / amp;
  const sy = (h / 2 - 10) / (amp * 0.5);
  return pts.map(([x, y], i) => `${i ? "L" : "M"}${(w / 2 + x * sx).toFixed(1)} ${(h / 2 + y * sy).toFixed(1)}`).join(" ");
}

export default function AdversarialStudio({ selectedId, onSelect }) {
  const { data, loading, error, reload } = useApi("/api/v1/demo/audit-trail");
  const [severity, setSeverity] = useState(3);
  const [hz, setHz] = useState(3);

  const sample = useMemo(() => {
    if (!data?.length) return null;
    return data.find((d) => d.id === selectedId) ?? data[0];
  }, [data, selectedId]);

  const record = sample && {
    id: sample.id,
    audit_trail: {
      ...sample.audit_trail,
      severity_level: severity,
      parameters: {
        kernel_px: 5 + severity * 10,
        imu_hz: Number(hz.toFixed(1)),
        trajectory: "sine(2πft) + 0.15·(t−0.5)³",
        exclusion_matrix: "flare ⊥ low-light/high-ISO → PASS",
        bbox_integrity: "≤ 50% area loss → PASS",
      },
    },
  };

  return (
    <Glass className="p-5">
      <Eyebrow>Module 02 · Adversarial Studio</Eyebrow>
      <h2 className="mt-1 text-lg font-semibold tracking-tight text-white">Constraint-Driven Attack Engine</h2>

      {error && <div className="mt-4"><ErrorState message={error} onRetry={reload} /></div>}
      {loading && <Skeleton className="mt-4 h-56" />}

      {sample && (
        <div className="mt-4 grid gap-5 lg:grid-cols-2">
          <div className="space-y-5">
            <label className="block">
              <span className="flex justify-between font-mono text-xs text-zinc-400">
                Sample
                <span className="text-zinc-600">{data.length} isolated</span>
              </span>
              <select
                value={sample.id}
                onChange={(e) => onSelect?.(e.target.value)}
                className="mt-1 w-full rounded-md border border-white/10 bg-black px-3 py-2 font-mono text-xs text-zinc-200"
              >
                {data.slice(0, 60).map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.id} · {d.audit_trail.insight_source}
                  </option>
                ))}
              </select>
            </label>

            <label className="block">
              <span className="flex justify-between font-mono text-xs text-zinc-400">
                Severity
                <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 text-amber-400">{severity} / 5</span>
              </span>
              <input type="range" min={1} max={5} step={1} value={severity} onChange={(e) => setSeverity(+e.target.value)} className="mt-2 w-full accent-amber-500" />
            </label>

            <label className="block">
              <span className="flex justify-between font-mono text-xs text-zinc-400">
                IMU Frequency
                <span className="rounded-full border border-cyan-400/30 bg-cyan-400/10 px-2 text-cyan-400">{hz.toFixed(1)} Hz</span>
              </span>
              <input type="range" min={2} max={5} step={0.1} value={hz} onChange={(e) => setHz(+e.target.value)} className="mt-2 w-full accent-cyan-400" />
            </label>

            <div className="rounded-md border border-white/10 bg-black/50 p-3">
              <p className="mb-1 font-mono text-[10px] uppercase tracking-widest text-zinc-500">Biomechanical blur trajectory</p>
              <svg viewBox="0 0 320 110" className="h-28 w-full" role="img" aria-label="IMU blur kernel trajectory">
                <line x1="0" y1="55" x2="320" y2="55" stroke="rgba(255,255,255,0.06)" />
                <path d={trajectoryPath(hz, severity)} fill="none" stroke="#22d3ee" strokeWidth="1.75" strokeLinecap="round" />
              </svg>
            </div>
          </div>

          <TerminalWindow title={`audit_trail.json · ${sample.id}`} data={record} />
        </div>
      )}
    </Glass>
  );
}
