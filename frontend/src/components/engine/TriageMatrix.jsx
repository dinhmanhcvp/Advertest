"use client";
import { useState } from "react";
import { motion } from "framer-motion";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { ErrorState, Eyebrow, Glass, SPRING, Skeleton } from "./ui";

const SOURCES = [
  { id: "yolov7-face", label: "yolov7-face · baseline", url: "datablock://models/yolov7-face/v1?sig=••••&ttl=900" },
  { id: "ego4d-social", label: "Ego4D Social Benchmark", url: "datablock://datasets/ego4d-social/val?sig=••••&ttl=900" },
];

export function MonoTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0];
  return (
    <div className="rounded-md border border-white/10 bg-black/80 px-3 py-2 font-mono text-xs text-zinc-200 backdrop-blur-xl">
      {p.name}: <span className="text-white">{p.value}</span>
    </div>
  );
}

export default function TriageMatrix({ onSelect, selectedId }) {
  const [model, setModel] = useState(SOURCES[0].id);
  const [dataset, setDataset] = useState(SOURCES[1].id);
  const [state, setState] = useState({ data: null, loading: false, error: null });

  const run = async () => {
    setState({ data: null, loading: true, error: null });
    try {
      const r = await fetch("/api/v1/triage/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: model, dataset_path: dataset }),
      });
      if (!r.ok) throw new Error(`triage/analyze → HTTP ${r.status}`);
      setState({ data: await r.json(), loading: false, error: null });
    } catch (e) {
      setState({ data: null, loading: false, error: e.message });
    }
  };

  const { data, loading, error } = state;
  const dist = data?.failure_distribution ?? [];
  const total = dist.reduce((a, d) => a + d.value, 0) || 1;

  return (
    <Glass className="p-5">
      <Eyebrow>Module 01 · Command Center</Eyebrow>
      <h2 className="mt-1 text-lg font-semibold tracking-tight text-white">Triage Matrix</h2>

      <div className="mt-4 grid gap-3 md:grid-cols-2">
        {[
          ["Model (presigned URL)", model, setModel],
          ["Dataset (presigned URL)", dataset, setDataset],
        ].map(([label, val, set]) => (
          <label key={label} className="block">
            <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">{label}</span>
            <select
              value={val}
              onChange={(e) => set(e.target.value)}
              className="mt-1 w-full rounded-md border border-white/10 bg-black px-3 py-2 font-mono text-xs text-zinc-200"
            >
              {SOURCES.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
            <span className="mt-1 block truncate font-mono text-[10px] text-zinc-600">
              {SOURCES.find((s) => s.id === val).url}
            </span>
          </label>
        ))}
      </div>

      <button
        onClick={run}
        disabled={loading}
        className="mt-4 rounded-md bg-white px-4 py-2 text-sm font-medium text-black transition hover:bg-zinc-200 disabled:opacity-50"
      >
        Fetch &amp; Isolate Failed Samples
      </button>

      <div className="mt-5">
        {error && <ErrorState message={error} onRetry={run} />}
        {loading && (
          <div className="grid gap-4 md:grid-cols-[220px_1fr]">
            <Skeleton className="h-48" />
            <Skeleton className="h-48" />
          </div>
        )}
        {!loading && !error && !data && (
          <p className="font-mono text-xs text-zinc-600">
            No analysis yet — run the triage to isolate hard negatives (IoU &lt; 0.5 or conf &lt; 0.4).
          </p>
        )}
        {data && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={SPRING}
            className="grid gap-5 md:grid-cols-[220px_1fr]"
          >
            <div>
              <div className="h-44">
                <ResponsiveContainer>
                  <PieChart>
                    <Pie data={dist} dataKey="value" nameKey="name" innerRadius={52} outerRadius={78} stroke="none" paddingAngle={2}>
                      {dist.map((d) => (
                        <Cell key={d.name} fill={d.color} />
                      ))}
                    </Pie>
                    <Tooltip content={<MonoTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <ul className="mt-2 space-y-1">
                {dist.map((d) => (
                  <li key={d.name} className="flex items-center justify-between font-mono text-[11px] text-zinc-400">
                    <span className="flex items-center gap-2">
                      <i className="h-2 w-2 rounded-full" style={{ background: d.color }} />
                      {d.name}
                    </span>
                    <span className="text-zinc-200">{((d.value / total) * 100).toFixed(1)}%</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="max-h-64 overflow-auto rounded-md border border-white/10">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-black/80 text-zinc-500 backdrop-blur">
                  <tr>
                    <th className="px-3 py-2 font-medium">Sample</th>
                    <th className="px-3 py-2 font-medium">Failure tag</th>
                    <th className="px-3 py-2 text-right font-medium">IoU</th>
                    <th className="px-3 py-2 text-right font-medium">Conf</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {data.worst_samples.slice(0, 25).map((s) => (
                    <tr
                      key={s.id}
                      onClick={() => onSelect?.(s.id)}
                      className={`cursor-pointer border-t border-white/5 hover:bg-white/[0.04] ${selectedId === s.id ? "bg-white/[0.06]" : ""}`}
                    >
                      <td className="px-3 py-1.5 text-zinc-200">{s.id}</td>
                      <td className="px-3 py-1.5 text-amber-400">{s.reason}</td>
                      <td className="px-3 py-1.5 text-right text-zinc-300">{Number(s.iou).toFixed(2)}</td>
                      <td className="px-3 py-1.5 text-right text-zinc-300">{Number(s.confidence ?? 0).toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="border-t border-white/10 px-3 py-2 font-mono text-[10px] text-zinc-500">
                {data.total_hard_negatives} hard negatives · source: {data.source}
              </p>
            </div>
          </motion.div>
        )}
      </div>
    </Glass>
  );
}
