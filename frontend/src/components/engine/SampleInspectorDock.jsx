"use client";
import TerminalWindow from "./TerminalWindow";
import { Eyebrow } from "./ui";

export default function SampleInspectorDock({ item, audit }) {
  if (!item) return null;
  const gain = item.model_v2.confidence - item.model_v1.confidence;
  const rows = [
    ["case_id", item.case_id],
    ["failure_tag", item.failure_tag],
    ["condition", item.condition],
    ["bbox [x,y,w,h]", `[${item.model_v1.bbox.join(", ")}]`],
  ];
  return (
    <aside className="flex w-full flex-col gap-4 lg:w-80">
      <div>
        <Eyebrow>Sample Inspector</Eyebrow>
        <dl className="mt-2 divide-y divide-white/5 rounded-md border border-white/10 text-xs">
          {rows.map(([k, v]) => (
            <div key={k} className="flex justify-between gap-3 px-3 py-1.5">
              <dt className="text-zinc-500">{k}</dt>
              <dd className="truncate text-right font-mono text-zinc-200">{v}</dd>
            </div>
          ))}
        </dl>
      </div>

      <div>
        <Eyebrow>Confidence delta</Eyebrow>
        <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/5">
          <div className="relative h-full">
            <div className="absolute h-full bg-red-500/70" style={{ width: `${item.model_v1.confidence * 100}%` }} />
            <div className="absolute h-full bg-emerald-500/80" style={{ left: `${item.model_v1.confidence * 100}%`, width: `${gain * 100}%` }} />
          </div>
        </div>
        <p className="mt-1.5 flex justify-between font-mono text-xs">
          <span className="text-red-400">V1 {item.model_v1.confidence.toFixed(2)}</span>
          <span className="text-emerald-400">V2 {item.model_v2.confidence.toFixed(2)}</span>
          <span className="text-zinc-200">+{gain.toFixed(2)}</span>
        </p>
      </div>

      {audit && <TerminalWindow title="audit_log" data={audit.audit_trail} />}
    </aside>
  );
}
