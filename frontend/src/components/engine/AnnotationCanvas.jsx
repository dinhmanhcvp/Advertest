"use client";
import { useRef, useState } from "react";
import { ErrorState, Eyebrow, Glass, Skeleton, useApi } from "./ui";
import DiffSlider from "./DiffSlider";
import SampleInspectorDock from "./SampleInspectorDock";
import StudioToolbar from "./StudioToolbar";
import { trajectoryPath } from "./AdversarialStudio";

/** Module 3 — pan/zoom viewport with layer toggles, A/B diff slider and inspector dock. */
export default function AnnotationCanvas() {
  const poc = useApi("/api/v1/demo/proof-of-cure");
  const audit = useApi("/api/v1/demo/audit-trail");
  const [caseIdx, setCaseIdx] = useState(0);
  const [layers, setLayers] = useState({ augmented: true, boxes: true, landmarks: true, imu: false });
  const [view, setView] = useState({ x: 0, y: 0, k: 1 });
  const drag = useRef(null);

  const item = poc.data?.[caseIdx];
  const auditRow = audit.data?.find((a) => a.id === item?.case_id);

  const toggle = (k) => setLayers((l) => ({ ...l, [k]: !l[k] }));
  const onWheel = (e) => {
    const k = Math.min(4, Math.max(1, view.k * (e.deltaY < 0 ? 1.1 : 0.9)));
    setView((v) => ({ ...v, k, ...(k === 1 ? { x: 0, y: 0 } : {}) }));
  };
  const onDown = (e) => {
    if (e.target.closest('[role="slider"]') || view.k === 1) return;
    drag.current = { x: e.clientX - view.x, y: e.clientY - view.y };
  };
  const onMove = (e) => {
    if (drag.current) setView((v) => ({ ...v, x: e.clientX - drag.current.x, y: e.clientY - drag.current.y }));
  };

  return (
    <Glass className="p-5">
      <Eyebrow>Module 03 · Visual Data Inspection</Eyebrow>
      <h2 className="mt-1 text-lg font-semibold tracking-tight text-white">Model V1 vs Model V2</h2>

      {(poc.error || audit.error) && (
        <div className="mt-4">
          <ErrorState message={poc.error || audit.error} onRetry={() => { poc.reload(); audit.reload(); }} />
        </div>
      )}
      {poc.loading && <Skeleton className="mt-4 h-72" />}

      {item && (
        <>
          <div className="mt-3 flex flex-wrap gap-2">
            {poc.data.map((c, i) => (
              <button
                key={c.case_id}
                onClick={() => setCaseIdx(i)}
                className={`rounded-md border px-2.5 py-1 font-mono text-[11px] ${i === caseIdx ? "border-white/40 text-white" : "border-white/10 text-zinc-500 hover:text-zinc-300"}`}
              >
                {c.case_id}
              </button>
            ))}
          </div>

          <div className="mt-4 flex flex-col gap-5 lg:flex-row">
            <div
              className="relative min-w-0 flex-1 overflow-hidden rounded-md"
              onWheel={onWheel}
              onPointerDown={onDown}
              onPointerMove={onMove}
              onPointerUp={() => (drag.current = null)}
              onPointerLeave={() => (drag.current = null)}
            >
              <StudioToolbar layers={layers} onToggle={toggle} />
              <div style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.k})`, transformOrigin: "center" }} className="relative">
                <DiffSlider
                  v1={item.model_v1}
                  v2={{ ...item.model_v2, bbox: item.model_v2.bbox }}
                  showBoxes={layers.boxes}
                  showLandmarks={layers.landmarks}
                />
                {layers.imu && (
                  <svg viewBox="0 0 320 110" className="pointer-events-none absolute inset-x-0 bottom-3 h-20 w-full">
                    <path d={trajectoryPath(3, 3)} fill="none" stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="4 3" />
                  </svg>
                )}
              </div>
              <p className="pointer-events-none absolute bottom-2 left-2 rounded bg-black/70 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
                zoom {view.k.toFixed(1)}× · {layers.augmented ? "augmented" : "raw"} · wheel to zoom, drag to pan
              </p>
              <p className="pointer-events-none absolute bottom-2 right-2 rounded bg-emerald-500/10 px-1.5 py-0.5 font-mono text-[10px] text-emerald-400">
                {item.delta}
              </p>
            </div>
            <SampleInspectorDock item={item} audit={auditRow} />
          </div>
        </>
      )}
    </Glass>
  );
}
