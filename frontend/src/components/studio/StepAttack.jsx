"use client";

import { motion } from "framer-motion";
import { ArrowRight, Check, Database, GripVertical, Loader2, Play, Zap } from "lucide-react";
import { useRef, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { INSIGHTS } from "@/lib/loop/catalog";
import { summarizeAttack } from "@/lib/loop/engine";
import { cn } from "@/lib/utils";
import { ErrPill } from "./StepTriage";
import { AttackChip, Delta, Meter, Panel, SampleCanvas, Stat, StepHeader, TagBadge } from "./ui";

export function CompareView({ sample, attacks, before, after }) {
  const [mode, setMode] = useState("slider");
  const [pos, setPos] = useState(50);
  const boxRef = useRef(null);
  const dragging = useRef(false);

  const move = (clientX) => {
    const r = boxRef.current?.getBoundingClientRect();
    if (!r) return;
    setPos(Math.max(2, Math.min(98, ((clientX - r.left) / r.width) * 100)));
  };

  if (!after) {
    return (
      <div className="relative">
        <SampleCanvas sample={sample} pred={before} className="w-full rounded-lg border border-white/10" />
        <div className="absolute inset-x-0 bottom-0 flex items-center justify-center bg-gradient-to-t from-black/80 to-transparent p-4 text-[12px] font-medium text-white/80">
          Chưa tấn công — nhấn “Tấn công sample này” hoặc “Chạy toàn bộ”
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <div className="flex rounded-lg border border-white/10 bg-black/30 p-0.5 text-[11.5px] font-semibold">
          {[
            ["slider", "Trượt so sánh"],
            ["split", "Song song"],
          ].map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setMode(id)}
              className={cn("rounded-md px-3 py-1 transition", mode === id ? "bg-white text-black" : "text-white/55 hover:text-white")}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap justify-end gap-1.5">
          {attacks.map((a) => (
            <AttackChip key={a.id} id={a.id} severity={a.severity} compact />
          ))}
        </div>
      </div>

      {mode === "split" ? (
        <div className="grid grid-cols-2 gap-3">
          <SampleCanvas sample={sample} pred={before} label="BEFORE" className="rounded-lg border border-white/10" />
          <SampleCanvas sample={sample} attacks={attacks} pred={after} label="AFTER" className="rounded-lg border border-rose-400/30" />
        </div>
      ) : (
        <div
          ref={boxRef}
          className="relative w-full touch-none select-none overflow-hidden rounded-lg border border-white/10"
          style={{ aspectRatio: "384 / 240", cursor: "ew-resize" }}
          onPointerDown={(e) => {
            dragging.current = true;
            e.currentTarget.setPointerCapture(e.pointerId);
            move(e.clientX);
          }}
          onPointerMove={(e) => dragging.current && move(e.clientX)}
          onPointerUp={() => {
            dragging.current = false;
          }}
        >
          <SampleCanvas fill sample={sample} attacks={attacks} pred={after} className="absolute inset-0 h-full w-full" />
          <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
            <SampleCanvas fill sample={sample} pred={before} className="absolute inset-0 h-full w-full" />
          </div>
          <span className="mono absolute left-2 top-2 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-bold tracking-wider text-emerald-300 backdrop-blur">BEFORE</span>
          <span className="mono absolute right-2 top-2 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-bold tracking-wider text-rose-300 backdrop-blur">AFTER</span>
          <div className="pointer-events-none absolute inset-y-0 w-px bg-white shadow-[0_0_12px_rgba(255,255,255,.8)]" style={{ left: `${pos}%` }}>
            <div className="absolute left-1/2 top-1/2 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-white/40 bg-black/80 backdrop-blur">
              <GripVertical className="h-4 w-4 text-white" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const STAGES = ["Triage", "Insight", "Combo", "Attack", "Retrain DB"];

function Pipeline({ done }) {
  return (
    <div className="flex items-center">
      {STAGES.map((s, i) => (
        <div key={s} className="flex flex-1 items-center last:flex-none">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "flex h-5 w-5 items-center justify-center rounded-full border text-[10px] font-bold",
                i < done ? "border-emerald-400/60 bg-emerald-400/20 text-emerald-300" : i === done ? "border-[#7c7cff] bg-[#7c7cff]/20 text-[#b3b3ff]" : "border-white/15 text-white/30",
              )}
            >
              {i < done ? <Check className="h-3 w-3" /> : i + 1}
            </span>
            <span className={cn("text-[11.5px] font-semibold", i <= done ? "text-white" : "text-white/35")}>{s}</span>
          </div>
          {i < STAGES.length - 1 && <div className={cn("mx-3 h-px flex-1", i < done ? "bg-emerald-400/40" : "bg-white/10")} />}
        </div>
      ))}
    </div>
  );
}

export default function StepAttack({
  worst,
  plan,
  runs,
  selectedId,
  setSelectedId,
  running,
  onRunOne,
  onRunAll,
  autoAdd,
  setAutoAdd,
  inDb,
  onAddToDb,
  onNext,
}) {
  const selected = worst.find((s) => s.id === selectedId) || worst[0];
  const results = Object.values(runs);
  const cohort = results.length ? summarizeAttack(results) : null;
  const sel = runs[selected.id];
  const attacks = plan[selected.attrTag]?.attacks || [];
  const before = { score: selected.score, conf: selected.conf, err: selected.err };
  const after = sel ? { score: sel.afterScore, conf: sel.afterConf, err: sel.afterErr } : null;
  const stagesDone = 3 + (sel ? 1 : 0) + (inDb.has(selected.id) ? 1 : 0);

  const perTag = Object.entries(
    worst.reduce((acc, s) => {
      const r = runs[s.id];
      if (!r) return acc;
      const slot = (acc[s.attrTag] ||= { n: 0, b: 0, a: 0 });
      slot.n += 1;
      slot.b += r.beforeScore;
      slot.a += r.afterScore;
      return acc;
    }, {}),
  ).map(([tag, v]) => ({ tag, name: INSIGHTS[tag].short, before: v.b / v.n, after: v.a / v.n }));

  return (
    <div className="space-y-6">
      <StepHeader
        index={6}
        kicker="Attack & compare"
        title="Trước & sau tấn công"
        desc="Mỗi sample được tấn công bằng tổ hợp ứng với insight của nó, so sánh trực quan trước/sau cùng các chỉ số suy giảm. Sample hiệu quả được đưa vào Retrain DB."
        right={
          <>
            <label className="flex cursor-pointer items-center gap-2 text-[12px] font-medium text-white/70">
              <input type="checkbox" checked={autoAdd} onChange={(e) => setAutoAdd(e.target.checked)} className="h-3.5 w-3.5 accent-[#7c7cff]" />
              Tự động thêm vào DB
            </label>
            <button type="button" className="st-btn st-btn-accent h-10 px-5" onClick={onRunAll} disabled={running}>
              {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
              {running ? "Đang tấn công…" : results.length ? "Chạy lại toàn bộ" : "Chạy toàn bộ"}
            </button>
            <button type="button" className="st-btn st-btn-primary h-10 px-5" onClick={onNext} disabled={!results.length || running}>
              Retrain DB <ArrowRight className="h-4 w-4" />
            </button>
          </>
        }
      />

      <Panel className="p-5">
        <div className="mb-4 flex items-center justify-between">
          <span className="st-h2">Chỉ số suy giảm toàn cohort</span>
          <span className="mono st-faint text-[11.5px]">
            {results.length}/{worst.length} sample đã xử lý
          </span>
        </div>
        <Meter value={results.length / worst.length} color="#7c7cff" height={4} className="mb-5" />
        {cohort ? (
          <div className="grid grid-cols-2 gap-x-6 gap-y-5 md:grid-cols-3 xl:grid-cols-6">
            <Stat label="IoU trung bình" value={cohort.scoreAfter.toFixed(3)} sub={`từ ${cohort.scoreBefore.toFixed(3)}`} delta={<Delta value={cohort.scoreAfter - cohort.scoreBefore} unit="raw" digits={3} />} />
            <Stat label="Conf trung bình" value={cohort.confAfter.toFixed(2)} sub={`từ ${cohort.confBefore.toFixed(2)}`} delta={<Delta value={cohort.confAfter - cohort.confBefore} unit="raw" digits={2} />} />
            <Stat label="Tỉ lệ phát hiện" value={`${(cohort.detAfter * 100).toFixed(0)}%`} sub={`từ ${(cohort.detBefore * 100).toFixed(0)}%`} delta={<Delta value={cohort.detAfter - cohort.detBefore} />} />
            <Stat label="Miss rate" value={`${(cohort.missAfter * 100).toFixed(0)}%`} sub={`từ ${(cohort.missBefore * 100).toFixed(0)}%`} tone="bad" delta={<Delta value={cohort.missAfter - cohort.missBefore} good="down" />} />
            <Stat label="Attack success" value={`${(cohort.asr * 100).toFixed(0)}%`} sub="suy giảm ≥ 40% hoặc mất phát hiện" tone="bad" />
            <Stat label="Label fidelity" value={`${(cohort.meanFidelity * 100).toFixed(0)}%`} sub="độ giữ nhãn trung bình" tone="ok" />
          </div>
        ) : (
          <div className="st-faint py-4 text-[12.5px]">Chưa có dữ liệu — chạy tấn công để xem chỉ số.</div>
        )}
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[300px_minmax(0,1fr)]">
        <Panel className="flex max-h-[760px] flex-col overflow-hidden">
          <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-3">
            <span className="st-h2">Hàng đợi sample</span>
            <span className="st-chip mono">{worst.length}</span>
          </div>
          <div className="flex-1 overflow-y-auto">
            {worst.map((s) => {
              const r = runs[s.id];
              const active = s.id === selected.id;
              return (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => setSelectedId(s.id)}
                  className={cn(
                    "flex w-full items-center justify-between gap-2 border-b border-white/[0.04] px-4 py-2.5 text-left transition",
                    active ? "bg-[#7c7cff]/10" : "hover:bg-white/[0.03]",
                  )}
                >
                  <div className="min-w-0">
                    <div className="mono text-[12px] font-semibold">{s.id}</div>
                    <div className="mt-1">
                      <TagBadge tag={s.attrTag} small />
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    {r ? (
                      <span className="mono text-[11px] font-semibold text-rose-300">
                        {s.score.toFixed(2)} → {r.afterScore.toFixed(2)}
                      </span>
                    ) : (
                      <span className="mono st-faint text-[11px]">{s.score.toFixed(2)}</span>
                    )}
                    {inDb.has(s.id) ? (
                      <span className="flex items-center gap-1 text-[10px] font-semibold text-emerald-300">
                        <Database className="h-3 w-3" /> in DB
                      </span>
                    ) : r ? (
                      <span className="text-[10px] font-semibold text-amber-300">đã tấn công</span>
                    ) : (
                      <span className="st-faint text-[10px] font-semibold">chờ</span>
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </Panel>

        <div className="space-y-4">
          <Panel className="p-5">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <span className="mono text-[15px] font-semibold">{selected.id}</span>
                <TagBadge tag={selected.attrTag} />
                <span className="st-chip">{selected.cls}</span>
              </div>
              <div className="flex gap-2">
                <button type="button" className="st-btn" disabled={running} onClick={() => onRunOne(selected)}>
                  <Play className="h-3.5 w-3.5" /> Tấn công sample này
                </button>
                {sel && !inDb.has(selected.id) && (
                  <button type="button" className="st-btn" onClick={() => onAddToDb(selected)}>
                    <Database className="h-3.5 w-3.5" /> Thêm vào DB
                  </button>
                )}
              </div>
            </div>

            <div className="relative">
              {running && selected && !sel && <div className="st-scan pointer-events-none absolute inset-0 z-10 overflow-hidden rounded-lg" />}
              <CompareView sample={selected} attacks={attacks} before={before} after={after} />
            </div>

            <div className="mt-5 rounded-lg border border-white/[0.06] bg-black/20 px-4 py-3.5">
              <Pipeline done={stagesDone} />
            </div>
          </Panel>

          <div className="grid gap-4 lg:grid-cols-2">
            <Panel className="p-5">
              <div className="st-h2 mb-4">Chỉ số sample</div>
              <table className="st-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Before</th>
                    <th>After</th>
                    <th>Δ</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="st-dim">IoU</td>
                    <td className="mono">{selected.score.toFixed(3)}</td>
                    <td className="mono">{sel ? sel.afterScore.toFixed(3) : "—"}</td>
                    <td>{sel ? <Delta value={sel.afterScore - selected.score} unit="raw" digits={3} /> : "—"}</td>
                  </tr>
                  <tr>
                    <td className="st-dim">Confidence</td>
                    <td className="mono">{selected.conf.toFixed(2)}</td>
                    <td className="mono">{sel ? sel.afterConf.toFixed(2) : "—"}</td>
                    <td>{sel ? <Delta value={sel.afterConf - selected.conf} unit="raw" digits={2} /> : "—"}</td>
                  </tr>
                  <tr>
                    <td className="st-dim">Trạng thái</td>
                    <td>
                      <ErrPill err={selected.err} />
                    </td>
                    <td>{sel ? <ErrPill err={sel.afterErr} /> : "—"}</td>
                    <td>{sel ? <span className="mono text-[11px] font-semibold text-rose-300">−{(sel.relDrop * 100).toFixed(0)}%</span> : "—"}</td>
                  </tr>
                  <tr>
                    <td className="st-dim">Fidelity</td>
                    <td className="mono">—</td>
                    <td className="mono">{sel ? `${(sel.fidelity * 100).toFixed(0)}%` : "—"}</td>
                    <td />
                  </tr>
                </tbody>
              </table>
            </Panel>

            <Panel className="p-5">
              <div className="st-h2 mb-1">IoU theo insight</div>
              <div className="st-faint mb-3 text-[12px]">Trước vs sau tấn công (trung bình).</div>
              {perTag.length ? (
                <div className="h-[170px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={perTag} margin={{ top: 4, right: 4, left: -22, bottom: 0 }} barGap={3}>
                      <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="name" tick={{ fill: "#9b9da5", fontSize: 10 }} axisLine={false} tickLine={false} />
                      <YAxis domain={[0, 0.6]} tick={{ fill: "#61646c", fontSize: 10 }} axisLine={false} tickLine={false} />
                      <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} contentStyle={{ background: "#0c0e11", border: "1px solid rgba(255,255,255,.12)", borderRadius: 8, fontSize: 12 }} formatter={(v) => v.toFixed(3)} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Bar dataKey="before" name="Before" fill="#7c7cff" radius={[3, 3, 0, 0]} />
                      <Bar dataKey="after" name="After" fill="#fb7185" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="st-faint flex h-[170px] items-center justify-center text-[12px]">Chưa có dữ liệu</div>
              )}
            </Panel>
          </div>
        </div>
      </div>
      {/* keep motion import used for subtle mount transitions */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="hidden" />
    </div>
  );
}
