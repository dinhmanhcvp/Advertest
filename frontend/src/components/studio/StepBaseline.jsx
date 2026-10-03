"use client";

import { ArrowRight, Crosshair } from "lucide-react";
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis } from "recharts";
import { INSIGHTS } from "@/lib/loop/catalog";
import { Delta, Meter, Panel, pct, Ring, Stat, StepHeader, TagBadge } from "./ui";

export default function StepBaseline({ baseline, model, dataset, modelState, onNext }) {
  const m = baseline.metrics;
  const accColor = m.accuracy >= 0.9 ? "#34d399" : m.accuracy >= 0.75 ? "#7c7cff" : "#fb7185";
  const last = modelState.history?.[modelState.history.length - 1];
  const prev = last && last.datasetId === dataset.id ? last : null;
  const hist = baseline.bins.map((b) => ({ x: b.lo, label: `${b.lo.toFixed(2)}–${b.hi.toFixed(2)}`, count: b.count, bad: b.hi <= 0.5 }));
  const sortedClass = [...baseline.byClass].sort((a, b) => a.acc - b.acc);
  const sortedTag = [...baseline.byTag].filter((t) => t.n > 0).sort((a, b) => a.acc - b.acc);

  return (
    <div className="space-y-8">
      <StepHeader
        index={2}
        kicker="Baseline"
        title="Độ chính xác hiện tại của model"
        desc={`${model.name} · v${modelState.version || 1} được đánh giá trên ${baseline.n.toLocaleString()} mẫu của ${dataset.name}. Một mẫu được tính đúng khi IoU ≥ 0.50.`}
        right={
          <button type="button" className="st-btn st-btn-primary h-10 px-5" onClick={onNext}>
            Bóc tách sample kém nhất <ArrowRight className="h-4 w-4" />
          </button>
        }
      />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
        <Panel className="relative overflow-hidden p-6">
          <div className="st-glow -left-10 -top-10 h-48 w-48" style={{ background: accColor }} />
          <div className="relative flex items-center gap-6">
            <Ring value={m.accuracy} size={168} stroke={11} color={accColor}>
              <span className="mono text-[34px] font-semibold leading-none tracking-tight">{(m.accuracy * 100).toFixed(1)}</span>
              <span className="st-label mt-1.5">Accuracy %</span>
            </Ring>
            <div className="space-y-3">
              <div className="st-label">Chi tiết</div>
              <div className="space-y-2 text-[12.5px]">
                <div className="flex items-center justify-between gap-6">
                  <span className="st-dim">Đúng</span>
                  <span className="mono font-semibold text-emerald-300">{m.ok}</span>
                </div>
                <div className="flex items-center justify-between gap-6">
                  <span className="st-dim">Lệch hộp</span>
                  <span className="mono font-semibold text-amber-300">{m.mis}</span>
                </div>
                <div className="flex items-center justify-between gap-6">
                  <span className="st-dim">Bỏ sót</span>
                  <span className="mono font-semibold text-rose-300">{m.missed}</span>
                </div>
              </div>
              {prev && (
                <div className="border-t border-white/10 pt-3">
                  <div className="st-label mb-1.5">So với v{modelState.version - 1}</div>
                  <Delta value={m.accuracy - prev.cleanAcc} />
                </div>
              )}
            </div>
          </div>
          <div className="relative mt-6 grid grid-cols-2 gap-5 border-t border-white/10 pt-5">
            <Stat label="mAP@50" value={pct(m.map50)} />
            <Stat label="Mean IoU" value={m.meanIou.toFixed(3)} />
            <Stat label="Miss rate" value={pct(m.missRate)} tone={m.missRate > 0.05 ? "bad" : undefined} />
            <Stat label="Mislocalized" value={pct(m.mislocRate)} tone={m.mislocRate > 0.1 ? "warn" : undefined} />
          </div>
        </Panel>

        <Panel className="p-5">
          <div className="mb-1 flex items-center justify-between">
            <div>
              <div className="st-h2">Phân bố điểm IoU</div>
              <div className="st-faint mt-1 text-[12px]">Vùng đỏ = mẫu dưới ngưỡng đúng (IoU &lt; 0.50) — nguồn gốc của vòng lặp cải thiện.</div>
            </div>
            <span className="st-chip mono">
              <Crosshair className="h-3 w-3" />
              thr 0.50
            </span>
          </div>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={hist} margin={{ top: 18, right: 8, left: 8, bottom: 0 }} barCategoryGap={2}>
                <XAxis dataKey="x" tickFormatter={(v) => v.toFixed(1)} tick={{ fill: "#61646c", fontSize: 10 }} axisLine={false} tickLine={false} interval={1} />
                <Tooltip
                  cursor={{ fill: "rgba(255,255,255,0.04)" }}
                  contentStyle={{ background: "#0c0e11", border: "1px solid rgba(255,255,255,.12)", borderRadius: 8, fontSize: 12 }}
                  labelFormatter={(_, p) => `IoU ${p?.[0]?.payload?.label}`}
                  formatter={(v) => [v, "samples"]}
                />
                <ReferenceLine x={0.5} stroke="rgba(255,255,255,.35)" strokeDasharray="3 3" />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {hist.map((h) => (
                    <Cell key={h.x} fill={h.bad ? "#fb7185" : "#7c7cff"} fillOpacity={h.bad ? 0.9 : 0.75} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel className="p-5">
          <div className="st-h2 mb-1">Accuracy theo lớp</div>
          <div className="st-faint mb-5 text-[12px]">Bóc tách classification — lớp yếu nhất ở trên cùng.</div>
          <div className="space-y-4">
            {sortedClass.map((c, i) => (
              <div key={c.cls}>
                <div className="mb-1.5 flex items-center justify-between text-[12.5px]">
                  <span className="font-medium">{c.cls}</span>
                  <span className="mono st-dim">
                    {pct(c.acc)} <span className="st-faint">· n={c.n}</span>
                  </span>
                </div>
                <Meter value={c.acc} color={c.acc < m.accuracy - 0.04 ? "#fb7185" : "#7c7cff"} delay={i * 0.05} />
              </div>
            ))}
          </div>
        </Panel>

        <Panel className="p-5">
          <div className="st-h2 mb-1">Accuracy theo điều kiện chủ đạo</div>
          <div className="st-faint mb-5 text-[12px]">Mẫu được gán vào điều kiện đóng góp nhiều nhất vào sai số (attribution).</div>
          <div className="space-y-4">
            {sortedTag.map((t, i) => (
              <div key={t.tag}>
                <div className="mb-1.5 flex items-center justify-between text-[12.5px]">
                  <TagBadge tag={t.tag} small />
                  <span className="mono st-dim">
                    {pct(t.acc)} <span className="st-faint">· n={t.n}</span>
                  </span>
                </div>
                <Meter value={t.acc} color={INSIGHTS[t.tag].color} delay={i * 0.05} />
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}
