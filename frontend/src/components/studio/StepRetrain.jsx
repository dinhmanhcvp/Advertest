"use client";

import { ArrowRight, BrainCircuit, Database, RotateCcw, Trash2 } from "lucide-react";
import { Area, AreaChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { INSIGHTS } from "@/lib/loop/catalog";
import { Delta, Meter, Panel, pct, StepHeader, TagBadge } from "./ui";

export default function StepRetrain({ retrain, running, onRetrain, onReset }) {
  if (!retrain) {
    return (
      <div className="space-y-8">
        <StepHeader index={7} kicker="Retrain DB" title="Huấn luyện lại mô hình & đánh giá" desc="Sử dụng các sample đối kháng từ Retrain DB để tinh chỉnh mô hình, cải thiện độ bền bỉ (robustness)." />
        <Panel className="flex h-64 flex-col items-center justify-center gap-4 text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-full border border-white/10 bg-white/5">
            <Database className="h-5 w-5 text-white/50" />
          </div>
          <div>
            <div className="st-h2">Chưa có kết quả retrain</div>
            <div className="st-faint mt-1 text-[13px]">Nhấn nút bên dưới để mô phỏng quá trình huấn luyện lại.</div>
          </div>
          <button type="button" className="st-btn st-btn-accent h-10 px-6 mt-2" onClick={onRetrain} disabled={running}>
            {running ? "Đang huấn luyện…" : "Bắt đầu Retrain"} <ArrowRight className="h-4 w-4" />
          </button>
        </Panel>
      </div>
    );
  }

  const { curve, perTag, before, after, dbSize, mixQuality, forgetting } = retrain;

  return (
    <div className="space-y-8">
      <StepHeader
        index={7}
        kicker="Retrain DB"
        title="Kết quả tinh chỉnh mô hình"
        desc={`Đã huấn luyện xong model v${retrain.newState.version}. RL memory đã được cập nhật dựa trên độ cải thiện thực tế (stress testing) của từng loại tấn công.`}
        right={
          <button type="button" className="st-btn h-10 px-5" onClick={onReset}>
            <RotateCcw className="h-4 w-4 text-white/70" /> Vòng lặp mới (v{retrain.newState.version})
          </button>
        }
      />

      <div className="grid gap-4 md:grid-cols-3">
        <Panel className="p-5">
          <div className="st-label mb-4">Tổng quan huấn luyện</div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="st-dim text-[12.5px]">Retrain DB size</span>
              <span className="mono font-semibold">{dbSize} mẫu</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="st-dim text-[12.5px]">Chất lượng mix</span>
              <span className="mono font-semibold">{pct(mixQuality)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="st-dim text-[12.5px]">Độ quên (forgetting)</span>
              <span className="mono font-semibold text-rose-300">{pct(forgetting)}</span>
            </div>
          </div>
        </Panel>

        <Panel className="col-span-2 p-5">
          <div className="st-label mb-2">Đường cong huấn luyện (mAP & Stress)</div>
          <div className="h-[140px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={curve} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorMap" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#7c7cff" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#7c7cff" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorStress" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#34d399" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="epoch" tick={{ fill: "#61646c", fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis domain={["auto", "auto"]} tick={{ fill: "#61646c", fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip cursor={{ stroke: "rgba(255,255,255,0.1)" }} contentStyle={{ background: "#0c0e11", border: "1px solid rgba(255,255,255,.12)", borderRadius: 8, fontSize: 12 }} />
                <Area type="monotone" dataKey="map50" name="Clean mAP@50" stroke="#7c7cff" strokeWidth={2} fillOpacity={1} fill="url(#colorMap)" />
                <Area type="monotone" dataKey="stress" name="Stress Acc" stroke="#34d399" strokeWidth={2} fillOpacity={1} fill="url(#colorStress)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel className="overflow-hidden">
          <div className="border-b border-white/[0.06] bg-black/20 px-5 py-4">
            <div className="st-h2">Đánh giá Clean (Tập gốc)</div>
            <div className="st-faint mt-1 text-[12px]">Độ chính xác trên tập dữ liệu ban đầu. Đảm bảo model không bị quên quá nhiều.</div>
            <div className="mt-4 flex items-center gap-6">
              <div>
                <div className="st-label mb-1">Accuracy</div>
                <div className="flex items-baseline gap-2">
                  <span className="mono text-[24px] font-semibold">{pct(after.eval.metrics.accuracy)}</span>
                  <Delta value={after.eval.metrics.accuracy - before.eval.metrics.accuracy} />
                </div>
              </div>
              <div>
                <div className="st-label mb-1">mAP@50</div>
                <div className="flex items-baseline gap-2">
                  <span className="mono text-[24px] font-semibold">{pct(after.eval.metrics.map50)}</span>
                  <Delta value={after.eval.metrics.map50 - before.eval.metrics.map50} />
                </div>
              </div>
            </div>
          </div>
          <table className="st-table">
            <thead>
              <tr>
                <th>Insight</th>
                <th>Before</th>
                <th>After</th>
                <th>Δ</th>
              </tr>
            </thead>
            <tbody>
              {perTag.map((t) => (
                <tr key={t.tag}>
                  <td>
                    <TagBadge tag={t.tag} small />
                  </td>
                  <td className="mono">{pct(t.cleanBefore)}</td>
                  <td className="mono">{pct(t.cleanAfter)}</td>
                  <td>
                    <Delta value={t.cleanAfter - t.cleanBefore} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>

        <Panel className="overflow-hidden">
          <div className="border-b border-white/[0.06] bg-black/20 px-5 py-4">
            <div className="st-h2 flex items-center justify-between">
              Đánh giá Stress (Tập đối kháng)
              <span className="flex items-center gap-1.5 rounded-md border border-emerald-400/20 bg-emerald-400/[0.06] px-2 py-1 text-[10px] font-semibold text-emerald-300">
                <BrainCircuit className="h-3 w-3" /> Cập nhật RL
              </span>
            </div>
            <div className="st-faint mt-1 text-[12px]">Độ chính xác khi bị tấn công bằng chiến lược tốt nhất. Mức tăng trưởng này chính là phần thưởng (reward) cho bandit.</div>
            <div className="mt-4 flex items-center gap-6">
              <div>
                <div className="st-label mb-1">Accuracy</div>
                <div className="flex items-baseline gap-2">
                  <span className="mono text-[24px] font-semibold text-emerald-300">{pct(after.stress.accuracy)}</span>
                  <Delta value={after.stress.accuracy - before.stress.accuracy} />
                </div>
              </div>
            </div>
          </div>
          <table className="st-table">
            <thead>
              <tr>
                <th>Insight</th>
                <th>DB Size</th>
                <th>Before</th>
                <th>After</th>
                <th>Δ (Reward)</th>
              </tr>
            </thead>
            <tbody>
              {perTag.map((t) => (
                <tr key={t.tag}>
                  <td>
                    <TagBadge tag={t.tag} small />
                  </td>
                  <td className="mono st-dim">{t.entries}</td>
                  <td className="mono">{pct(t.stressBefore)}</td>
                  <td className="mono">{pct(t.stressAfter)}</td>
                  <td>{t.stressAfter != null ? <Delta value={t.stressAfter - t.stressBefore} className="font-bold text-emerald-300" /> : <span className="st-faint mono">—</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </div>
    </div>
  );
}
