"use client";

import { motion } from "framer-motion";
import { ArrowRight, SlidersHorizontal } from "lucide-react";
import { cn } from "@/lib/utils";
import { Panel, SampleCanvas, Stat, StepHeader, TagBadge } from "./ui";

const ERR = {
  ok: { label: "OK", cls: "text-emerald-300 border-emerald-400/30 bg-emerald-400/10" },
  mislocalized: { label: "Lệch hộp", cls: "text-amber-300 border-amber-400/30 bg-amber-400/10" },
  missed: { label: "Bỏ sót", cls: "text-rose-300 border-rose-400/30 bg-rose-400/10" },
};

export function ErrPill({ err }) {
  const e = ERR[err] || ERR.ok;
  return <span className={cn("inline-flex h-5 items-center rounded-md border px-1.5 text-[10.5px] font-semibold", e.cls)}>{e.label}</span>;
}

export default function StepTriage({ baseline, worst, k, setK, onNext }) {
  const mean = worst.reduce((a, s) => a + s.score, 0) / (worst.length || 1);
  const maxScore = worst[worst.length - 1]?.score ?? 0;
  const failing = worst.filter((s) => !s.correct).length;
  const pctFill = ((k - 8) / (48 - 8)) * 100;

  return (
    <div className="space-y-8">
      <StepHeader
        index={3}
        kicker="Triage"
        title="Lọc những sample có chỉ số kém nhất"
        desc="Xếp hạng toàn bộ mẫu theo IoU tăng dần và giữ lại nhóm yếu nhất. Đây là nhóm sẽ được phân tích nguyên nhân và dùng làm hạt giống để tấn công."
        right={
          <button type="button" className="st-btn st-btn-primary h-10 px-5" onClick={onNext}>
            Trích xuất insight <ArrowRight className="h-4 w-4" />
          </button>
        }
      />

      <Panel className="grid gap-6 p-5 md:grid-cols-[minmax(0,1.2fr)_repeat(3,minmax(0,1fr))] md:items-center">
        <div>
          <div className="mb-3 flex items-center justify-between">
            <span className="flex items-center gap-2 text-[12.5px] font-semibold">
              <SlidersHorizontal className="h-3.5 w-3.5 text-white/50" /> Số sample kém nhất
            </span>
            <span className="mono rounded-md border border-white/15 bg-white/5 px-2 py-0.5 text-[12px] font-semibold">{k}</span>
          </div>
          <input
            aria-label="Số sample kém nhất"
            type="range"
            min={8}
            max={48}
            step={4}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            className="st-range"
            style={{ "--pct": `${pctFill}%` }}
          />
          <div className="st-faint mono mt-2 flex justify-between text-[10px]">
            <span>8</span>
            <span>48</span>
          </div>
        </div>
        <Stat label="Chiếm dataset" value={`${((k / baseline.n) * 100).toFixed(1)}%`} sub={`${k} / ${baseline.n.toLocaleString()} mẫu`} />
        <Stat label="IoU trung bình" value={mean.toFixed(3)} tone="bad" sub={`trần ${maxScore.toFixed(3)}`} />
        <Stat label="Đang sai" value={`${failing}/${k}`} tone={failing ? "bad" : "ok"} sub="IoU < 0.50" />
      </Panel>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-4">
        {worst.map((s, i) => (
          <motion.div
            key={s.id}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: Math.min(i * 0.025, 0.5), duration: 0.35 }}
            className="st-panel st-panel-hover group overflow-hidden"
          >
            <div className="relative">
              <SampleCanvas sample={s} pred={{ score: s.score, conf: s.conf, err: s.err }} className="w-full" />
              <span className="mono absolute right-2 top-2 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-semibold text-white/80 backdrop-blur">#{i + 1}</span>
            </div>
            <div className="space-y-2.5 p-3">
              <div className="flex items-center justify-between">
                <span className="mono text-[12px] font-semibold">{s.id}</span>
                <ErrPill err={s.err} />
              </div>
              <div className="flex items-center justify-between">
                <TagBadge tag={s.attrTag} small />
                <span className="st-chip">{s.cls}</span>
              </div>
              <div className="flex items-center justify-between border-t border-white/[0.06] pt-2 text-[11.5px]">
                <span className="st-faint">IoU</span>
                <span className="mono font-semibold text-rose-300">{s.score.toFixed(3)}</span>
                <span className="st-faint">conf</span>
                <span className="mono font-semibold">{s.conf.toFixed(2)}</span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
