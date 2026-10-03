"use client";

import { motion } from "framer-motion";
import { ArrowRight, Lightbulb, Wrench } from "lucide-react";
import { INSIGHTS } from "@/lib/loop/catalog";
import { Meter, Panel, pct, SampleCanvas, StepHeader, TagBadge } from "./ui";

const LEVEL_STYLE = {
  critical: "text-rose-300 border-rose-400/30 bg-rose-400/10",
  high: "text-amber-300 border-amber-400/30 bg-amber-400/10",
  medium: "text-sky-300 border-sky-400/30 bg-sky-400/10",
  low: "text-zinc-300 border-white/15 bg-white/5",
};

export default function StepInsights({ insights, worst, onNext }) {
  const byId = Object.fromEntries(worst.map((s) => [s.id, s]));
  return (
    <div className="space-y-8">
      <StepHeader
        index={4}
        kicker="Insight extraction"
        title="Nguyên nhân gốc rễ từ nhóm sample yếu"
        desc="Mỗi mẫu được phân rã thành đóng góp của từng loại suy giảm (cường độ × độ nhạy của model). Các mẫu cùng nguyên nhân gốc được gom thành một insight."
        right={
          <button type="button" className="st-btn st-btn-primary h-10 px-5" onClick={onNext}>
            Đề xuất tổ hợp tấn công <ArrowRight className="h-4 w-4" />
          </button>
        }
      />

      <Panel className="p-5">
        <div className="mb-3 flex items-center justify-between">
          <span className="st-h2">Phân bố insight · {worst.length} mẫu</span>
          <span className="st-faint mono text-[11px]">{insights.length} nhóm nguyên nhân</span>
        </div>
        <div className="flex h-3 w-full gap-[3px] overflow-hidden rounded-full">
          {insights.map((ins, i) => (
            <motion.div
              key={ins.tag}
              initial={{ flexGrow: 0 }}
              animate={{ flexGrow: ins.share }}
              transition={{ duration: 0.8, delay: i * 0.08 }}
              className="h-full rounded-full"
              style={{ background: INSIGHTS[ins.tag].color, flexBasis: 0 }}
              title={`${INSIGHTS[ins.tag].label}: ${pct(ins.share, 0)}`}
            />
          ))}
        </div>
        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2">
          {insights.map((ins) => (
            <span key={ins.tag} className="flex items-center gap-2 text-[12px]">
              <span className="st-dot" style={{ color: INSIGHTS[ins.tag].color }} />
              <span className="st-dim">{INSIGHTS[ins.tag].short}</span>
              <span className="mono font-semibold">{pct(ins.share, 0)}</span>
            </span>
          ))}
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-2">
        {insights.map((ins, i) => {
          const meta = INSIGHTS[ins.tag];
          return (
            <motion.div
              key={ins.tag}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.07 }}
              className="st-panel st-panel-hover relative overflow-hidden"
            >
              <div className="absolute inset-x-0 top-0 h-px" style={{ background: `linear-gradient(90deg, transparent, ${meta.color}, transparent)` }} />
              <div className="p-5">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="mb-2 flex items-center gap-2">
                      <TagBadge tag={ins.tag} />
                      <span className={`inline-flex h-6 items-center rounded-md border px-2 text-[11px] font-semibold ${LEVEL_STYLE[ins.level.key]}`}>{ins.level.label}</span>
                    </div>
                    <h3 className="text-[17px] font-semibold tracking-tight">{meta.label}</h3>
                  </div>
                  <div className="text-right">
                    <div className="mono text-[26px] font-semibold leading-none">{pct(ins.share, 0)}</div>
                    <div className="st-faint mt-1 text-[11px]">
                      {ins.count} / {worst.length} mẫu
                    </div>
                  </div>
                </div>

                <p className="st-dim mt-3 flex gap-2 text-[12.5px] leading-relaxed">
                  <Lightbulb className="mt-0.5 h-3.5 w-3.5 shrink-0" style={{ color: meta.color }} />
                  {meta.hypothesis}
                </p>

                <div className="mt-5 grid grid-cols-3 gap-4">
                  <div>
                    <div className="st-label mb-1.5">IoU TB</div>
                    <div className="mono text-[16px] font-semibold text-rose-300">{ins.avgScore.toFixed(3)}</div>
                  </div>
                  <div>
                    <div className="st-label mb-1.5">Cường độ</div>
                    <Meter value={ins.avgIntensity} color={meta.color} className="mt-2.5" />
                  </div>
                  <div>
                    <div className="st-label mb-1.5">Tin cậy</div>
                    <div className="mono text-[16px] font-semibold">{pct(ins.confidence, 0)}</div>
                  </div>
                </div>

                {ins.secondary.length > 0 && (
                  <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-white/[0.06] pt-4">
                    <span className="st-label">Đồng xuất hiện</span>
                    {ins.secondary.map((s) => (
                      <span key={s.tag} className="flex items-center gap-1.5">
                        <TagBadge tag={s.tag} small />
                        <span className="mono st-faint text-[11px]">{pct(s.share, 0)}</span>
                      </span>
                    ))}
                  </div>
                )}

                <div className="mt-4 flex items-start gap-2 rounded-lg border border-white/[0.06] bg-white/[0.02] p-3 text-[12px]">
                  <Wrench className="mt-0.5 h-3.5 w-3.5 shrink-0 text-white/40" />
                  <span className="st-dim leading-relaxed">{meta.remedy}</span>
                </div>

                <div className="mt-4 grid grid-cols-3 gap-2">
                  {ins.sampleIds.slice(0, 3).map((id) => (
                    <SampleCanvas key={id} sample={byId[id]} overlay={false} className="rounded-md border border-white/10" label={id.split("-")[1]} />
                  ))}
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
