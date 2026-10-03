"use client";

import { motion } from "framer-motion";
import { ArrowRight, Box, Check, Database, Layers, ScanSearch } from "lucide-react";
import { DATASETS, INSIGHT_TAGS, INSIGHTS, MODELS } from "@/lib/loop/catalog";
import { generateSamples } from "@/lib/loop/engine";
import { cn } from "@/lib/utils";
import { Panel, SampleCanvas, StepHeader } from "./ui";

function Fingerprint({ model, fixes = {} }) {
  return (
    <div className="flex items-end gap-[3px]" title="Độ nhạy theo từng loại suy giảm">
      {INSIGHT_TAGS.map((tag) => {
        const sens = model.sens[tag] * (1 - (fixes[tag] || 0));
        return (
          <div key={tag} className="flex h-9 w-[7px] items-end rounded-[2px] bg-white/[0.06]">
            <motion.div
              className="w-full rounded-[2px]"
              style={{ background: INSIGHTS[tag].color }}
              initial={{ height: 0 }}
              animate={{ height: `${Math.max(8, sens * 100)}%` }}
              transition={{ duration: 0.6 }}
            />
          </div>
        );
      })}
    </div>
  );
}

export default function StepSelect({ modelId, setModelId, datasetId, setDatasetId, store, onRun, running }) {
  return (
    <div className="space-y-8">
      <StepHeader
        index={1}
        kicker="Chọn nguồn"
        title="Chọn model & dataset để bắt đầu vòng lặp"
        desc="Hệ thống sẽ đánh giá model trên toàn bộ dataset, tìm ra nhóm sample yếu nhất rồi tự động đề xuất chiến lược tấn công để làm giàu dữ liệu retrain."
      />

      <div className="grid gap-6 xl:grid-cols-2">
        <section>
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Box className="h-4 w-4 text-white/50" />
              <span className="st-h2">Model</span>
            </div>
            <span className="st-faint mono text-[11px]">{MODELS.length} checkpoints</span>
          </div>
          <div className="space-y-2.5">
            {MODELS.map((m) => {
              const state = store.models[m.id];
              const active = m.id === modelId;
              return (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setModelId(m.id)}
                  className={cn(
                    "st-panel st-panel-hover relative flex w-full items-center gap-4 p-4 text-left",
                    active && "!border-[#7c7cff]/70 bg-[#7c7cff]/[0.05] shadow-[0_0_0_1px_rgba(124,124,255,0.25),0_16px_40px_-20px_rgba(124,124,255,0.55)]",
                  )}
                >
                  <div className={cn("flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border", active ? "border-[#7c7cff]/60 bg-[#7c7cff]/20" : "border-white/10 bg-white/[0.03]")}>
                    {active ? <Check className="h-4 w-4 text-[#b3b3ff]" /> : <Box className="h-4 w-4 text-white/45" />}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-[14.5px] font-semibold tracking-tight">{m.name}</span>
                      <span className="mono rounded border border-white/10 bg-white/5 px-1.5 text-[10px] font-semibold text-white/70">
                        v{state?.version || 1}
                      </span>
                    </div>
                    <div className="st-faint mt-1 truncate text-[12px]">
                      {m.arch} · {m.params} params · {m.task}
                    </div>
                  </div>
                  <Fingerprint model={m} fixes={state?.fixes} />
                </button>
              );
            })}
          </div>
          <p className="st-faint mt-3 flex items-center gap-2 text-[11.5px]">
            <span className="flex gap-[2px]">
              {INSIGHT_TAGS.map((t) => (
                <span key={t} className="h-2 w-[5px] rounded-[1px]" style={{ background: INSIGHTS[t].color }} />
              ))}
            </span>
            Dấu vân tay điểm yếu: độ nhạy của model với 7 loại suy giảm (cao hơn = dễ lỗi hơn).
          </p>
        </section>

        <section>
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-white/50" />
              <span className="st-h2">Dataset</span>
            </div>
            <span className="st-faint mono text-[11px]">{DATASETS.length} sets</span>
          </div>
          <div className="space-y-2.5">
            {DATASETS.map((d) => {
              const active = d.id === datasetId;
              const preview = generateSamples(d.id)[7];
              return (
                <button
                  key={d.id}
                  type="button"
                  onClick={() => setDatasetId(d.id)}
                  className={cn(
                    "st-panel st-panel-hover relative flex w-full items-center gap-4 overflow-hidden p-3 text-left",
                    active && "!border-[#7c7cff]/70 bg-[#7c7cff]/[0.05] shadow-[0_0_0_1px_rgba(124,124,255,0.25),0_16px_40px_-20px_rgba(124,124,255,0.55)]",
                  )}
                >
                  <SampleCanvas sample={preview} clean overlay={false} className="w-[118px] shrink-0 rounded-md border border-white/10" />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-[14.5px] font-semibold tracking-tight">{d.name}</span>
                      {active && <Check className="h-3.5 w-3.5 text-[#b3b3ff]" />}
                    </div>
                    <div className="st-faint mt-1 text-[12px] leading-snug">{d.note}</div>
                    <div className="mt-2 flex flex-wrap items-center gap-1.5">
                      <span className="st-chip mono">
                        <Layers className="h-3 w-3" />
                        {d.size.toLocaleString()} ảnh
                      </span>
                      {d.classes.map((c) => (
                        <span key={c} className="st-chip">
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </section>
      </div>

      <Panel className="flex flex-col items-start justify-between gap-4 p-5 md:flex-row md:items-center">
        <div className="flex items-center gap-3 text-[13px]">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-white/[0.03]">
            <ScanSearch className="h-4 w-4 text-white/60" />
          </div>
          <div>
            <div className="font-semibold">{MODELS.find((m) => m.id === modelId)?.name}</div>
            <div className="st-faint text-[12px]">
              trên {DATASETS.find((d) => d.id === datasetId)?.name} · {DATASETS.find((d) => d.id === datasetId)?.size.toLocaleString()} mẫu
            </div>
          </div>
        </div>
        <button type="button" className="st-btn st-btn-primary h-10 px-5" onClick={onRun} disabled={running}>
          {running ? "Đang đánh giá…" : "Đánh giá baseline"}
          <ArrowRight className="h-4 w-4" />
        </button>
      </Panel>
    </div>
  );
}
