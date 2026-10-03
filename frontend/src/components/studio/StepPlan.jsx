"use client";

import { motion } from "framer-motion";
import { ArrowRight, Brain, Check, Plus, X } from "lucide-react";
import { AGGRESSION_LEVELS } from "@/lib/loop/engine";
import { ATTACK_BY_ID, ATTACKS, FAMILY_COLOR, INSIGHTS, SEVERITY_LABELS } from "@/lib/loop/catalog";
import { cn } from "@/lib/utils";
import { Meter, Panel, pct, SourceBadge, StepHeader, TagBadge } from "./ui";

function SeverityPicker({ value, onChange, color }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex items-center gap-[3px]">
        {[1, 2, 3, 4, 5].map((i) => (
          <button
            key={i}
            type="button"
            aria-label={`Mức ${i}`}
            onClick={() => onChange(i)}
            className="h-4 w-[9px] rounded-[2px] transition-all hover:scale-y-125"
            style={{ background: i <= value ? color : "rgba(255,255,255,0.1)" }}
          />
        ))}
      </div>
      <span className="mono w-[58px] text-[10.5px] font-semibold" style={{ color }}>
        {value} · {SEVERITY_LABELS[value]}
      </span>
    </div>
  );
}

function ComboCard({ combo, selected, onSelect, onEdit }) {
  const available = ATTACKS.filter((a) => !combo.attacks.some((x) => x.id === a.id));
  const setSeverity = (id, severity) => onEdit(combo.attacks.map((a) => (a.id === id ? { ...a, severity } : a)));
  const remove = (id) => combo.attacks.length > 2 && onEdit(combo.attacks.filter((a) => a.id !== id));
  const add = (id) => id && onEdit([...combo.attacks, { id, severity: 3 }]);
  const mem = combo.mem;

  return (
    <motion.div
      layout
      className={cn(
        "st-panel relative flex flex-col overflow-hidden transition-all",
        selected ? "!border-[#7c7cff]/70 bg-[#7c7cff]/[0.04] shadow-[0_0_0_1px_rgba(124,124,255,0.25),0_20px_50px_-24px_rgba(124,124,255,0.6)]" : "st-panel-hover",
      )}
    >
      <button type="button" onClick={onSelect} className="flex items-center justify-between gap-2 border-b border-white/[0.06] px-4 py-3 text-left">
        <div className="flex items-center gap-2.5">
          <span className={cn("flex h-[18px] w-[18px] items-center justify-center rounded-full border", selected ? "border-[#7c7cff] bg-[#7c7cff]" : "border-white/25")}>
            {selected && <Check className="h-3 w-3 text-white" />}
          </span>
          <span className="text-[13px] font-semibold">{combo.role}</span>
          <span className="mono rounded border border-white/10 bg-white/5 px-1.5 text-[10px] font-semibold text-white/70">{combo.attacks.length} attacks</span>
        </div>
        <SourceBadge source={combo.source} />
      </button>

      <div className="flex-1 space-y-2.5 px-4 py-3.5">
        {combo.attacks.map((a) => {
          const def = ATTACK_BY_ID[a.id];
          const color = FAMILY_COLOR[def.family];
          return (
            <div key={a.id} className="flex items-center justify-between gap-2">
              <div className="min-w-0">
                <div className="flex items-center gap-1.5 text-[12.5px] font-semibold">
                  <span className="st-dot" style={{ color }} />
                  <span className="truncate">{def.name}</span>
                </div>
                <div className="st-faint truncate pl-3 text-[10.5px]">{def.desc}</div>
              </div>
              <div className="flex items-center gap-2">
                <SeverityPicker value={a.severity} onChange={(s) => setSeverity(a.id, s)} color={color} />
                <button
                  type="button"
                  aria-label={`Bỏ ${def.name}`}
                  disabled={combo.attacks.length <= 2}
                  onClick={() => remove(a.id)}
                  className="rounded p-0.5 text-white/30 transition hover:bg-white/10 hover:text-white disabled:opacity-20 disabled:hover:bg-transparent"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          );
        })}
        {combo.attacks.length < 5 && (
          <label className="relative mt-1 inline-flex cursor-pointer items-center gap-1.5 rounded-md border border-dashed border-white/15 px-2 py-1 text-[11.5px] text-white/50 transition hover:border-white/30 hover:text-white/80">
            <Plus className="h-3 w-3" /> Thêm attack
            <select
              aria-label="Thêm attack"
              value=""
              onChange={(e) => add(e.target.value)}
              className="absolute inset-0 cursor-pointer opacity-0"
            >
              <option value="">—</option>
              {available.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      <div className="space-y-3 border-t border-white/[0.06] bg-black/20 px-4 py-3.5">
        <div>
          <div className="mb-1.5 flex items-center justify-between text-[11.5px]">
            <span className="st-dim">Suy giảm dự đoán</span>
            <span className="mono font-semibold text-rose-300">−{combo.predictedDrop.toFixed(3)} IoU</span>
          </div>
          <Meter value={combo.predictedDrop / 0.55} color="#fb7185" height={5} />
        </div>
        <div>
          <div className="mb-1.5 flex items-center justify-between text-[11.5px]">
            <span className="st-dim">Giữ nhãn (fidelity)</span>
            <span className="mono font-semibold text-emerald-300">{pct(combo.fidelity, 0)}</span>
          </div>
          <Meter value={combo.fidelity} color="#34d399" height={5} />
        </div>
        {mem && mem.n > 0 ? (
          <div className="flex items-center gap-2 rounded-md border border-emerald-400/20 bg-emerald-400/[0.06] px-2.5 py-1.5 text-[11px] text-emerald-200">
            <Brain className="h-3.5 w-3.5 shrink-0" />
            <span>
              Đã chạy <b className="mono">{mem.n}×</b> · suy giảm TB <b className="mono">{(mem.sumDrop / mem.n).toFixed(3)}</b>
              {mem.nReward > 0 && (
                <>
                  {" "}
                  · reward <b className="mono">{(mem.sumReward / mem.nReward).toFixed(2)}</b>
                </>
              )}
            </span>
          </div>
        ) : (
          <div className="st-faint text-[11px]">Chưa có lịch sử RL cho recipe này.</div>
        )}
      </div>
    </motion.div>
  );
}

export default function StepPlan({ insights, combosByTag, chosen, onSelect, onEdit, aggression, setAggression, onNext, totalSamples }) {
  return (
    <div className="space-y-8">
      <StepHeader
        index={5}
        kicker="Attack strategy"
        title="Tổ hợp tấn công đề xuất cho từng insight"
        desc="Mỗi insight nhận ít nhất 2 loại tấn công kết hợp, kèm mức độ riêng. Gợi ý dựa trên độ tương thích, hiệu ứng cộng hưởng và bộ nhớ RL từ các chu kỳ trước. Bạn có thể chỉnh mức độ hoặc thêm/bớt attack."
        right={
          <button type="button" className="st-btn st-btn-primary h-10 px-5" onClick={onNext}>
            Thực thi trên {totalSamples} sample <ArrowRight className="h-4 w-4" />
          </button>
        }
      />

      <Panel className="flex flex-col items-start justify-between gap-4 p-4 md:flex-row md:items-center">
        <div>
          <div className="st-h2">Mức độ tấn công tổng thể</div>
          <div className="st-faint mt-1 text-[12px]">Điều chỉnh mức độ đề xuất cho mọi tổ hợp (tạo lại gợi ý).</div>
        </div>
        <div className="flex rounded-lg border border-white/10 bg-black/30 p-1">
          {AGGRESSION_LEVELS.map((lv) => (
            <button
              key={lv.id}
              type="button"
              onClick={() => setAggression(lv.id)}
              title={lv.hint}
              className={cn(
                "rounded-md px-4 py-1.5 text-[12.5px] font-semibold transition",
                aggression === lv.id ? "bg-white text-black" : "text-white/55 hover:text-white",
              )}
            >
              {lv.label}
            </button>
          ))}
        </div>
      </Panel>

      <div className="space-y-9">
        {insights.map((ins) => {
          const combos = combosByTag[ins.tag] || [];
          return (
            <section key={ins.tag}>
              <div className="mb-3 flex items-center gap-3">
                <TagBadge tag={ins.tag} />
                <span className="text-[14px] font-semibold">{INSIGHTS[ins.tag].label}</span>
                <span className="st-faint mono text-[11.5px]">
                  {ins.count} sample · {pct(ins.share, 0)}
                </span>
                <span className="h-px flex-1 bg-white/[0.07]" />
              </div>
              <div className="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
                {combos.map((c) => (
                  <ComboCard
                    key={c.id}
                    combo={c}
                    selected={chosen[ins.tag] === c.id}
                    onSelect={() => onSelect(ins.tag, c.id)}
                    onEdit={(attacks) => onEdit(ins.tag, c.id, attacks)}
                  />
                ))}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}
