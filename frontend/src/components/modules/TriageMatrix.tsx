"use client";
import { useState } from "react";
import { motion } from "framer-motion";
import { ArrowRight, ScanSearch } from "lucide-react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { Badge, Card, Eyebrow, PrimaryButton, SectionTitle, Skeleton, SPRING } from "@/components/ui/primitives";
import { pct, TAGS } from "@/lib/taxonomy";
import type { WorkspaceData } from "@/lib/types";

const DEMO_FIELDS = [
  { label: "Source", value: "Data Block Presigned Stream: ego4d_social_val.tar" },
  { label: "Base reference", value: "wider_face_val.pt" },
  { label: "Model", value: "yolov7-face-baseline.pt" },
];

interface TipProps {
  active?: boolean;
  payload?: { name?: string; value?: number }[];
}
function MonoTip({ active, payload }: TipProps) {
  if (!active || !payload?.length) return null;
  const p = payload[0];
  return (
    <div className="rounded-md border border-white/10 bg-black/80 px-3 py-2 font-mono text-xs text-zinc-200 backdrop-blur-xl">
      {p.name}: <span className="text-white">{p.value}</span>
    </div>
  );
}

export default function TriageMatrix({
  data,
  loading,
  selected,
  onSelect,
  onProceed,
}: {
  data: WorkspaceData;
  loading: boolean;
  selected: string | null;
  onSelect: (id: string) => void;
  onProceed: () => void;
}) {
  const [scanning, setScanning] = useState(false);
  const [ran, setRan] = useState(true);
  const busy = scanning || loading;

  const run = () => {
    setScanning(true);
    setRan(false);
    window.setTimeout(() => {
      setScanning(false);
      setRan(true);
    }, 1400);
  };

  const total = data.slices.reduce((a, s) => a + s.count, 0);
  const prov = data.source !== "local" ? data.report.provenance : undefined;
  const measured = !!prov?.n_train; // real Colab run: show what was actually used instead of the spec text
  const fields = measured
    ? [
        { label: "Source", value: `WIDER FACE validation (streamed) · ${prov?.n_train} train / ${prov?.n_blind} blind images` },
        { label: "Base reference", value: "WIDER FACE · clean (no corruption)" },
        { label: "Model", value: "derronqi/yolov7-face · yolov7-tiny-face (baseline V1)" },
      ]
    : DEMO_FIELDS;

  return (
    <section className="space-y-4">
      <SectionTitle
        eyebrow="Stage 01 · Triage & Smart Sampling"
        title="Baseline failure diagnostic"
        aside={<Badge tone="fail">hard negative = IoU &lt; 0.50 or conf &lt; 0.40</Badge>}
      />

      <Card className="p-4">
        <div className="grid gap-3 md:grid-cols-[1fr_1fr_1fr_auto] md:items-end">
          {fields.map((f) => (
            <label key={f.label} className="block min-w-0">
              <Eyebrow>{f.label}</Eyebrow>
              <input
                readOnly
                value={f.value}
                className="mt-1 w-full truncate rounded-md border border-white/10 bg-black/60 px-3 py-2 font-mono text-xs text-zinc-200 outline-none focus:border-cyan-400/50"
              />
            </label>
          ))}
          <PrimaryButton onClick={run} disabled={busy}>
            <ScanSearch size={15} /> Run Baseline Diagnostic
          </PrimaryButton>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        {/* Failure taxonomy */}
        <Card className="p-5">
          <Eyebrow>Failure taxonomy · why V1 failed</Eyebrow>
          {busy ? (
            <div className="relative mt-4 space-y-3 overflow-hidden">
              <Skeleton className="mx-auto h-44 w-44 rounded-full" />
              {[0, 1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-5" />
              ))}
              <div className="absolute inset-x-0 h-12 animate-scan bg-gradient-to-b from-transparent via-cyan-400/15 to-transparent" />
            </div>
          ) : (
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: ran ? 1 : 0, y: 0 }} transition={SPRING}>
              <div className="relative mx-auto mt-2 h-48 w-48">
                <ResponsiveContainer width="100%" height={192}>
                  <PieChart>
                    <Pie data={data.slices} dataKey="count" nameKey="label" innerRadius={62} outerRadius={88} stroke="none" paddingAngle={2}>
                      {data.slices.map((s) => (
                        <Cell key={s.tag} fill={TAGS[s.tag].hex} />
                      ))}
                    </Pie>
                    <Tooltip content={<MonoTip />} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
                  <span className="font-mono text-2xl text-white">{total}</span>
                  <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">hard neg.</span>
                </div>
              </div>
              <ul className="mt-3 space-y-3">
                {data.slices.map((s) => (
                  <li key={s.tag}>
                    <div className="flex items-baseline justify-between text-xs">
                      <span className="text-zinc-300">{s.label}</span>
                      <span className={`font-mono ${TAGS[s.tag].text}`}>{pct(s.pct)}</span>
                    </div>
                    <div className="mt-1 h-1 overflow-hidden rounded-full bg-white/[0.06]">
                      <motion.div
                        className={`h-full ${TAGS[s.tag].bar}`}
                        initial={{ width: 0 }}
                        animate={{ width: `${s.pct * 100}%` }}
                        transition={SPRING}
                      />
                    </div>
                  </li>
                ))}
              </ul>
            </motion.div>
          )}
        </Card>

        {/* Hard negative table */}
        <Card className="overflow-hidden">
          <div className="flex items-center justify-between border-b border-white/[0.08] px-5 py-3">
            <Eyebrow>Hard negative isolation · {measured ? "worst frames (measured)" : "worst Ego4D frames"}</Eyebrow>
            <span className="font-mono text-[11px] text-zinc-500">{data.hardNegatives.length} shown</span>
          </div>
          <div className="max-h-[420px] overflow-auto">
            <table className="w-full min-w-[640px] text-left text-xs">
              <thead className="sticky top-0 z-10 bg-black/90 font-mono text-[10px] uppercase tracking-wider text-zinc-500 backdrop-blur">
                <tr>
                  <th className="px-5 py-2 font-medium">Sample ID</th>
                  <th className="px-3 py-2 font-medium">Capture mount</th>
                  <th className="px-3 py-2 font-medium">Dominant failure</th>
                  <th className="px-3 py-2 font-medium">V1 confidence</th>
                  <th className="px-3 py-2 text-right font-medium">IoU</th>
                  <th className="px-5 py-2 text-right font-medium">Action</th>
                </tr>
              </thead>
              <tbody>
                {busy
                  ? Array.from({ length: 7 }, (_, i) => (
                      <tr key={i} className="border-t border-white/5">
                        <td colSpan={6} className="px-5 py-2">
                          <Skeleton className="h-5" />
                        </td>
                      </tr>
                    ))
                  : data.hardNegatives.map((r) => (
                      <tr
                        key={r.id}
                        className={`border-t border-white/5 transition-colors hover:bg-white/[0.03] ${selected === r.id ? "bg-white/[0.05]" : ""}`}
                      >
                        <td className="px-5 py-2 font-mono text-zinc-100">{r.id}</td>
                        <td className="px-3 py-2 text-zinc-400">{r.mount}</td>
                        <td className={`px-3 py-2 ${TAGS[r.tag].text}`}>{TAGS[r.tag].label}</td>
                        <td className="px-3 py-2">
                          <div className="flex items-center gap-2">
                            <div className="h-1 w-16 overflow-hidden rounded-full bg-white/[0.06]">
                              <div className="h-full bg-red-500" style={{ width: `${r.conf * 100}%` }} />
                            </div>
                            <span className="font-mono text-red-400">{r.conf.toFixed(2)}</span>
                          </div>
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-zinc-300">{r.iou.toFixed(2)}</td>
                        <td className="px-5 py-2 text-right">
                          <button
                            onClick={() => onSelect(r.id)}
                            className="rounded border border-white/10 px-2 py-0.5 text-[11px] text-zinc-300 hover:bg-white/10"
                          >
                            Inspect
                          </button>
                        </td>
                      </tr>
                    ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {/* Active-learning banner */}
      <Card className="relative overflow-hidden p-5" active>
        <div className="flex flex-wrap items-center justify-between gap-4">
          <p className="max-w-3xl text-sm text-zinc-300">
            {measured ? (
              <>
                The baseline failed on <span className="font-mono text-white">{total}</span> of{" "}
                <span className="font-mono text-white">{prov?.n_train}</span> streamed images (IoU &lt; 0.50 or conf &lt; 0.40).
                Only these hard negatives go to targeted synthesis; the raw dataset is streamed and never stored on the laptop.
              </>
            ) : (
              <>
                Instead of downloading <span className="font-mono text-white">2.4 TB</span> of raw video, isolate the top{" "}
                <span className="font-mono text-white">500</span> hard negative frames (<span className="font-mono text-white">48 MB</span>) for
                targeted adversarial synthesis. Raw frames are purged from the temporary cache after extraction.
              </>
            )}
          </p>
          <button
            onClick={onProceed}
            className="group relative rounded-md bg-white px-5 py-2.5 text-sm font-medium text-black shadow-[0_0_32px_-4px_rgba(139,92,246,0.7)] transition hover:bg-zinc-200"
          >
            <span className="flex items-center gap-2">
              Isolate Subset &amp; Proceed to Studio <ArrowRight size={15} className="transition-transform group-hover:translate-x-0.5" />
            </span>
          </button>
        </div>
      </Card>
    </section>
  );
}
