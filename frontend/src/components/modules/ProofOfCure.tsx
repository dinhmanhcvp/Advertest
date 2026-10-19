"use client";
import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowDownRight, ArrowUpRight, Check, DatabaseZap, PackageCheck, X } from "lucide-react";
import CVFrame from "@/components/ui/CVFrame";
import DiffSlider from "@/components/ui/DiffSlider";
import { Badge, Card, cx, Eyebrow, PrimaryButton, SectionTitle, SPRING } from "@/components/ui/primitives";
import { landmarksFor, pct, TAGS } from "@/lib/taxonomy";
import type { CureCase, FailureTag, WorkspaceData } from "@/lib/types";

const signed = (v: number, d = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}%`;

function effects(tag: FailureTag) {
  switch (tag) {
    case "error_fisheye":
      return { fisheye: 0.85, blur: 0.8 };
    case "error_motion_blur":
      return { blur: 3.4, smear: [[-5, 0.4], [-2.5, -0.6], [0, 0.2], [2.5, -0.5], [5, 0.3]] as [number, number][] };
    case "error_sensor_noise":
      return { exposure: 1.95, grain: 0.7 };
    default:
      return { occlusion: true };
  }
}

function Scene({ c, side }: { c: CureCase; side: "v1" | "v2" }) {
  const m = side === "v1" ? c.model_v1 : c.model_v2;
  const face = c.model_v2.bbox;
  const noDet = side === "v1" && m.confidence <= 0; // V1 produced nothing for this face: show where it should be
  const [x, y, w, h] = noDet ? face : m.bbox;
  const color = side === "v1" ? "#ef4444" : "#10b981";
  return (
    <CVFrame box={face} {...(c.image ? {} : effects(c.failure_tag))} imageUrl={c.image} hud={`${c.case_id} · ${c.condition.toUpperCase()}`} className="h-full w-full">
      <g>
        <rect x={x} y={y} width={w} height={h} fill="none" stroke={color} strokeWidth="1" strokeDasharray={side === "v1" ? "3 2" : undefined} />
        <text x={x} y={y - 1.8} fill={color} fontSize="3.4" fontFamily="var(--font-jb-mono), monospace">
          {side === "v1" ? "V1" : "V2"} · {noDet ? "no detection" : `conf ${m.confidence.toFixed(2)}`} [{side === "v1" ? "FAILED" : "SUCCESS"}]
        </text>
        {side === "v2" &&
          (m.landmarks ?? landmarksFor(m.bbox)).map(([lx, ly], i) => (
            <circle key={i} cx={lx} cy={ly} r="1.1" fill="#06b6d4" style={{ filter: "drop-shadow(0 0 2px #06b6d4)" }} />
          ))}
      </g>
    </CVFrame>
  );
}

function Metric({
  eyebrow,
  from,
  to,
  badge,
  tone,
  up,
}: {
  eyebrow: string;
  from?: string;
  to: string;
  badge: string;
  tone: "cure" | "fail" | "neutral";
  up?: boolean;
}) {
  return (
    <Card className="p-5">
      <Eyebrow>{eyebrow}</Eyebrow>
      <p className="mt-3 flex items-baseline gap-2 font-mono">
        {from && (
          <>
            <span className="text-lg text-zinc-500 line-through decoration-zinc-700">{from}</span>
            <span className="text-zinc-600">→</span>
          </>
        )}
        <span className="text-4xl font-medium text-white">{to}</span>
      </p>
      <Badge tone={tone} className="mt-4">
        {up === undefined ? null : up ? <ArrowUpRight size={11} /> : <ArrowDownRight size={11} />}
        {badge}
      </Badge>
    </Card>
  );
}

export default function ProofOfCure({ data }: { data: WorkspaceData }) {
  const { report, cases } = data;
  const [caseIdx, setCaseIdx] = useState(0);
  const [modal, setModal] = useState(false);
  const c = cases[Math.min(caseIdx, cases.length - 1)];

  const baseDrop = (report.base_model.base_map - report.advertest_model.base_map) * 100;
  const gain = (report.advertest_model.corrupted_map - report.base_model.corrupted_map) * 100;
  const gatePass = baseDrop <= report.gate.max_base_drop * 100 + 1e-9 && gain >= report.gate.min_corrupted_gain * 100;
  const confGain = c.model_v2.confidence - c.model_v1.confidence;
  const total = data.slices.reduce((a, s) => a + s.count, 0);

  const lineage = [
    { file: "audit_trail.json", rows: `${total} records`, note: "why each attack chain + severity was chosen", tone: "cure" as const, state: "READY" },
    { file: "proof_of_cure.json", rows: `${cases.length} A/B cases`, note: "V1 vs V2 boxes, confidence, 5-pt landmarks", tone: "cure" as const, state: "READY" },
    { file: "yolov7-face-v2.pt", rows: "model weights", note: "released by the GPU worker after the gate passes", tone: gatePass ? ("cure" as const) : ("fail" as const), state: gatePass ? "GATE PASSED" : "GATE FAILED" },
  ];

  const download = () => {
    const manifest = {
      exported_at: new Date().toISOString(),
      destination: "core-data-lake://advertest/lineage",
      schema: "ENTERPRISE_PII_V2",
      gate: { base_map_drop_pp: Number(baseDrop.toFixed(2)), corrupted_map_gain_pp: Number(gain.toFixed(2)), accepted: gatePass },
      failure_taxonomy: data.slices.map((s) => ({ tag: s.tag, count: s.count })),
      evaluation: report,
      proof_of_cure: cases,
    };
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(manifest, null, 2)], { type: "application/json" }));
    a.download = "advertest_lineage_manifest.json";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <section className="space-y-4">
      <SectionTitle eyebrow="Stage 04 · Proof of Cure & Enterprise Handover" title="Blind evaluation gate" aside={<Badge tone={gatePass ? "cure" : "fail"}>{gatePass ? "model v2 accepted" : "model v2 rejected"}</Badge>} />

      <div className="grid gap-4 md:grid-cols-3">
        <Metric
          eyebrow="Base domain mAP · WIDER FACE"
          from={pct(report.base_model.base_map)}
          to={pct(report.advertest_model.base_map)}
          badge={`Δ ${signed(-baseDrop)} | ${baseDrop <= 1 ? "PASSED ≤ 1.0% DROP GATE" : "FAILED GATE"}`}
          tone={baseDrop <= 1 ? "cure" : "fail"}
          up={baseDrop > 0}
        />
        <Metric
          eyebrow={`Target mAP · ${report.provenance?.target_domain?.includes("proxy") ? "egocentric proxy (WIDER + physical attacks)" : "Ego4D"}`}
          from={pct(report.base_model.corrupted_map)}
          to={pct(report.advertest_model.corrupted_map)}
          badge={`▲ ${signed(gain)} robustness gain`}
          tone="cure"
        />
        <Metric eyebrow="Hard negatives cured" to={pct(report.hard_negative_recovery)} badge={`of ${total} isolated hard negatives`} tone="neutral" />
      </div>

      <Card className="p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap gap-2">
            {cases.map((k, i) => (
              <button
                key={k.case_id}
                onClick={() => setCaseIdx(i)}
                className={cx(
                  "relative rounded-md border px-3 py-1.5 text-xs transition",
                  i === caseIdx ? "border-white/40 text-white" : "border-white/10 text-zinc-500 hover:text-zinc-300",
                )}
              >
                {i === caseIdx && <motion.span layoutId="case-pill" transition={SPRING} className="absolute inset-0 rounded-md bg-white/[0.07]" />}
                <span className="relative">{k.condition}</span>
              </button>
            ))}
          </div>
          <span className="font-mono text-[11px] text-zinc-500">{c.case_id} · drag the divider</span>
        </div>

        <div className="relative mx-auto max-w-[760px]">
          <DiffSlider className="aspect-[4/3] w-full" left={<Scene c={c} side="v1" />} right={<Scene c={c} side="v2" />} />
          <span className="pointer-events-none absolute left-3 top-3 rounded bg-black/70 px-2 py-1 font-mono text-[10px] text-red-400">MODEL V1 · BASELINE</span>
          <span className="pointer-events-none absolute right-3 top-3 rounded bg-black/70 px-2 py-1 font-mono text-[10px] text-emerald-400">MODEL V2 · ADVERTEST RETRAINED</span>
          <motion.span
            key={c.case_id}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={SPRING}
            className="pointer-events-none absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full border border-emerald-400/40 bg-emerald-500/15 px-3 py-1 font-mono text-xs text-emerald-300 shadow-[0_0_28px_-4px_rgba(16,185,129,0.8)] backdrop-blur"
          >
            +{confGain.toFixed(2)} Confidence Gain
          </motion.span>
        </div>

        <div className="mt-3 grid gap-3 font-mono text-[11px] sm:grid-cols-2">
          <p className="rounded-md border border-red-500/20 bg-red-500/[0.04] px-3 py-2 text-red-300">
            V1 bbox x:{c.model_v1.bbox[0]}, y:{c.model_v1.bbox[1]}, w:{c.model_v1.bbox[2]}, h:{c.model_v1.bbox[3]} · conf {c.model_v1.confidence.toFixed(2)} [FAILED]
          </p>
          <p className="rounded-md border border-emerald-500/20 bg-emerald-500/[0.04] px-3 py-2 text-emerald-300">
            V2 bbox x:{c.model_v2.bbox[0]}, y:{c.model_v2.bbox[1]}, w:{c.model_v2.bbox[2]}, h:{c.model_v2.bbox[3]} · conf {c.model_v2.confidence.toFixed(2)} [SUCCESS]
          </p>
        </div>
        <p className={cx("mt-2 font-mono text-[11px]", TAGS[c.failure_tag].text)}>failure_tag: {c.failure_tag}</p>
      </Card>

      <Card className="p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <DatabaseZap size={16} className="text-zinc-400" />
            <Eyebrow>Data lineage · handover to Core Data Lake</Eyebrow>
          </div>
          <Badge tone="cure">
            <Check size={11} /> Schema mapped: ENTERPRISE_PII_V2
          </Badge>
        </div>
        <table className="mt-4 w-full text-left text-xs">
          <thead className="font-mono text-[10px] uppercase tracking-wider text-zinc-500">
            <tr>
              <th className="py-2 font-medium">Artifact</th>
              <th className="py-2 font-medium">Content</th>
              <th className="hidden py-2 font-medium md:table-cell">Lineage</th>
              <th className="py-2 text-right font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {lineage.map((r) => (
              <tr key={r.file} className="border-t border-white/5">
                <td className="py-2.5 font-mono text-zinc-100">{r.file}</td>
                <td className="py-2.5 font-mono text-zinc-400">{r.rows}</td>
                <td className="hidden py-2.5 text-zinc-500 md:table-cell">{r.note}</td>
                <td className="py-2.5 text-right">
                  <Badge tone={r.tone}>{r.state}</Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-5 flex justify-end">
          <PrimaryButton onClick={() => setModal(true)} disabled={!gatePass}>
            <PackageCheck size={15} /> Export Cured Weights &amp; Lineage JSON to Data Lake
          </PrimaryButton>
        </div>
      </Card>

      <AnimatePresence>
        {modal && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setModal(false)}
          >
            <motion.div
              role="dialog"
              aria-modal="true"
              aria-label="Export package"
              onClick={(e) => e.stopPropagation()}
              initial={{ scale: 0.96, y: 12, opacity: 0 }}
              animate={{ scale: 1, y: 0, opacity: 1 }}
              exit={{ scale: 0.96, opacity: 0 }}
              transition={SPRING}
              className="w-full max-w-md rounded-lg border border-white/10 bg-zinc-950 p-5 shadow-2xl"
            >
              <div className="flex items-start justify-between">
                <div>
                  <Eyebrow>Handover package</Eyebrow>
                  <h3 className="mt-1 text-lg font-semibold tracking-tight text-white">Ready for the Core Data Lake</h3>
                </div>
                <button aria-label="Close" onClick={() => setModal(false)} className="rounded p-1 text-zinc-500 hover:bg-white/10 hover:text-white">
                  <X size={16} />
                </button>
              </div>
              <ul className="mt-4 space-y-2 font-mono text-xs">
                {lineage.map((r, i) => (
                  <motion.li
                    key={r.file}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ ...SPRING, delay: 0.12 * i }}
                    className="flex items-center justify-between rounded-md border border-white/[0.08] px-3 py-2"
                  >
                    <span className="text-zinc-200">{r.file}</span>
                    <span className="flex items-center gap-1 text-emerald-400">
                      <Check size={12} /> {r.state}
                    </span>
                  </motion.li>
                ))}
              </ul>
              <p className="mt-3 text-xs text-zinc-500">
                Download writes the lineage manifest (gate result, failure taxonomy, A/B cases). Model weights are released by the GPU worker once the gate passes.
              </p>
              <div className="mt-5 flex justify-end gap-2">
                <button onClick={() => setModal(false)} className="rounded-md border border-white/10 px-3 py-2 text-sm text-zinc-300 hover:bg-white/5">
                  Close
                </button>
                <PrimaryButton onClick={download}>Download lineage manifest</PrimaryButton>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
