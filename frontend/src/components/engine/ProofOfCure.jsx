"use client";
import { useState } from "react";
import { toast } from "sonner";
import { ErrorState, Eyebrow, Glass, Skeleton, useApi } from "./ui";

const pct = (v) => `${(v * 100).toFixed(1)}%`;

export default function ProofOfCure() {
  const { data, loading, error, reload } = useApi("/api/v1/eval-report");
  const poc = useApi("/api/v1/demo/proof-of-cure");
  const [exporting, setExporting] = useState(false);

  const baseDrop = data ? data.base_model.base_map - data.advertest_model.base_map : 0;
  const gain = data ? data.advertest_model.corrupted_map - data.base_model.corrupted_map : 0;
  const gatePass = data && baseDrop <= 0.01 + 1e-9 && gain > 0;

  const exportBundle = () => {
    setExporting(true);
    const bundle = {
      exported_at: new Date().toISOString(),
      destination: "core-data-lake://advertest/lineage",
      gate: { base_map_drop: baseDrop, corrupted_map_gain: gain, accepted: gatePass },
      evaluation: data,
      proof_of_cure: poc.data,
    };
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "advertest_lineage_bundle.json";
    a.click();
    URL.revokeObjectURL(a.href);
    toast.success("Lineage bundle exported (handover manifest ready for the Core Data Lake).");
    setExporting(false);
  };

  return (
    <Glass className="p-5">
      <Eyebrow>Module 05 · Proof of Cure</Eyebrow>
      <h2 className="mt-1 text-lg font-semibold tracking-tight text-white">Blind Evaluation Gate</h2>

      {error && <div className="mt-4"><ErrorState message={error} onRetry={reload} /></div>}
      {loading && (
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <Skeleton className="h-32" /><Skeleton className="h-32" /><Skeleton className="h-32" />
        </div>
      )}

      {data && (
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <div className="rounded-md border border-white/10 p-4">
            <Eyebrow>Base mAP · WIDER FACE</Eyebrow>
            <p className="mt-2 font-mono text-3xl text-white">{pct(data.advertest_model.base_map)}</p>
            <p className="mt-1 font-mono text-xs text-zinc-500">was {pct(data.base_model.base_map)}</p>
            <span className={`mt-3 inline-block rounded-full border px-2 py-0.5 font-mono text-[11px] ${
              baseDrop <= 0.01 + 1e-9 ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400" : "border-red-500/30 bg-red-500/10 text-red-400"
            }`}>
              drop {pct(Math.max(0, baseDrop))} · limit ≤ 1.0%
            </span>
          </div>

          <div className="rounded-md border border-white/10 p-4">
            <Eyebrow>Corrupted mAP · Egocentric (mPC)</Eyebrow>
            <p className="mt-2 font-mono text-3xl text-white">{pct(data.advertest_model.corrupted_map)}</p>
            <p className="mt-1 font-mono text-xs text-zinc-500">was {pct(data.base_model.corrupted_map)}</p>
            <span className="mt-3 inline-block rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 font-mono text-[11px] text-emerald-400">
              +{pct(gain)}
            </span>
          </div>

          <div className="flex flex-col justify-between rounded-md border border-white/10 p-4">
            <div>
              <Eyebrow>Gate verdict</Eyebrow>
              <p className={`mt-2 font-mono text-xl ${gatePass ? "text-emerald-400" : "text-red-400"}`}>
                {gatePass ? "MODEL V2 ACCEPTED" : "MODEL V2 REJECTED"}
              </p>
              <p className="mt-1 text-xs text-zinc-500">
                Accepted only if corrupted mAP rises and base mAP drops by ≤ 1%.
              </p>
            </div>
            <button
              onClick={exportBundle}
              disabled={!gatePass || exporting}
              className="mt-4 rounded-md bg-white px-4 py-2 text-sm font-medium text-black transition hover:bg-zinc-200 disabled:opacity-40"
            >
              Export Lineage &amp; Model to Enterprise Data Lake
            </button>
          </div>
        </div>
      )}
    </Glass>
  );
}
