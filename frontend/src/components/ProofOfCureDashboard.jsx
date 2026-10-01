"use client";
import React from "react";
import { DownloadCloud, ArrowUpRight, CheckCircle2, AlertTriangle, ShieldCheck } from "lucide-react";

export default function ProofOfCureDashboard({ retrainData }) {
  // Mock data to demonstrate the UI structure requested
  const metrics = {
    baselineMap: 92.4,
    corruptedMap: 90.1,
    delta: 27.5,
    framesRecovered: 1420
  };

  const lineageData = [
    { id: "WF_042", errorClass: "False Negative", aug: "Biomechanical Blur (4Hz)", delta: "+0.45" },
    { id: "WF_089", errorClass: "False Negative", aug: "Overexposure (Sev 3)", delta: "+0.38" },
    { id: "EGO_112", errorClass: "BBox Drift", aug: "Motion Blur (Linear)", delta: "+0.29" },
    { id: "EGO_404", errorClass: "False Positive", aug: "Gaussian Noise", delta: "+0.15" },
  ];

  return (
    <div className="flex flex-col gap-4 w-full max-w-6xl mx-auto text-zinc-100 bg-lab-bg p-4 font-sans">
      <div className="flex justify-between items-end mb-2">
        <div>
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <ShieldCheck className="w-6 h-6 text-signal-teal" /> Proof of Cure Validation
          </h2>
          <p className="text-sm text-zinc-500 mt-1">Validation on held-out edge cases</p>
        </div>
        <button className="bg-white text-black font-semibold py-2.5 px-5 rounded-md flex items-center gap-2 hover:bg-zinc-200 transition-colors shadow-[0_0_15px_rgba(16,185,129,0.5)]">
          <DownloadCloud className="w-5 h-5" /> Export Model & Lineage to Data Lake
        </button>
      </div>

      {/* Top Row: Metrics (Bento Box) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Baseline mAP */}
        <div className="bg-lab-surface border border-lab-border rounded-lg p-5 flex flex-col justify-between">
          <span className="text-sm text-zinc-400 font-semibold uppercase tracking-wider">Baseline mAP</span>
          <div className="mt-4 flex items-end gap-2">
            <span className="font-mono text-4xl text-zinc-200 font-light">{metrics.baselineMap}%</span>
            <span className="text-sm font-mono text-zinc-500 mb-1">STABLE</span>
          </div>
        </div>

        {/* Corrupted mAP */}
        <div className="bg-lab-surface border border-lab-border rounded-lg p-5 flex flex-col justify-between relative overflow-hidden">
          <span className="text-sm text-zinc-400 font-semibold uppercase tracking-wider">Corrupted mAP</span>
          <div className="mt-4 flex items-end gap-3 z-10">
            <span className="font-mono text-4xl text-zinc-200 font-light">{metrics.corruptedMap}%</span>
            <span className="text-lg font-mono text-signal-teal flex items-center mb-1 font-bold">
              <ArrowUpRight className="w-5 h-5 mr-1" /> +{metrics.delta}%
            </span>
          </div>
          {/* Subtle bg glow for emphasis without breaking the strict UI */}
          <div className="absolute -bottom-4 -right-4 w-24 h-24 bg-signal-teal/10 blur-2xl rounded-full"></div>
        </div>

        {/* Frames Recovered */}
        <div className="bg-lab-surface border border-lab-border rounded-lg p-5 flex flex-col justify-between">
          <span className="text-sm text-zinc-400 font-semibold uppercase tracking-wider">Frames Recovered</span>
          <div className="mt-4 flex items-end gap-2">
            <span className="font-mono text-4xl text-zinc-200 font-light">{metrics.framesRecovered}</span>
            <span className="text-sm font-mono text-signal-teal mb-1 flex items-center gap-1">
              <CheckCircle2 className="w-4 h-4" /> VERIFIED
            </span>
          </div>
        </div>
      </div>

      {/* Bottom Row: Data Lineage Table */}
      <div className="bg-lab-surface border border-lab-border rounded-lg overflow-hidden mt-2 flex flex-col">
        <div className="p-4 border-b border-lab-border flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-signal-amber" />
          <h3 className="text-sm font-semibold text-zinc-300">Top Corrected Edge Cases</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm whitespace-nowrap">
            <thead className="bg-zinc-950/50 border-b border-lab-border text-zinc-400">
              <tr>
                <th className="px-6 py-3 font-semibold uppercase tracking-wider text-xs">Sample ID</th>
                <th className="px-6 py-3 font-semibold uppercase tracking-wider text-xs">Error Class</th>
                <th className="px-6 py-3 font-semibold uppercase tracking-wider text-xs">Applied IMU Augmentation</th>
                <th className="px-6 py-3 font-semibold uppercase tracking-wider text-xs text-right">Confidence Delta</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-lab-border text-zinc-300">
              {lineageData.map((row, idx) => (
                <tr key={idx} className="hover:bg-zinc-800/30 transition-colors">
                  <td className="px-6 py-4 font-mono text-zinc-400">{row.id}</td>
                  <td className="px-6 py-4">
                    <span className="bg-zinc-800 border border-lab-border px-2 py-1 rounded text-xs">
                      {row.errorClass}
                    </span>
                  </td>
                  <td className="px-6 py-4 font-mono text-xs">{row.aug}</td>
                  <td className="px-6 py-4 font-mono text-right text-signal-teal">{row.delta}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
