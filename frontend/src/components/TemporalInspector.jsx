"use client";
import React, { useState } from "react";
import { Clock, ShieldAlert, Crosshair, ToggleLeft, ToggleRight } from "lucide-react";

export default function TemporalInspector() {
  const [trackerEnabled, setTrackerEnabled] = useState(false);
  const frames = [
    { label: "t-2", active: false },
    { label: "t-1", active: false },
    { label: "t", active: true },
    { label: "t+1", active: false },
    { label: "t+2", active: false },
  ];

  return (
    <div className="flex flex-col gap-4 w-full max-w-5xl mx-auto h-[700px] text-zinc-100 bg-lab-bg p-4">
      {/* Top Header / Controls */}
      <div className="flex justify-between items-center bg-lab-surface border border-lab-border rounded-lg p-4">
        <div className="flex items-center gap-3">
          <Clock className="w-5 h-5 text-zinc-400" />
          <div>
            <h2 className="text-sm font-semibold font-sans">Temporal Inspector</h2>
            <p className="text-xs text-zinc-500 font-sans">Sequence Analysis & Velocity Tracking</p>
          </div>
        </div>
        
        <button 
          onClick={() => setTrackerEnabled(!trackerEnabled)}
          className="flex items-center gap-3 bg-zinc-950 border border-lab-border px-4 py-2 rounded-md hover:bg-zinc-800 transition-colors"
        >
          <span className="text-sm font-mono text-zinc-300">Enable Temporal Tracker</span>
          {trackerEnabled ? (
            <ToggleRight className="w-6 h-6 text-signal-teal" />
          ) : (
            <ToggleLeft className="w-6 h-6 text-zinc-600" />
          )}
        </button>
      </div>

      {/* Main Stage */}
      <div className="flex-1 bg-lab-surface border border-lab-border rounded-lg relative overflow-hidden flex items-center justify-center bg-noise bg-blueprint-grid">
        {/* Status Badge */}
        <div className="absolute top-4 left-4 z-20">
          {trackerEnabled ? (
            <div className="bg-signal-teal/10 border border-signal-teal/30 text-signal-teal px-3 py-1.5 rounded-md text-xs font-mono flex items-center gap-2">
              <Crosshair className="w-4 h-4" /> Box Recovered via Velocity Tracking
            </div>
          ) : (
            <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-3 py-1.5 rounded-md text-xs font-mono flex items-center gap-2">
              <ShieldAlert className="w-4 h-4" /> YOLO Detection Failed
            </div>
          )}
        </div>

        {/* Enlarged Frame Image */}
        <div className="relative w-[80%] h-[80%] bg-[url('https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=800&auto=format&fit=crop&q=80')] bg-cover bg-center rounded border border-lab-border" style={{ filter: "blur(4px)" }}>
          {/* Bounding Box (Only when Tracker is ON) */}
          {trackerEnabled && (
            <div className="absolute border-2 border-dashed border-signal-teal bg-signal-teal/10 z-10 flex items-start"
                 style={{ left: '35%', top: '25%', width: '30%', height: '50%' }}>
              <div className="bg-signal-teal text-black text-[10px] font-mono font-bold px-1.5 py-0.5 absolute -top-5 left-[-2px]">
                ID:42 conf:0.89
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Bar: Filmstrip */}
      <div className="h-28 bg-lab-surface border border-lab-border rounded-lg p-3 flex gap-3 overflow-x-auto">
        {frames.map((frame, i) => (
          <div 
            key={i} 
            className={`flex-1 relative rounded border overflow-hidden flex items-center justify-center bg-zinc-950 ${
              frame.active ? "border-signal-amber shadow-[0_0_10px_rgba(245,158,11,0.2)]" : "border-lab-border opacity-50"
            }`}
          >
            <div 
              className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=200&auto=format&fit=crop&q=80')] bg-cover bg-center" 
              style={{ filter: frame.active ? "blur(4px)" : "none" }}
            />
            <div className="absolute bottom-1 right-1 bg-black/80 px-1.5 py-0.5 rounded text-[10px] font-mono text-zinc-300 z-10 border border-lab-border">
              {frame.label}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
