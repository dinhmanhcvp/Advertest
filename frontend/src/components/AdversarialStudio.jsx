"use client";
import React, { useState } from "react";
import { Settings2, Activity, Play, ActivitySquare } from "lucide-react";

export default function AdversarialStudio() {
  const [frequency, setFrequency] = useState(2);
  const [amplitude, setAmplitude] = useState(5);

  // Generate SVG path for a sine wave representing the biomechanical blur
  const generateWavePath = () => {
    let d = "M 0 50 ";
    const points = 100;
    for (let i = 0; i <= points; i++) {
      const x = (i / points) * 300;
      const y = 50 + Math.sin(i * 0.1 * frequency) * amplitude * 4;
      d += `L ${x} ${y} `;
    }
    return d;
  };

  return (
    <div className="flex flex-col md:flex-row gap-4 w-full max-w-6xl mx-auto h-[600px] text-zinc-100">
      {/* Left Pane (Controls) */}
      <div className="w-full md:w-1/3 bg-lab-surface border border-lab-border rounded-lg p-6 flex flex-col gap-6">
        <div>
          <h2 className="text-lg font-sans font-semibold flex items-center gap-2 mb-1">
            <Settings2 className="w-5 h-5 text-zinc-400" />
            IMU Blur Configurator
          </h2>
          <p className="text-xs text-zinc-500 font-sans">
            Configure Biomechanical Egocentric Motion
          </p>
        </div>

        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center text-sm">
              <label className="text-zinc-400">Frequency (Hz)</label>
              <span className="font-mono text-zinc-200">{frequency} Hz</span>
            </div>
            <input
              type="range"
              min="1"
              max="10"
              value={frequency}
              onChange={(e) => setFrequency(Number(e.target.value))}
              className="w-full accent-white"
            />
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center text-sm">
              <label className="text-zinc-400">Amplitude</label>
              <span className="font-mono text-zinc-200">{amplitude}</span>
            </div>
            <input
              type="range"
              min="1"
              max="10"
              value={amplitude}
              onChange={(e) => setAmplitude(Number(e.target.value))}
              className="w-full accent-white"
            />
          </div>
        </div>

        <div className="mt-4 p-4 border border-lab-border bg-lab-bg rounded-md">
          <div className="flex items-center gap-2 mb-3 text-xs text-zinc-400 uppercase tracking-widest font-semibold">
            <ActivitySquare className="w-4 h-4" /> Trajectory Simulation
          </div>
          <svg className="w-full h-24 stroke-signal-amber fill-none" viewBox="0 0 300 100">
            <path d={generateWavePath()} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <line x1="0" y1="50" x2="300" y2="50" stroke="rgba(255,255,255,0.1)" strokeWidth="1" strokeDasharray="4" />
          </svg>
        </div>

        <button className="mt-auto bg-white text-black font-semibold rounded-md py-2.5 flex items-center justify-center gap-2 hover:bg-zinc-200 transition-colors">
          <Play className="w-4 h-4 fill-black" /> Apply Trajectory Blur
        </button>
      </div>

      {/* Right Pane (Live Preview) */}
      <div className="w-full md:w-2/3 bg-lab-surface border border-lab-border rounded-lg relative overflow-hidden flex items-center justify-center bg-noise bg-blueprint-grid">
        <div className="absolute top-4 left-4 bg-lab-bg border border-lab-border px-3 py-1.5 rounded text-xs font-mono text-zinc-300 z-10 flex items-center gap-2">
          <span>Target: WIDERFACE_042</span>
        </div>
        
        <div className="absolute top-4 right-4 bg-lab-bg border border-signal-amber text-signal-amber px-3 py-1.5 rounded text-xs font-mono font-bold z-10 shadow-[0_0_10px_rgba(245,158,11,0.2)]">
          Severity Level: {Math.min(5, Math.ceil((frequency * amplitude) / 10))}
        </div>

        {/* Fake Split Before/After Effect */}
        <div className="relative w-[90%] h-[80%] rounded-lg overflow-hidden border border-lab-border">
          {/* Before Image (Left Half) */}
          <div className="absolute inset-0 w-1/2 bg-[url('https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=800&auto=format&fit=crop&q=80')] bg-cover bg-center border-r border-lab-border z-10">
            <div className="absolute bottom-4 left-4 bg-lab-bg/80 backdrop-blur border border-lab-border px-2 py-1 rounded text-[10px] font-mono text-zinc-400">
              ORIGINAL_CLEAN
            </div>
          </div>
          
          {/* After Image (Full, blurred, shown on right half) */}
          <div className="absolute inset-0 w-full bg-[url('https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=800&auto=format&fit=crop&q=80')] bg-cover bg-center" style={{ filter: `blur(${Math.min(10, amplitude)}px)` }}>
            <div className="absolute bottom-4 right-4 bg-lab-bg/80 backdrop-blur border border-lab-border px-2 py-1 rounded text-[10px] font-mono text-signal-amber">
              AUGMENTED_T_BLUR
            </div>
          </div>
          
          {/* Divider Line */}
          <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-white/20 z-20"></div>
        </div>
      </div>
    </div>
  );
}
