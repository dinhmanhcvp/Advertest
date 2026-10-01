"use client";
import React, { useState } from "react";
import { Settings2, Activity, Play, ActivitySquare } from "lucide-react";
import { motion } from "framer-motion";

export default function AdversarialStudio() {
  const [frequency, setFrequency] = useState(2);
  const [amplitude, setAmplitude] = useState(5);

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
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="w-full max-w-7xl mx-auto h-[700px] text-zinc-100 bg-black p-4 md:p-8 font-sans"
    >
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6 h-full">
        {/* Left Column: Controls (Bento Box Panel) */}
        <div className="md:col-span-4 flex flex-col gap-6">
          <div className="bg-white/[0.02] backdrop-blur-xl border-[1px] border-white/10 rounded-2xl p-6 flex flex-col gap-6 h-full hover:border-white/20 transition-colors duration-300 shadow-[0_0_20px_rgba(0,0,0,0.5)] relative overflow-hidden group">
            {/* Subtle Gradient Glow inside the card */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-3/4 h-24 bg-violet-600/10 blur-3xl rounded-full pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-700"></div>
            
            <div>
              <h2 className="text-xl font-medium tracking-tight text-white flex items-center gap-2 mb-1">
                <Settings2 className="w-5 h-5 text-zinc-400" />
                IMU Blur Configurator
              </h2>
              <p className="text-sm text-zinc-400">
                Configure Biomechanical Egocentric Motion
              </p>
            </div>

            <div className="flex flex-col gap-6 flex-1">
              {/* Frequency Control */}
              <div className="flex flex-col gap-3">
                <div className="flex justify-between items-center text-sm">
                  <label className="text-zinc-400 tracking-tight">Frequency</label>
                  <span className="font-mono text-cyan-400 bg-cyan-400/10 border border-cyan-400/20 rounded-full px-2 py-0.5 text-xs">{frequency} Hz</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={frequency}
                  onChange={(e) => setFrequency(Number(e.target.value))}
                  className="w-full accent-violet-500 h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              {/* Amplitude Control */}
              <div className="flex flex-col gap-3">
                <div className="flex justify-between items-center text-sm">
                  <label className="text-zinc-400 tracking-tight">Amplitude</label>
                  <span className="font-mono text-fuchsia-400 bg-fuchsia-400/10 border border-fuchsia-400/20 rounded-full px-2 py-0.5 text-xs">{amplitude}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={amplitude}
                  onChange={(e) => setAmplitude(Number(e.target.value))}
                  className="w-full accent-fuchsia-500 h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            </div>

            <div className="mt-4 p-4 border-[1px] border-white/10 bg-black/50 rounded-xl relative overflow-hidden">
              <div className="flex items-center gap-2 mb-3 text-[10px] text-zinc-500 uppercase tracking-widest font-mono">
                <ActivitySquare className="w-4 h-4" /> Trajectory Matrix
              </div>
              <svg className="w-full h-24 stroke-cyan-400/80 fill-none" viewBox="0 0 300 100">
                <path d={generateWavePath()} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="drop-shadow-[0_0_8px_rgba(34,211,238,0.5)]" />
                <line x1="0" y1="50" x2="300" y2="50" stroke="rgba(255,255,255,0.05)" strokeWidth="1" strokeDasharray="4" />
              </svg>
            </div>

            <button className="mt-auto group relative w-full flex items-center justify-center gap-2 bg-white text-black font-medium rounded-xl py-3 hover:bg-zinc-200 transition-colors">
              <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-violet-600 via-fuchsia-500 to-cyan-400 opacity-0 group-hover:opacity-100 blur transition-opacity duration-300 -z-10"></div>
              <Play className="w-4 h-4 fill-black" /> Apply Trajectory
            </button>
          </div>
        </div>

        {/* Right Column: Live Preview & Images (Bento Box Main) */}
        <div className="md:col-span-8 bg-black border-[1px] border-white/10 rounded-2xl relative overflow-hidden flex items-center justify-center bg-blueprint-grid">
          {/* Subtle Glow Behind Images */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-cyan-500/10 blur-[100px] pointer-events-none rounded-full"></div>
          
          {/* Pill Badges overlay */}
          <div className="absolute top-6 left-6 flex gap-2 z-10">
            <div className="bg-black/60 backdrop-blur-md border border-white/10 px-3 py-1 rounded-full text-xs font-mono text-zinc-300">
              Target: <span className="text-white">WIDERFACE_042</span>
            </div>
          </div>
          
          <div className="absolute top-6 right-6 flex gap-2 z-10">
            <div className="bg-amber-500/10 backdrop-blur-md border border-amber-500/30 text-amber-400 px-3 py-1 rounded-full text-xs font-mono">
              Level: {Math.min(5, Math.ceil((frequency * amplitude) / 10))}
            </div>
            <div className="bg-violet-500/10 backdrop-blur-md border border-violet-500/30 text-violet-400 px-3 py-1 rounded-full text-xs font-mono">
              [T_Blur]
            </div>
          </div>

          {/* Fake Split Before/After Effect */}
          <div className="relative w-[85%] h-[75%] rounded-xl overflow-hidden border border-white/10 shadow-[0_0_50px_rgba(0,0,0,0.8)]">
            {/* Before Image (Left Half) */}
            <div className="absolute inset-0 w-1/2 bg-[url('https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=800&auto=format&fit=crop&q=80')] bg-cover bg-center border-r border-white/10 z-10">
              <div className="absolute bottom-4 left-4 bg-black/60 backdrop-blur-md border border-white/10 px-2 py-1 rounded-full text-[10px] font-mono text-zinc-400">
                ORIGINAL_CLEAN
              </div>
            </div>
            
            {/* After Image (Full, blurred, shown on right half) */}
            <div className="absolute inset-0 w-full bg-[url('https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=800&auto=format&fit=crop&q=80')] bg-cover bg-center" style={{ filter: `blur(${Math.min(10, amplitude)}px)` }}>
              <div className="absolute bottom-4 right-4 bg-cyan-500/10 backdrop-blur-md border border-cyan-500/30 px-2 py-1 rounded-full text-[10px] font-mono text-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.2)]">
                AUGMENTED_SYNTHETIC
              </div>
            </div>
            
            {/* Divider Line */}
            <div className="absolute top-0 bottom-0 left-1/2 w-[1px] bg-white/20 z-20 shadow-[0_0_10px_rgba(255,255,255,0.5)]"></div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
