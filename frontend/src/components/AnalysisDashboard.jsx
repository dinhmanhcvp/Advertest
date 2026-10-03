"use client";
import React, { useState } from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { motion } from 'framer-motion';
import { toast } from 'sonner';
import { Terminal, ShieldAlert, Cpu } from 'lucide-react';

const AnalysisDashboard = ({ analysisData, onIsolate }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  if (!analysisData) return null;

  const { failure_distribution, worst_samples, total_hard_negatives } = analysisData;
  
  const totalPages = Math.ceil(worst_samples.length / itemsPerPage);
  const currentSamples = worst_samples.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  const handleIsolate = async () => {
    setLoading(true);
    setError(null);
    const toastId = toast.loading('Isolating and generating adversarial samples...', { style: { background: '#000', color: '#fff', border: '1px solid rgba(255,255,255,0.1)' } });
    try {
      const response = await fetch('/api/v1/workflow/attack', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample_ids: worst_samples.map(s => s.id) })
      });
      if (!response.ok) throw new Error('Failed to fetch from API');
      const data = await response.json();
      toast.success('Successfully generated adversarial attack pool!', { id: toastId });
      onIsolate(data); // Pass audit_trail data to next step
    } catch (err) {
      console.error(err);
      setError(err.message);
      toast.error('Failed to isolate samples', { id: toastId });
    }
    setLoading(false);
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      className="flex flex-col gap-6 mt-6 max-w-7xl mx-auto p-4 md:p-8 bg-black min-h-screen text-zinc-100 font-sans"
    >
      {/* Header Panel */}
      <div className="bg-white/[0.02] backdrop-blur-xl border-[1px] border-white/10 rounded-2xl p-6 shadow-2xl flex flex-col md:flex-row justify-between items-start md:items-center relative overflow-hidden group">
        <div className="absolute -top-24 -left-24 w-64 h-64 bg-violet-600/20 blur-[100px] rounded-full pointer-events-none"></div>
        
        <div className="relative z-10">
          <h2 className="text-2xl font-medium tracking-tight text-white flex items-center gap-3">
            <Cpu className="w-6 h-6 text-cyan-400" />
            Visual Triage & Smart Sampling
          </h2>
          <p className="text-sm text-zinc-400 mt-2 font-mono">
            &gt; DETECTED <span className="text-fuchsia-400 font-bold">[{total_hard_negatives}]</span> HARD NEGATIVES (IoU &lt; 0.5)
          </p>
        </div>
        <button 
          onClick={handleIsolate} 
          disabled={loading}
          className="mt-4 md:mt-0 group relative flex items-center justify-center gap-2 bg-white text-black font-medium rounded-xl py-2.5 px-6 hover:bg-zinc-200 transition-colors z-10 disabled:opacity-50"
        >
          <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-violet-600 via-fuchsia-500 to-cyan-400 opacity-0 group-hover:opacity-100 blur transition-opacity duration-300 -z-10"></div>
          {loading ? (
            <><i className="fa-solid fa-circle-notch fa-spin"></i> Isolating...</>
          ) : (
            <>Fetch & Isolate Dataset</>
          )}
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-xl flex items-center gap-3 font-mono text-sm shadow-[0_0_15px_rgba(239,68,68,0.15)]">
          <ShieldAlert className="w-5 h-5 text-red-400" />
          <div>
            <h4 className="text-red-400 font-bold">SYS_ERROR</h4>
            <p className="text-red-400/80 text-xs mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Bento Box Layout Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[600px]">
        {/* Left Column: Stats & Distribution (4 cols) */}
        <div className="lg:col-span-4 flex flex-col gap-6 h-full">
          <div className="bg-white/[0.02] backdrop-blur-xl border-[1px] border-white/10 rounded-2xl p-6 shadow-2xl flex flex-col flex-1 relative overflow-hidden group hover:border-white/20 transition-colors">
            <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-400/10 blur-[50px] rounded-full pointer-events-none"></div>
            
            <h3 className="text-xs font-mono font-bold text-zinc-500 mb-4 tracking-widest uppercase flex items-center gap-2">
              <Terminal className="w-4 h-4" /> Failure Distribution
            </h3>
            
            <div className="flex-1 min-h-0 relative z-10">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={failure_distribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                    stroke="none"
                  >
                    {failure_distribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} opacity={0.8} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#000', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', color: '#fff', fontFamily: 'var(--font-mono)', fontSize: '12px', boxShadow: '0 0 20px rgba(0,0,0,0.8)' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <Legend verticalAlign="bottom" height={36} wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: '#a1a1aa' }}/>
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Right Column: Terminal Data Window (8 cols) */}
        <div className="lg:col-span-8 h-full flex flex-col bg-[#050505] border-[1px] border-white/10 rounded-2xl shadow-[0_0_30px_rgba(0,0,0,0.8)] relative overflow-hidden">
          
          {/* macOS Style Terminal Header */}
          <div className="bg-white/[0.02] border-b border-white/10 px-4 py-3 flex items-center justify-between">
            <div className="flex gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
              <div className="w-3 h-3 rounded-full bg-green-500/80"></div>
            </div>
            <div className="text-[11px] font-mono text-zinc-500">audit_trail.json — {worst_samples.length} items</div>
            <div className="w-10"></div> {/* Spacer for symmetry */}
          </div>

          <div className="flex-1 overflow-y-auto min-h-0 p-4 custom-scrollbar">
            <table className="w-full text-left border-collapse font-mono text-sm">
              <thead>
                <tr className="sticky top-0 bg-[#050505] z-10 text-[10px] text-zinc-500 uppercase tracking-widest">
                  <th className="pb-4 font-medium px-2">ID</th>
                  <th className="pb-4 font-medium px-2">Target</th>
                  <th className="pb-4 font-medium px-2">IoU</th>
                  <th className="pb-4 font-medium px-2">Classification</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {currentSamples.map((sample) => {
                  const distData = failure_distribution.find(d => d.name === sample.reason);
                  const colorCode = distData?.color || '#94a3b8';
                  
                  return (
                    <tr key={sample.id} className="hover:bg-white/[0.02] transition-colors group">
                      <td className="py-3 px-2 text-cyan-400/70 group-hover:text-cyan-400 transition-colors">
                        {sample.id.split('-')[0]}
                      </td>
                      <td className="py-3 px-2 text-zinc-400 truncate max-w-[120px]">
                        &quot;{sample.image}&quot;
                      </td>
                      <td className="py-3 px-2">
                        <span className="text-fuchsia-400">
                          {sample.iou.toFixed(3)}
                        </span>
                      </td>
                      <td className="py-3 px-2 text-zinc-300">
                         <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border bg-black" style={{ borderColor: `${colorCode}40` }}>
                            <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: colorCode }}></div>
                            <span className="text-[11px] tracking-wide" style={{ color: colorCode }}>{sample.reason}</span>
                         </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Terminal Footer / Pagination */}
          {totalPages > 1 && (
            <div className="flex justify-between items-center p-3 border-t border-white/10 bg-white/[0.02]">
              <button 
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="text-xs font-mono text-zinc-500 hover:text-white disabled:opacity-30 transition-colors"
              >
                &lt; PREV
              </button>
              <div className="flex items-center gap-2">
                {/* Subtle Grid Pattern inside the pagination area for aesthetic */}
                <div className="text-[10px] text-zinc-600 font-mono tracking-widest">
                  PAGE {currentPage.toString().padStart(2, '0')} / {totalPages.toString().padStart(2, '0')}
                </div>
              </div>
              <button 
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="text-xs font-mono text-zinc-500 hover:text-white disabled:opacity-30 transition-colors"
              >
                NEXT &gt;
              </button>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default AnalysisDashboard;
