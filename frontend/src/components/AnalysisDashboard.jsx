"use client";
import React, { useState } from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { motion } from 'framer-motion';
import { toast } from 'sonner';

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
    const toastId = toast.loading('Isolating and generating adversarial samples...');
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
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="flex flex-col gap-6 mt-6 max-w-5xl mx-auto"
    >
      {/* Header */}
      <div className="bg-slate-900 border border-slate-700/50 rounded-2xl p-6 shadow-lg flex justify-between items-center">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-3">
            <i className="fa-solid fa-chart-pie text-indigo-400"></i>
            Visual Triage & Smart Sampling
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Found <span className="text-red-400 font-bold">{total_hard_negatives}</span> Hard Negatives (IoU &lt; 0.5) in the Validation Dataset.
          </p>
        </div>
        <button 
          onClick={handleIsolate} 
          disabled={loading}
          className="bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-2 px-6 rounded-lg shadow-[0_0_15px_rgba(79,70,229,0.3)] transition-all flex items-center gap-2 disabled:opacity-50"
        >
          {loading ? (
            <><i className="fa-solid fa-circle-notch fa-spin"></i> Isolating...</>
          ) : (
            <><i className="fa-solid fa-filter"></i> Fetch & Isolate Failed Samples</>
          )}
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-xl flex items-center gap-3">
          <i className="fa-solid fa-triangle-exclamation text-red-400 text-xl"></i>
          <div>
            <h4 className="text-red-400 font-bold">Failed to Fetch Action</h4>
            <p className="text-red-400/80 text-xs">Error: {error}. Check backend logs.</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Failure Distribution Chart */}
        <div className="bg-slate-900 border border-slate-700/50 rounded-2xl p-6 shadow-lg flex flex-col h-96">
          <h3 className="text-sm font-bold text-slate-300 mb-4 border-b border-slate-800 pb-2">Failure Distribution</h3>
          <div className="flex-1 min-h-0 relative">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={failure_distribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                  stroke="none"
                >
                  {failure_distribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.5rem', color: '#f8fafc' }}
                  itemStyle={{ color: '#f8fafc' }}
                />
                <Legend verticalAlign="bottom" height={36} wrapperStyle={{ fontSize: '12px', color: '#cbd5e1' }}/>
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Worst Performing Samples Table */}
        <div className="bg-slate-900 border border-slate-700/50 rounded-2xl p-6 shadow-lg flex flex-col h-96">
          <h3 className="text-sm font-bold text-slate-300 mb-4 border-b border-slate-800 pb-2">
            Top 100 Worst Samples (IoU &lt; 0.5)
          </h3>
          <div className="flex-1 overflow-y-auto min-h-0 pr-2 custom-scrollbar">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="sticky top-0 bg-slate-900 z-10 text-xs text-slate-500 uppercase tracking-wider">
                  <th className="pb-3 font-medium">Sample ID</th>
                  <th className="pb-3 font-medium">Image</th>
                  <th className="pb-3 font-medium">IoU Score</th>
                  <th className="pb-3 font-medium">Primary Reason</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {currentSamples.map((sample) => (
                  <tr key={sample.id} className="hover:bg-slate-800/50 transition-colors">
                    <td className="py-3 text-sm font-mono text-slate-400">{sample.id}</td>
                    <td className="py-3 text-sm text-slate-300 truncate max-w-[120px]">{sample.image}</td>
                    <td className="py-3">
                      <span className="text-xs font-bold text-red-400 bg-red-500/10 px-2 py-1 rounded border border-red-500/20">
                        {sample.iou.toFixed(2)}
                      </span>
                    </td>
                    <td className="py-3 text-sm text-slate-300">
                       <span className="flex items-center gap-1.5">
                          <i className="fa-solid fa-circle text-[8px]" style={{ color: failure_distribution.find(d => d.name === sample.reason)?.color || '#94a3b8' }}></i>
                          {sample.reason}
                       </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div className="flex justify-between items-center mt-4 pt-3 border-t border-slate-800">
              <button 
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="text-xs text-slate-400 hover:text-white disabled:opacity-30 disabled:hover:text-slate-400 transition-colors flex items-center gap-1"
              >
                <i className="fa-solid fa-chevron-left"></i> Prev
              </button>
              <span className="text-xs text-slate-500 font-mono">
                Page {currentPage} of {totalPages}
              </span>
              <button 
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="text-xs text-slate-400 hover:text-white disabled:opacity-30 disabled:hover:text-slate-400 transition-colors flex items-center gap-1"
              >
                Next <i className="fa-solid fa-chevron-right"></i>
              </button>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default AnalysisDashboard;
