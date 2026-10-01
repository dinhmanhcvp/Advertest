"use client";
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'sonner';
import { Database, UploadCloud, Search, ShieldAlert, Cpu } from 'lucide-react';

const DataSetupView = ({ onAnalyze }) => {
  const [modelFile, setModelFile] = useState(null);
  const [datasetPath, setDatasetPath] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!datasetPath) {
      toast.warning('Using default Ego4D_Validation dataset.', { style: { background: '#000', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.2)' } });
    }
    setLoading(true);
    setError(null);
    const toastId = toast.loading('Initializing validation protocol...', { style: { background: '#000', color: '#fff', border: '1px solid rgba(255,255,255,0.1)' } });
    try {
      const response = await fetch('/api/v1/triage/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: 'yolov7-tiny-face.pt', dataset_path: datasetPath || 'Ego4D_Validation' })
      });
      if (!response.ok) throw new Error('Failed to fetch from API');
      const data = await response.json();
      toast.success('Baseline inference completed successfully!', { id: toastId });
      onAnalyze(data);
    } catch (err) {
      console.error(err);
      setError(err.message);
      toast.error('Failed to run baseline inference', { id: toastId });
    }
    setLoading(false);
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      className="bg-black border-[1px] border-white/10 rounded-2xl p-8 shadow-[0_0_50px_rgba(0,0,0,0.8)] max-w-2xl mx-auto mt-10 relative overflow-hidden"
    >
      {/* Background Subtle Glow */}
      <div className="absolute -top-32 -right-32 w-64 h-64 bg-cyan-500/10 blur-[100px] rounded-full pointer-events-none"></div>

      <div className="mb-8 border-b border-white/10 pb-5">
        <h2 className="text-2xl font-medium tracking-tight text-white flex items-center gap-3">
          <Database className="w-6 h-6 text-cyan-400" />
          Resource Initialization
        </h2>
        <p className="text-sm text-zinc-400 mt-2 font-mono">
          &gt; CONFIGURE TARGET [MODEL_WEIGHTS] AND [VALIDATION_SET]
        </p>
      </div>

      <div className="flex flex-col gap-8">
        {/* Model Selection */}
        <div>
          <label className="block text-xs font-mono tracking-widest text-zinc-500 mb-3 uppercase">Target Checkpoint (.pt)</label>
          <div className="border border-dashed border-white/20 bg-white/[0.02] backdrop-blur-md rounded-xl p-8 text-center hover:border-cyan-500/50 hover:bg-cyan-500/5 transition-all cursor-pointer relative group">
            <div className="absolute inset-0 bg-gradient-to-t from-cyan-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity rounded-xl"></div>
            <Cpu className="w-8 h-8 text-zinc-600 group-hover:text-cyan-400 mx-auto mb-3 transition-colors" />
            <p className="text-sm text-zinc-400 relative z-10">
              Drag & Drop weights, or <span className="text-cyan-400 group-hover:underline">browse registry</span>
            </p>
            <p className="text-[11px] font-mono text-zinc-500 mt-2 relative z-10 bg-black px-2 py-1 inline-block rounded border border-white/10">
              yolov7-tiny-face.pt
            </p>
          </div>
        </div>

        {/* Dataset Selection */}
        <div>
          <label className="block text-xs font-mono tracking-widest text-zinc-500 mb-3 uppercase">Validation Dataset</label>
          <div className="flex bg-black border border-white/10 rounded-xl overflow-hidden focus-within:border-cyan-500/50 focus-within:shadow-[0_0_15px_rgba(34,211,238,0.15)] transition-all">
            <div className="bg-white/[0.02] px-4 flex items-center border-r border-white/10">
              <UploadCloud className="w-4 h-4 text-zinc-500" />
            </div>
            <input 
              type="text" 
              placeholder="e.g. data/ego4d_sample/" 
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              className="bg-transparent border-none outline-none text-zinc-300 text-sm p-3 w-full font-mono placeholder:text-zinc-700"
            />
          </div>
        </div>

        {/* Analyze Button */}
        <button 
          onClick={handleAnalyze} 
          disabled={loading}
          className="mt-2 group relative w-full flex items-center justify-center gap-2 bg-white text-black font-medium rounded-xl py-3.5 hover:bg-zinc-200 transition-colors disabled:opacity-50"
        >
          <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-violet-600 via-fuchsia-500 to-cyan-400 opacity-0 group-hover:opacity-100 blur transition-opacity duration-300 -z-10"></div>
          {loading ? (
            <><i className="fa-solid fa-circle-notch fa-spin"></i> EXECUTING...</>
          ) : (
            <><Search className="w-4 h-4" /> Run Baseline Inference</>
          )}
        </button>

        {error && (
          <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-xl flex items-start gap-3 font-mono text-sm shadow-[0_0_15px_rgba(239,68,68,0.15)]">
            <ShieldAlert className="w-5 h-5 text-red-400 shrink-0" />
            <div>
              <h4 className="text-red-400 font-bold">SYS_ERROR</h4>
              <p className="text-red-400/80 text-[11px] mt-1 leading-relaxed">Failed to initialize: {error}. Check backend connection.</p>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default DataSetupView;
