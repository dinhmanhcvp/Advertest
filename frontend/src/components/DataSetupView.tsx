"use client";
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'sonner';

const DataSetupView = ({ onAnalyze }) => {
  const [modelFile, setModelFile] = useState(null);
  const [datasetPath, setDatasetPath] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!datasetPath) {
      toast.warning('Using default Ego4D_Validation dataset.');
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/triage/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: 'yolov7-tiny-face.pt', dataset_path: datasetPath || 'Ego4D_Validation' })
      });
      if (!response.ok) throw new Error('Failed to fetch from API');
      const data = await response.json();
      toast.success('Baseline inference completed successfully!');
      onAnalyze(data);
    } catch (err) {
      console.error(err);
      setError(err.message);
      toast.error('Failed to run baseline inference');
    }
    setLoading(false);
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="bg-slate-900 border border-slate-700/50 rounded-2xl p-8 shadow-[0_0_40px_rgba(0,0,0,0.5)] max-w-2xl mx-auto mt-10"
    >
      <div className="mb-8 border-b border-slate-800 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center gap-3">
          <i className="fa-solid fa-server text-blue-400"></i>
          Resource Setup
        </h2>
        <p className="text-sm text-slate-400 mt-2">Configure your model weights and target validation dataset for baseline inference.</p>
      </div>

      <div className="flex flex-col gap-6">
        {/* Model Selection */}
        <div>
          <label className="block text-sm font-bold text-slate-300 mb-2">Pre-trained Model (.pt)</label>
          <div className="border-2 border-dashed border-slate-700 bg-slate-800/50 rounded-xl p-6 text-center hover:border-blue-500/50 transition-colors cursor-pointer relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-t from-slate-900 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <i className="fa-solid fa-weight-hanging text-3xl text-slate-500 mb-2 relative z-10"></i>
            <p className="text-sm text-slate-400 relative z-10">
              Drag & Drop your YOLO weights here, or <span className="text-blue-400">browse</span>
            </p>
            <p className="text-xs text-slate-500 mt-1 relative z-10">yolov7-tiny-face.pt</p>
          </div>
        </div>

        {/* Dataset Selection */}
        <div>
          <label className="block text-sm font-bold text-slate-300 mb-2">Validation Dataset</label>
          <div className="flex bg-slate-950 border border-slate-700 rounded-lg overflow-hidden focus-within:border-blue-500/50 transition-colors">
            <div className="bg-slate-800 px-4 flex items-center border-r border-slate-700">
              <i className="fa-solid fa-folder-open text-slate-400"></i>
            </div>
            <input 
              type="text" 
              placeholder="e.g. data/ego4d_sample/" 
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              className="bg-transparent border-none outline-none text-slate-300 text-sm p-3 w-full"
            />
          </div>
        </div>

        {/* Analyze Button */}
        <button 
          onClick={handleAnalyze} 
          disabled={loading}
          className="mt-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold py-3 px-6 rounded-lg shadow-[0_0_20px_rgba(59,130,246,0.3)] transition-all flex items-center justify-center gap-2 disabled:opacity-50"
        >
          {loading ? (
            <><i className="fa-solid fa-circle-notch fa-spin"></i> Analyzing...</>
          ) : (
            <><i className="fa-solid fa-magnifying-glass-chart"></i> Run Baseline Inference</>
          )}
        </button>

        {error && (
          <div className="mt-2 bg-red-500/10 border border-red-500/30 p-4 rounded-xl flex items-center gap-3">
            <i className="fa-solid fa-triangle-exclamation text-red-400 text-xl"></i>
            <div>
              <h4 className="text-red-400 font-bold">Failed to Fetch Action</h4>
              <p className="text-red-400/80 text-xs">Error: {error}. Is the backend running?</p>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default DataSetupView;
