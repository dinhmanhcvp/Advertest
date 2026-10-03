import React from 'react';

const AttackRecipeModal = ({ isOpen, onClose, attackData }) => {
  if (!isOpen || !attackData) return null;

  const { audit_trail, image_url } = attackData;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm">
      <div className="bg-slate-900 border border-slate-700/50 rounded-2xl p-6 w-full max-w-5xl shadow-[0_0_50px_rgba(0,0,0,0.5)] relative overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="flex justify-between items-center mb-6 border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <i className="fa-solid fa-flask text-2xl text-blue-400"></i>
            <div>
              <h2 className="text-xl font-bold text-white">Simulation Recipe (Audit Trail)</h2>
              <p className="text-xs text-slate-400">ID: {attackData.id} • Traceability & Explainability</p>
            </div>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-white transition">
            <i className="fa-solid fa-xmark text-xl"></i>
          </button>
        </div>

        {/* 3-Column Layout */}
        <div className="flex flex-col lg:flex-row gap-6 flex-1 overflow-y-auto min-h-0">
          
          {/* Left: Original + Tag */}
          <div className="flex-1 flex flex-col">
            <h3 className="text-sm font-bold text-slate-300 mb-3 flex justify-between items-center">
              <span>Original Input</span>
              <span className="bg-slate-800 text-xs px-2 py-1 rounded text-slate-400 border border-slate-700">Before</span>
            </h3>
            <div className="bg-slate-950 rounded-xl flex-1 flex items-center justify-center border border-slate-800 relative overflow-hidden p-2 min-h-[200px]">
              <div className="absolute inset-0 opacity-20 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] pointer-events-none"></div>
              {/* Placeholder for original image, for demo we just show a gray box or the same image with lower contrast */}
              <div className="w-full h-full bg-slate-800/50 rounded-lg flex items-center justify-center border border-dashed border-slate-700">
                <i className="fa-regular fa-image text-4xl text-slate-600 mb-2"></i>
              </div>
            </div>
            <div className="mt-4 p-3 bg-red-950/30 border border-red-500/20 rounded-lg flex items-start gap-3">
              <i className="fa-solid fa-tag text-red-400 mt-1"></i>
              <div>
                <span className="text-xs text-slate-400 block">Insight Source</span>
                <span className="text-sm font-mono font-bold text-red-400">{audit_trail.insight_source}</span>
              </div>
            </div>
          </div>

          {/* Middle: Node-chain + Justification */}
          <div className="flex-1 flex flex-col justify-center border-x border-slate-800/50 px-6">
            <h3 className="text-sm font-bold text-blue-400 mb-4 text-center">Augmentation Chain</h3>
            
            <div className="flex flex-col items-center gap-2 mb-6">
              {audit_trail.applied_chain.map((step, idx) => (
                <React.Fragment key={idx}>
                  <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg py-2 px-4 w-full text-center shadow-[0_0_15px_rgba(59,130,246,0.1)]">
                    <span className="text-sm font-mono text-blue-300">{step}</span>
                  </div>
                  {idx < audit_trail.applied_chain.length - 1 && (
                    <i className="fa-solid fa-arrow-down text-slate-600 text-xs"></i>
                  )}
                </React.Fragment>
              ))}
              <i className="fa-solid fa-arrow-down text-slate-600 text-xs"></i>
              <div className="flex items-center gap-2 bg-orange-950/30 border border-orange-500/30 rounded-full px-3 py-1">
                <i className="fa-solid fa-fire text-orange-400 text-xs"></i>
                <span className="text-xs text-orange-300">Severity: {audit_trail.severity_level}/5</span>
              </div>
            </div>

            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700 relative">
              <i className="fa-solid fa-quote-left absolute top-2 left-2 text-slate-600/30 text-2xl"></i>
              <h4 className="text-xs font-bold text-slate-300 mb-2 relative z-10">The &quot;Why&quot; (Justification)</h4>
              <p className="text-sm text-slate-400 leading-relaxed relative z-10 italic">
                &quot;{audit_trail.justification}&quot;
              </p>
            </div>
          </div>

          {/* Right: Final Synthesized */}
          <div className="flex-1 flex flex-col">
            <h3 className="text-sm font-bold text-slate-300 mb-3 flex justify-between items-center">
              <span>Synthesized Attack</span>
              <span className="bg-emerald-500/20 text-xs px-2 py-1 rounded text-emerald-400 border border-emerald-500/30">After</span>
            </h3>
            <div className="bg-slate-950 rounded-xl flex-1 flex items-center justify-center border border-emerald-500/20 relative overflow-hidden p-2 min-h-[200px] shadow-[0_0_30px_rgba(16,185,129,0.05)]">
               {/* In a real scenario, use image_url */}
               <div className="w-full h-full bg-slate-800 rounded-lg flex items-center justify-center overflow-hidden">
                  <div className="w-full h-full bg-gradient-to-br from-slate-700 to-slate-900 flex flex-col items-center justify-center text-slate-500">
                     <i className="fa-solid fa-wand-magic-sparkles text-3xl mb-2 text-emerald-500/50"></i>
                     <span className="text-xs font-mono">Synthesized Artifact</span>
                  </div>
               </div>
            </div>
            <div className="mt-4 p-3 bg-emerald-950/20 border border-emerald-500/20 rounded-lg flex justify-between items-center">
              <span className="text-xs text-slate-400">Status</span>
              <span className="text-xs font-bold text-emerald-400 flex items-center gap-1">
                <i className="fa-solid fa-check-circle"></i> Passed QA Gate
              </span>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default AttackRecipeModal;
