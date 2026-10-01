"use client";
import React, { useState, useEffect } from 'react';

const ProofOfCureDashboard = ({ retrainData }) => {
  if (!retrainData) return null;

  const { map_improvement, proof_of_cure } = retrainData;

  return (
    <div className="flex flex-col gap-6 mt-6 max-w-5xl mx-auto">
      {/* Header & Overall Improvement */}
      <div className="bg-slate-900 border border-slate-700/50 rounded-2xl p-6 shadow-lg">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-3">
              <i className="fa-solid fa-shield-check text-emerald-400"></i>
              Proof of Cure Validation
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Retraining complete. Evaluating against held-out validation set.
            </p>
          </div>
          <div className="bg-emerald-500/10 border border-emerald-500/30 px-4 py-2 rounded-lg flex items-center gap-3">
             <div className="flex flex-col">
               <span className="text-[10px] text-emerald-400/80 uppercase font-bold tracking-wider">mPC Improvement</span>
               <span className="text-xl font-black text-emerald-400">+{map_improvement.corrupted_improvement_pct.toFixed(1)}%</span>
             </div>
             <div className="h-8 w-px bg-emerald-500/20 mx-2"></div>
             <div className="flex flex-col">
               <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Base mAP Drop</span>
               <span className="text-sm font-bold text-slate-300">-{map_improvement.base_drop_pct.toFixed(1)}%</span>
             </div>
          </div>
        </div>

        {/* Before / After Cards */}
        <h3 className="text-sm font-bold text-slate-300 mb-4 border-b border-slate-800 pb-2">Edge Case Validation Samples</h3>
        
        <div className="flex flex-col gap-6">
          {proof_of_cure.map((item) => (
            <div key={item.case_id} className="bg-slate-950 border border-slate-800 rounded-xl p-5 relative overflow-hidden shadow-inner">
              <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-2">
                  <span className="text-xs bg-slate-800 text-slate-400 px-2 py-1 rounded font-mono border border-slate-700">{item.case_id}</span>
                  <h4 className="text-sm font-bold text-slate-200">{item.condition}</h4>
                </div>
                <div className="bg-gradient-to-r from-emerald-500/20 to-teal-500/20 border border-emerald-500/40 px-3 py-1 rounded-full">
                  <span className="text-emerald-400 font-bold flex items-center gap-1.5 text-xs">
                    <i className="fa-solid fa-arrow-trend-up"></i>
                    {item.delta}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Model V1 */}
                <div className="bg-slate-900 border border-red-500/20 rounded-lg p-3 relative overflow-hidden group">
                  <div className="flex justify-between items-center mb-2 z-10 relative">
                    <span className="text-xs font-bold text-slate-400">Model V1 (Baseline)</span>
                    <span className="text-[10px] bg-red-500/20 text-red-400 px-1.5 py-0.5 rounded font-bold uppercase tracking-wider">
                      {item.model_v1.status}
                    </span>
                  </div>
                  <div className="h-48 bg-slate-800 rounded-md relative flex items-center justify-center border border-slate-700/50">
                     <i className="fa-regular fa-image text-3xl text-slate-600 absolute"></i>
                     {item.model_v1.bbox ? (
                       <div 
                         className="absolute border-2 border-red-500 bg-red-500/10 flex items-end pb-1"
                         style={{ left: '30%', top: '30%', width: '40%', height: '40%' }} // Mock coords
                       >
                         <span className="bg-red-500 text-white text-[9px] font-bold px-1 absolute -top-4 left-0">
                            {item.model_v1.confidence.toFixed(2)}
                         </span>
                       </div>
                     ) : (
                       <div className="absolute flex flex-col items-center">
                         <i className="fa-solid fa-ghost text-red-500/40 text-2xl mb-1"></i>
                         <span className="text-red-400/60 text-[10px] font-bold">Missed</span>
                       </div>
                     )}
                  </div>
                </div>

                {/* Model V2 */}
                <div className="bg-slate-900 border border-emerald-500/30 rounded-lg p-3 relative overflow-hidden">
                  <div className="flex justify-between items-center mb-2 z-10 relative">
                    <span className="text-xs font-bold text-emerald-400/80">Model V2 (Retrained)</span>
                    <span className="text-[10px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded font-bold uppercase tracking-wider shadow-[0_0_10px_rgba(16,185,129,0.2)]">
                      {item.model_v2.status}
                    </span>
                  </div>
                  <div className="h-48 bg-slate-800 rounded-md relative flex items-center justify-center border border-slate-700/50">
                     <i className="fa-regular fa-image text-3xl text-slate-600 absolute"></i>
                     <div 
                       className="absolute border-2 border-emerald-500 bg-emerald-500/10 flex items-end pb-1 shadow-[0_0_15px_rgba(16,185,129,0.2)]"
                       style={{ left: '28%', top: '28%', width: '44%', height: '44%' }}
                     >
                       <span className="bg-emerald-500 text-white text-[9px] font-bold px-1 absolute -top-4 left-0 flex items-center gap-1">
                          <i className="fa-solid fa-check text-[8px]"></i> {item.model_v2.confidence.toFixed(2)}
                       </span>
                     </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ProofOfCureDashboard;
