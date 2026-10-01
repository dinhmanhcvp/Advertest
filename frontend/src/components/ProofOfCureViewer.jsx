import React, { useState, useEffect } from 'react';

const ProofOfCureViewer = () => {
  const [evalData, setEvalData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch from the new demo endpoint
    fetch('/api/v1/demo/proof-of-cure')
      .then(res => res.json())
      .then(data => {
        setEvalData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load proof of cure data", err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div className="text-slate-400 text-sm animate-pulse">Loading Validation Proofs...</div>;
  }

  return (
    <div className="flex flex-col gap-8 mt-6">
      {evalData.map((item) => (
        <div key={item.case_id} className="bg-slate-900 border border-slate-700/50 rounded-2xl p-6 relative overflow-hidden shadow-lg">
          {/* Header */}
          <div className="flex justify-between items-center mb-6">
            <div className="flex items-center gap-3">
              <i className="fa-solid fa-microscope text-xl text-cyan-400"></i>
              <div>
                <h3 className="text-white font-bold text-lg">{item.condition}</h3>
                <p className="text-xs text-slate-400">Case ID: {item.case_id} • Edge Case Validation</p>
              </div>
            </div>
            {/* Delta Badge */}
            <div className="bg-gradient-to-r from-emerald-500/20 to-teal-500/20 border border-emerald-500/40 px-4 py-2 rounded-full shadow-[0_0_20px_rgba(16,185,129,0.2)]">
              <span className="text-emerald-400 font-black flex items-center gap-2 text-sm tracking-wide">
                <i className="fa-solid fa-arrow-trend-up"></i>
                {item.delta}
              </span>
            </div>
          </div>

          {/* Side-by-Side Comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* Model V1 (Left) */}
            <div className="flex flex-col">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-bold text-slate-300">Model V1 (Baseline)</span>
                <span className="text-xs bg-red-500/20 text-red-400 px-2 py-1 rounded font-bold border border-red-500/30">
                  {item.model_v1.status}
                </span>
              </div>
              <div className="bg-slate-950 rounded-xl relative overflow-hidden border border-slate-800 h-64 flex items-center justify-center">
                 {/* Visual Representation of Image + Red BBox */}
                 <div className="absolute inset-0 bg-gradient-to-br from-slate-800 to-slate-900 opacity-50"></div>
                 <i className="fa-regular fa-image text-4xl text-slate-700 mb-2 absolute"></i>
                 
                 {item.model_v1.bbox ? (
                   <div 
                     className="absolute border-2 border-red-500 bg-red-500/10 shadow-[0_0_15px_rgba(239,68,68,0.5)] flex items-end justify-center pb-1"
                     style={{
                       left: '30%', top: '30%', width: '40%', height: '40%' // Mock coordinates
                     }}
                   >
                     <span className="bg-red-500 text-white text-[10px] font-bold px-1 absolute -top-4 left-0">
                        Conf: {item.model_v1.confidence.toFixed(2)}
                     </span>
                   </div>
                 ) : (
                   <div className="absolute flex flex-col items-center">
                     <i className="fa-solid fa-ghost text-red-500/50 text-3xl mb-1"></i>
                     <span className="text-red-400/80 text-xs font-bold">Missed Detection</span>
                   </div>
                 )}
              </div>
            </div>

            {/* Model V2 (Right) */}
            <div className="flex flex-col">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-bold text-slate-300">Model V2 (AdverTest Retrained)</span>
                <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-1 rounded font-bold border border-emerald-500/30">
                  {item.model_v2.status}
                </span>
              </div>
              <div className="bg-slate-950 rounded-xl relative overflow-hidden border border-emerald-500/30 h-64 flex items-center justify-center shadow-[0_0_30px_rgba(16,185,129,0.1)]">
                 <div className="absolute inset-0 bg-gradient-to-br from-slate-800 to-slate-900 opacity-50"></div>
                 <i className="fa-regular fa-image text-4xl text-slate-700 mb-2 absolute"></i>
                 
                 {/* Green BBox */}
                 <div 
                   className="absolute border-2 border-emerald-500 bg-emerald-500/10 shadow-[0_0_15px_rgba(16,185,129,0.5)] flex items-end justify-center pb-1 transition-all duration-1000 ease-in-out"
                   style={{
                     left: '28%', top: '28%', width: '44%', height: '44%' // Slightly better bounding box representation
                   }}
                 >
                   <span className="bg-emerald-500 text-white text-[10px] font-bold px-1 absolute -top-4 left-0 flex items-center gap-1">
                      <i className="fa-solid fa-check"></i> Conf: {item.model_v2.confidence.toFixed(2)}
                   </span>
                 </div>
              </div>
            </div>

          </div>
        </div>
      ))}
    </div>
  );
};

export default ProofOfCureViewer;
