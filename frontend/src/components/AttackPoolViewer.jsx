"use client";
import React, { useState, useEffect } from 'react';
import AttackRecipeModal from './AttackRecipeModal';
import { motion } from 'framer-motion';

const AttackPoolViewer = () => {
  const [attacks, setAttacks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedAttack, setSelectedAttack] = useState(null);

  useEffect(() => {
    fetch('/api/v1/demo/audit-trail')
      .then(res => {
        if (!res.ok) throw new Error('API Response was not OK');
        return res.json();
      })
      .then(data => {
        setAttacks(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load attack pool data", err);
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="mt-6"
      >
        <div className="flex items-center gap-3 mb-6">
          <i className="fa-solid fa-viruses text-xl text-purple-400/50"></i>
          <div>
            <div className="h-5 w-48 bg-slate-800 rounded animate-pulse mb-2"></div>
            <div className="h-3 w-64 bg-slate-800/50 rounded animate-pulse"></div>
          </div>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg h-64 flex flex-col">
               <div className="h-40 bg-slate-800/50 animate-pulse"></div>
               <div className="p-4 flex-1 flex flex-col justify-between">
                 <div>
                   <div className="h-3 w-16 bg-slate-800 rounded animate-pulse mb-3"></div>
                   <div className="h-4 w-3/4 bg-slate-800 rounded animate-pulse mb-2"></div>
                   <div className="flex gap-2 mt-3">
                     <div className="h-3 w-12 bg-slate-800 rounded animate-pulse"></div>
                     <div className="h-3 w-12 bg-slate-800 rounded animate-pulse"></div>
                   </div>
                 </div>
               </div>
            </div>
          ))}
        </div>
      </motion.div>
    );
  }

  if (error) {
    return (
      <motion.div 
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className="mt-6 bg-red-500/10 border border-red-500/30 p-4 rounded-xl flex items-center gap-3"
      >
        <i className="fa-solid fa-triangle-exclamation text-red-400 text-xl"></i>
        <div>
          <h4 className="text-red-400 font-bold">Failed to load Attack Pool</h4>
          <p className="text-red-400/80 text-xs">Ensure the backend API is running. Error: {error}</p>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, staggerChildren: 0.1 }}
      className="mt-6"
    >
      <div className="flex items-center gap-3 mb-6">
        <i className="fa-solid fa-viruses text-xl text-purple-400"></i>
        <div>
          <h3 className="text-white font-bold text-lg">Generated Attack Pool</h3>
          <p className="text-xs text-slate-400">Click any synthesized image to view its generation recipe (Explainability).</p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {attacks.map((attack) => (
          <div 
            key={attack.id}
            onClick={() => setSelectedAttack(attack)}
            className="group cursor-pointer bg-slate-900 border border-slate-700/50 rounded-xl overflow-hidden hover:border-purple-500/50 transition-all duration-300 shadow-lg hover:shadow-[0_0_20px_rgba(168,85,247,0.2)] flex flex-col"
          >
            <div className="h-40 bg-slate-950 relative flex items-center justify-center overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-t from-slate-900 to-transparent opacity-60 z-10"></div>
                {/* Fallback visual if image_url fails to load, representing the synthetic data */}
                <i className="fa-solid fa-image text-4xl text-slate-700 group-hover:scale-110 transition-transform duration-500"></i>
                <div className="absolute top-2 right-2 z-20 bg-purple-500/20 text-purple-400 text-[10px] font-bold px-2 py-1 rounded border border-purple-500/30">
                  Synthesized
                </div>
            </div>
            
            <div className="p-4 flex-1 flex flex-col justify-between bg-slate-900 z-20">
               <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-[10px] font-mono text-slate-500">{attack.id}</span>
                  </div>
                  <h4 className="text-sm font-bold text-slate-200 mb-1 truncate">
                    Target: {attack.audit_trail.insight_source}
                  </h4>
                  <div className="flex flex-wrap gap-1 mt-2">
                     {attack.audit_trail.applied_chain.slice(0, 2).map((chainItem, idx) => (
                       <span key={idx} className="text-[9px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded border border-slate-700">
                         {chainItem}
                       </span>
                     ))}
                     {attack.audit_trail.applied_chain.length > 2 && (
                       <span className="text-[9px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded border border-slate-700">
                         +{attack.audit_trail.applied_chain.length - 2} more
                       </span>
                     )}
                  </div>
               </div>
               
               <div className="mt-4 flex justify-between items-center border-t border-slate-800 pt-3">
                 <span className="text-[10px] text-orange-400 flex items-center gap-1">
                   <i className="fa-solid fa-fire"></i> Severity: {attack.audit_trail.severity_level}/5
                 </span>
                 <span className="text-[10px] text-blue-400 font-bold group-hover:text-blue-300 transition-colors flex items-center gap-1">
                   View Recipe <i className="fa-solid fa-arrow-right"></i>
                 </span>
               </div>
            </div>
          </div>
        ))}
      </div>

      <AttackRecipeModal 
        isOpen={!!selectedAttack} 
        onClose={() => setSelectedAttack(null)} 
        attackData={selectedAttack} 
      />
    </motion.div>
  );
};

export default AttackPoolViewer;
