"use client";
import React, { useState } from 'react';
import DataSetupView from '@/components/DataSetupView';
import AnalysisDashboard from '@/components/AnalysisDashboard';
import AttackPoolViewer from '@/components/AttackPoolViewer';
import ProofOfCureDashboard from '@/components/ProofOfCureDashboard';
import { motion } from 'framer-motion';
import { toast } from 'sonner';

export default function DemoWorkflowPage() {
  const [analysisData, setAnalysisData] = useState(null);
  const [showAttackPool, setShowAttackPool] = useState(false);
  const [retrainData, setRetrainData] = useState(null);
  const [isRetraining, setIsRetraining] = useState(false);

  const handleAnalyze = (data) => {
    setAnalysisData(data);
  };

  const handleIsolate = (data) => {
    // The data here is the audit_trail generated, but AttackPoolViewer fetches it itself
    setShowAttackPool(true);
  };

  const handleRetrain = async () => {
    setIsRetraining(true);
    const toastId = toast.loading('Initializing Ray Cluster & Retraining Model...');
    try {
      const response = await fetch('/api/v1/workflow/retrain', {
        method: 'POST',
      });
      if (!response.ok) throw new Error('API Response was not OK');
      const data = await response.json();
      setRetrainData(data);
      toast.success('Retraining complete! Evaluation ready.', { id: toastId });
    } catch (err) {
      console.error(err);
      toast.error('Retraining failed: ' + err.message, { id: toastId });
    }
    setIsRetraining(false);
  };

  return (
    <div className="min-h-screen bg-slate-950 p-6 md:p-12">
      <div className="max-w-6xl mx-auto space-y-12 pb-24">
        
        {/* Step 1: Setup */}
        <section>
          <div className="mb-4">
            <h1 className="text-3xl font-black text-white">End-to-End Demo Workflow</h1>
            <p className="text-slate-400">Step 1: Configure resources and run baseline inference.</p>
          </div>
          <DataSetupView onAnalyze={handleAnalyze} />
        </section>

        {/* Step 2: Triage */}
        {analysisData && (
          <motion.section 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="mb-4">
              <h2 className="text-2xl font-black text-white">Step 2: Visual Triage</h2>
              <p className="text-slate-400">Analyzing the failure distribution across the dataset.</p>
            </div>
            <AnalysisDashboard analysisData={analysisData} onIsolate={handleIsolate} />
          </motion.section>
        )}

        {/* Step 3: Attack Generation */}
        {showAttackPool && (
          <motion.section 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="mb-4 flex justify-between items-end border-b border-slate-800 pb-4">
              <div>
                <h2 className="text-2xl font-black text-white">Step 3: Attack Generation (Curriculum)</h2>
                <p className="text-slate-400">Synthesized edge cases based on triage insights.</p>
              </div>
              
              {!retrainData && (
                <button 
                  onClick={handleRetrain}
                  disabled={isRetraining}
                  className="bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-2 px-6 rounded-lg shadow-[0_0_15px_rgba(16,185,129,0.3)] transition-all flex items-center gap-2 disabled:opacity-50"
                >
                  {isRetraining ? (
                    <><i className="fa-solid fa-circle-notch fa-spin"></i> Training...</>
                  ) : (
                    <><i className="fa-solid fa-bolt"></i> Trigger Ray Retraining Loop</>
                  )}
                </button>
              )}
            </div>
            <AttackPoolViewer />
          </motion.section>
        )}

        {/* Step 4: Proof of Cure */}
        {retrainData && (
          <motion.section 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="mb-4">
              <h2 className="text-2xl font-black text-white">Step 4: Proof of Cure</h2>
              <p className="text-slate-400">Validating the newly trained model against edge cases.</p>
            </div>
            <ProofOfCureDashboard retrainData={retrainData} />
          </motion.section>
        )}

      </div>
    </div>
  );
}
