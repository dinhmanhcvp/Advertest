"use client";

import { motion } from "framer-motion";
import {
  AlertTriangle,
  CloudUpload,
  Workflow,
  Plus,
  Upload,
  Box,
  Layers,
  Activity,
  ArrowRight,
  ShieldCheck,
  Zap,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import QuickUploadDialog from "@/components/dashboard/QuickUploadDialog";
import { useProject } from "@/context/ProjectContext";
import { getCatalogAttacks, getCatalogDatasets, getModelVersions, listRuns } from "@/lib/api";

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: { type: "spring", stiffness: 300, damping: 24 },
  },
};

export default function DashboardView() {
  const { activeProject, activeProjectId } = useProject();
  const [runs, setRuns] = useState([]);
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [attacks, setAttacks] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [uploadKind, setUploadKind] = useState(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const loadAssets = useCallback(async () => {
    const results = await Promise.allSettled([
      listRuns(activeProjectId || undefined),
      getModelVersions(),
      getCatalogDatasets(),
      getCatalogAttacks(),
    ]);
    if (!mountedRef.current) return;
    const [runsRes, modelsRes, datasetsRes, attacksRes] = results;
    if (runsRes.status === "fulfilled" && Array.isArray(runsRes.value)) setRuns(runsRes.value);
    if (modelsRes.status === "fulfilled" && Array.isArray(modelsRes.value)) setModels(modelsRes.value);
    if (datasetsRes.status === "fulfilled" && Array.isArray(datasetsRes.value)) setDatasets(datasetsRes.value);
    if (attacksRes.status === "fulfilled" && Array.isArray(attacksRes.value)) setAttacks(attacksRes.value);
    setIsLoading(false);
  }, [activeProjectId]);

  useEffect(() => {
    loadAssets();
  }, [loadAssets]);

  const highRiskRun = runs.some(
    (run) => run.report && (run.report.asr || run.report.metrics?.robustness?.attack_success_rate || 0) > 0.4,
  );
  
  const scopedHref = (href) =>
    activeProjectId
      ? `${href}${href.includes("?") ? "&" : "?"}project_id=${encodeURIComponent(activeProjectId)}`
      : href;

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="relative space-y-8 w-full max-w-7xl mx-auto p-4 sm:p-6"
    >
      {/* Background Orbs for Aesthetic */}
      <div className="glow-orb-primary -top-20 -left-20"></div>
      <div className="glow-orb-secondary top-40 -right-20"></div>

      {/* Hero Section */}
      <motion.div variants={itemVariants} className="relative glass-panel rounded-2xl p-8 overflow-hidden group">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 via-transparent to-teal-500/10 opacity-50 group-hover:opacity-100 transition-opacity duration-700"></div>
        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-8">
          <div className="space-y-4 max-w-3xl">
            <h1 className="text-3xl font-bold font-sans tracking-tight">
              Welcome to <span className="text-gradient bg-gradient-to-r from-purple-400 to-teal-400">AdverTest</span>
            </h1>
            <p className="text-zinc-400 leading-relaxed text-sm md:text-base">
              The ultimate adversarial testing platform. Evaluate model robustness against real-world physical anomalies, 
              synthesize edge-case datasets, and fortify your perception systems through an end-to-end closed loop.
            </p>
            
            <div className="flex flex-wrap items-center gap-3 pt-2">
              <span className="text-xs font-mono text-zinc-500 uppercase tracking-widest">Supported Tasks</span>
              <span className="px-3 py-1 rounded-full bg-teal-500/10 border border-teal-500/20 text-teal-400 text-xs font-semibold flex items-center gap-1.5 shadow-[0_0_10px_rgba(45,212,191,0.1)]">
                <Box className="w-3.5 h-3.5" /> Object Detection (2D/3D)
              </span>
              <span className="px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-400 text-xs font-semibold flex items-center gap-1.5 shadow-[0_0_10px_rgba(168,85,247,0.1)]">
                <Layers className="w-3.5 h-3.5" /> Segmentation
              </span>
            </div>
          </div>

          <div className="shrink-0 flex flex-col gap-3">
            <Link
              href={scopedHref("/experiments/new")}
              className="group relative inline-flex items-center justify-center gap-3 rounded-xl bg-white px-8 py-4 text-sm font-bold text-black shadow-[0_0_20px_rgba(255,255,255,0.3)] transition-all hover:scale-105 hover:shadow-[0_0_30px_rgba(255,255,255,0.5)] overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-white via-zinc-200 to-white opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <Zap className="relative h-5 w-5 fill-black" />
              <span className="relative">Start New Benchmark</span>
            </Link>
          </div>
        </div>
      </motion.div>

      {/* Grid Layout for Assets & Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Quick Actions */}
        <motion.div variants={itemVariants} className="lg:col-span-1 space-y-6">
          <div className="glass-panel rounded-2xl p-6 h-full flex flex-col">
            <h3 className="text-sm font-semibold text-zinc-300 flex items-center gap-2 mb-6 uppercase tracking-wider font-mono">
              <Activity className="w-4 h-4 text-purple-400" /> Quick Actions
            </h3>
            
            <div className="grid grid-cols-1 gap-3 flex-1">
              <button
                onClick={() => setUploadKind("model")}
                className="group flex items-center justify-between p-4 rounded-xl border border-zinc-800 bg-zinc-900/50 hover:bg-zinc-800 hover:border-zinc-700 transition-all"
              >
                <div className="flex items-center gap-3 text-sm font-medium text-zinc-200">
                  <div className="p-2 rounded-lg bg-zinc-800 group-hover:bg-purple-500/20 group-hover:text-purple-400 transition-colors">
                    <Upload className="w-4 h-4" />
                  </div>
                  Upload Checkpoint
                </div>
                <Plus className="w-4 h-4 text-zinc-600 group-hover:text-zinc-300 transition-colors" />
              </button>

              <button
                onClick={() => setUploadKind("dataset")}
                className="group flex items-center justify-between p-4 rounded-xl border border-zinc-800 bg-zinc-900/50 hover:bg-zinc-800 hover:border-zinc-700 transition-all"
              >
                <div className="flex items-center gap-3 text-sm font-medium text-zinc-200">
                  <div className="p-2 rounded-lg bg-zinc-800 group-hover:bg-teal-500/20 group-hover:text-teal-400 transition-colors">
                    <CloudUpload className="w-4 h-4" />
                  </div>
                  Import Dataset
                </div>
                <Plus className="w-4 h-4 text-zinc-600 group-hover:text-zinc-300 transition-colors" />
              </button>

              <Link
                href="/demo"
                className="group flex items-center justify-between p-4 rounded-xl border border-zinc-800 bg-zinc-900/50 hover:bg-zinc-800 hover:border-purple-500/50 hover:shadow-[0_0_15px_rgba(168,85,247,0.15)] transition-all mt-auto"
              >
                <div className="flex items-center gap-3 text-sm font-medium text-zinc-200">
                  <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400">
                    <Workflow className="w-4 h-4" />
                  </div>
                  E2E Demo Workflow
                </div>
                <ArrowRight className="w-4 h-4 text-zinc-600 group-hover:text-purple-400 group-hover:translate-x-1 transition-all" />
              </Link>
            </div>
          </div>
        </motion.div>

        {/* Right Column: Assets (Models & Datasets) */}
        <motion.div variants={itemVariants} className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Models Bento */}
          <div className="glass-panel rounded-2xl p-6 relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-b from-blue-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-sm font-semibold text-zinc-300 flex items-center gap-2 uppercase tracking-wider font-mono">
                  <Box className="w-4 h-4 text-blue-400" /> Checkpoint Registry
                </h3>
              </div>

              {isLoading ? (
                <div className="h-32 flex items-center justify-center">
                  <div className="w-5 h-5 rounded-full border-2 border-blue-500 border-t-transparent animate-spin"></div>
                </div>
              ) : models.length === 0 ? (
                <div className="h-32 flex items-center justify-center text-xs font-mono text-zinc-600">
                  NO_MODELS_FOUND
                </div>
              ) : (
                <div className="space-y-3">
                  {models.slice(0, 3).map((model) => (
                    <div key={model.id} className="flex items-center justify-between p-3 rounded-xl bg-zinc-900/50 border border-zinc-800/50 hover:border-zinc-700 transition-colors">
                      <div className="min-w-0 flex-1 pr-3">
                        <p className="truncate text-sm font-medium text-zinc-200">{model.model_name || model.id}</p>
                        <p className="truncate text-[10px] font-mono text-zinc-500 mt-0.5">{model.id}</p>
                      </div>
                      <div className={`shrink-0 w-2 h-2 rounded-full ${model.runnable ? "bg-teal-400 shadow-[0_0_8px_rgba(45,212,191,0.6)]" : "bg-zinc-600"}`}></div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Datasets Bento */}
          <div className="glass-panel rounded-2xl p-6 relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-b from-teal-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-sm font-semibold text-zinc-300 flex items-center gap-2 uppercase tracking-wider font-mono">
                  <Database className="w-4 h-4 text-teal-400" /> Verified Datasets
                </h3>
              </div>

              {isLoading ? (
                <div className="h-32 flex items-center justify-center">
                  <div className="w-5 h-5 rounded-full border-2 border-teal-500 border-t-transparent animate-spin"></div>
                </div>
              ) : datasets.length === 0 ? (
                <div className="h-32 flex items-center justify-center text-xs font-mono text-zinc-600">
                  NO_DATASETS_FOUND
                </div>
              ) : (
                <div className="space-y-3">
                  {datasets.slice(0, 3).map((dataset) => (
                    <div key={dataset.name} className="flex items-center justify-between p-3 rounded-xl bg-zinc-900/50 border border-zinc-800/50 hover:border-zinc-700 transition-colors">
                      <div className="min-w-0 flex-1 pr-3">
                        <p className="truncate text-sm font-medium text-zinc-200">{dataset.title || dataset.name}</p>
                        <p className="text-[10px] text-zinc-500 uppercase mt-0.5 tracking-wider">{dataset.task_id}</p>
                      </div>
                      <div className={`shrink-0 px-2 py-0.5 rounded text-[10px] font-bold ${dataset.anonymized ? "bg-teal-500/10 text-teal-400 border border-teal-500/20" : "bg-zinc-800 text-zinc-400 border border-zinc-700"}`}>
                        {dataset.anonymized ? "ANONYMIZED" : "PENDING"}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </motion.div>
      </div>

      {/* Intelligence & Alerts */}
      <motion.div variants={itemVariants} className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-panel rounded-2xl p-6">
           <h3 className="text-sm font-semibold text-zinc-300 flex items-center gap-2 mb-4 uppercase tracking-wider font-mono">
            <ShieldCheck className="w-4 h-4 text-purple-400" /> Active Intelligence
          </h3>
          {highRiskRun ? (
            <div className="flex items-start gap-3 p-4 rounded-xl border border-red-500/20 bg-red-500/5 relative overflow-hidden group">
              <div className="absolute inset-0 bg-red-500/10 translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-out"></div>
              <div className="p-2 rounded-full bg-red-500/20 shrink-0 relative z-10">
                <AlertTriangle className="w-4 h-4 text-red-400" />
              </div>
              <div className="relative z-10">
                <p className="text-sm font-semibold text-red-200">Critical Performance Degradation</p>
                <p className="mt-1 text-xs leading-relaxed text-red-300/70">
                  A recent benchmark exhibits Attack Success Rate (ASR) exceeding the 40% threshold. Recommended action: 
                  Export samples to the Retraining Backlog.
                </p>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-24 border border-dashed border-zinc-800 rounded-xl">
              <p className="text-xs font-mono text-zinc-600">SYSTEM_NOMINAL // NO_ALERTS</p>
            </div>
          )}
        </div>

        <div className="glass-panel rounded-2xl p-6 flex flex-col justify-center items-center text-center relative overflow-hidden">
           <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&w=800&q=80')] bg-cover bg-center opacity-5 mix-blend-overlay"></div>
           <div className="relative z-10 space-y-3">
             <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-blue-500/10 border border-blue-500/20 mb-2 shadow-[0_0_15px_rgba(59,130,246,0.2)]">
               <Workflow className="w-6 h-6 text-blue-400" />
             </div>
             <h3 className="text-lg font-semibold text-zinc-100">Closed-Loop Defense</h3>
             <p className="text-xs text-zinc-400 max-w-xs mx-auto leading-relaxed">
               Seamlessly pipe adversarial failures back into your training pipeline to harden the base model.
             </p>
             <Link href="/defense" className="inline-flex items-center gap-2 mt-2 text-xs font-semibold text-blue-400 hover:text-blue-300 transition-colors group">
               Enter Defense Matrix <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
             </Link>
           </div>
        </div>
      </motion.div>

      {uploadKind && (
        <QuickUploadDialog
          kind={uploadKind}
          project={activeProject}
          onClose={() => setUploadKind(null)}
          onSuccess={loadAssets}
        />
      )}
    </motion.div>
  );
}
