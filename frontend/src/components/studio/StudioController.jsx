"use client";

import { useEffect, useState } from "react";
import { getMemory, loadState, saveState } from "@/lib/loop/store";
import { MODELS, DATASETS } from "@/lib/loop/catalog";
import { generateBaseline, evaluateBaseline } from "@/lib/loop/engine";

import { StudioLayout } from "./StudioLayout";
import StepSelect from "./StepSelect";
import StepBaseline from "./StepBaseline";
import StepTriage from "./StepTriage";
import StepInsights from "./StepInsights";
import StepPlan from "./StepPlan";
import StepAttack from "./StepAttack";
import StepRetrain from "./StepRetrain";

export default function StudioController() {
  const [activeStep, setActiveStep] = useState(1);
  const [appState, setAppState] = useState(null);
  
  // Transient run states
  const [runFlags, setRunFlags] = useState({
    evaluating: false,
    triaging: false,
    planning: false,
    attacking: false,
    retraining: false,
  });

  // Load state on mount
  useEffect(() => {
    const s = loadState();
    setAppState(s);
  }, []);

  if (!appState) return null; // loading

  // Generic flag updater
  const setFlag = (k, v) => setRunFlags((prev) => ({ ...prev, [k]: v }));

  // Transitions
  const handleSelect = (modelId, datasetId) => {
    const updated = {
      ...appState,
      model: MODELS.find(m => m.id === modelId),
      dataset: DATASETS.find(d => d.id === datasetId),
      baseline: null,
      triage: null,
      insights: null,
      plan: null,
      attackRes: null,
      retrainRes: null,
    };
    saveState(updated);
    setAppState(updated);
    setActiveStep(2);
  };

  const handleRunBaseline = async () => {
    setFlag("evaluating", true);
    // Simulate latency
    await new Promise((r) => setTimeout(r, 1200));
    
    // Call engine logic
    const { getEngineBaseline } = await import("@/lib/loop/engine");
    const baseline = getEngineBaseline(appState.model, appState.dataset);
    
    const updated = { ...appState, baseline };
    saveState(updated);
    setAppState(updated);
    setFlag("evaluating", false);
    setActiveStep(3);
  };

  const handleRunTriage = async () => {
    setFlag("triaging", true);
    await new Promise((r) => setTimeout(r, 1500));
    
    const { triageWorstSamples } = await import("@/lib/loop/engine");
    const triage = triageWorstSamples(appState.baseline);
    
    const updated = { ...appState, triage };
    saveState(updated);
    setAppState(updated);
    setFlag("triaging", false);
    setActiveStep(4);
  };
  
  const handleExtractInsights = async () => {
    setFlag("triaging", true); // reuse flag
    await new Promise((r) => setTimeout(r, 1000));
    
    const { extractInsights } = await import("@/lib/loop/engine");
    const insights = extractInsights(appState.triage);
    
    const updated = { ...appState, insights };
    saveState(updated);
    setAppState(updated);
    setFlag("triaging", false);
    setActiveStep(5);
  };

  const handleGeneratePlan = async () => {
    setFlag("planning", true);
    await new Promise((r) => setTimeout(r, 1500));
    
    const { generateAttackPlan } = await import("@/lib/loop/engine");
    const memory = getMemory();
    const plan = generateAttackPlan(appState.insights, memory);
    
    const updated = { ...appState, plan };
    saveState(updated);
    setAppState(updated);
    setFlag("planning", false);
    setActiveStep(6);
  };

  const handleAttack = async () => {
    setFlag("attacking", true);
    await new Promise((r) => setTimeout(r, 2000));
    
    const { executeAttackPlan } = await import("@/lib/loop/engine");
    const attackRes = executeAttackPlan(appState.plan, appState.triage);
    
    const updated = { ...appState, attackRes };
    saveState(updated);
    setAppState(updated);
    setFlag("attacking", false);
    setActiveStep(7);
  };

  const handleRetrain = async () => {
    setFlag("retraining", true);
    await new Promise((r) => setTimeout(r, 2500));
    
    const { simulateRetrain } = await import("@/lib/loop/engine");
    const retrainRes = simulateRetrain(appState);
    
    const updated = { ...appState, retrainRes };
    saveState(updated);
    setAppState(updated);
    setFlag("retraining", false);
  };

  const handleReset = () => {
    // Keep version but reset pipeline
    const s = loadState(); // loads fresh pipeline but persists version and memory
    setAppState(s);
    setActiveStep(1);
  };

  return (
    <StudioLayout 
      version={appState.version} 
      dbSize={appState.retrainDb.length} 
      activeStep={activeStep}
      setActiveStep={setActiveStep}
    >
      <div className="max-w-5xl mx-auto py-8 px-4">
        {activeStep === 1 && (
          <StepSelect 
            onSelect={handleSelect} 
            selectedModel={appState.model?.id}
            selectedDataset={appState.dataset?.id}
          />
        )}
        
        {activeStep === 2 && (
          <StepBaseline 
            baseline={appState.baseline} 
            running={runFlags.evaluating} 
            onRun={handleRunBaseline}
            modelName={appState.model?.name}
            datasetName={appState.dataset?.name}
          />
        )}

        {activeStep === 3 && (
          <StepTriage 
            triage={appState.triage}
            running={runFlags.triaging}
            onRun={handleRunTriage}
            baseline={appState.baseline}
          />
        )}

        {activeStep === 4 && (
          <StepInsights
            insights={appState.insights}
            running={runFlags.triaging}
            onExtract={handleExtractInsights}
            triage={appState.triage}
          />
        )}

        {activeStep === 5 && (
          <StepPlan
            plan={appState.plan}
            running={runFlags.planning}
            onGenerate={handleGeneratePlan}
            insights={appState.insights}
          />
        )}

        {activeStep === 6 && (
          <StepAttack
            attackRes={appState.attackRes}
            running={runFlags.attacking}
            onAttack={handleAttack}
            plan={appState.plan}
          />
        )}

        {activeStep === 7 && (
          <StepRetrain
            retrain={appState.retrainRes}
            running={runFlags.retraining}
            onRetrain={handleRetrain}
            onReset={handleReset}
          />
        )}
      </div>
    </StudioLayout>
  );
}
