"use client";

import { useEffect, useState } from "react";
import { MODELS, DATASETS } from "@/lib/loop/catalog";

import { StudioLayout } from "./StudioLayout";
import StepSelect from "./StepSelect";
import StepBaseline from "./StepBaseline";
import StepTriage from "./StepTriage";
import StepInsights from "./StepInsights";
import StepPlan from "./StepPlan";
import StepAttack from "./StepAttack";
import StepRetrain from "./StepRetrain";

const PIPELINE_KEY = "advertest.studio.state.v1";
const STORE_KEY = "advertest.closedloop.v1"; // from store.js

function loadPipeline() {
  try {
    const raw = window.localStorage.getItem(PIPELINE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {}
  return {
    version: 1,
    retrainDb: [],
    model: null,
    dataset: null,
    baseline: null,
    triage: null,
    insights: null,
    plan: null,
    attackRes: null,
    retrainRes: null,
  };
}

function savePipeline(state) {
  try {
    window.localStorage.setItem(PIPELINE_KEY, JSON.stringify(state));
  } catch {}
}

function loadRLMemory() {
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    if (raw) return JSON.parse(raw).memory || {};
  } catch {}
  return {};
}

function updateGlobalStore(updater) {
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    const store = raw ? JSON.parse(raw) : { v: 1, db: [], memory: {}, models: {}, cycles: [] };
    const next = updater(store);
    window.localStorage.setItem(STORE_KEY, JSON.stringify(next));
  } catch {}
}

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
    setAppState(loadPipeline());
  }, []);

  if (!appState) return null; // loading

  const setFlag = (k, v) => setRunFlags((prev) => ({ ...prev, [k]: v }));

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
    savePipeline(updated);
    setAppState(updated);
    setActiveStep(2);
  };

  const handleRunBaseline = async () => {
    setFlag("evaluating", true);
    await new Promise((r) => setTimeout(r, 800));
    const { evaluate } = await import("@/lib/loop/engine");
    
    // Evaluate clean
    const baseline = evaluate(appState.model.id, appState.dataset.id, { version: appState.version, fixes: appState.fixes, baseDelta: appState.baseDelta });
    
    const updated = { ...appState, baseline };
    savePipeline(updated);
    setAppState(updated);
    setFlag("evaluating", false);
    setActiveStep(3);
  };

  const handleRunTriage = async () => {
    setFlag("triaging", true);
    await new Promise((r) => setTimeout(r, 800));
    const { triage } = await import("@/lib/loop/engine");
    
    const triaged = triage(appState.baseline, 64);
    
    const updated = { ...appState, triage: triaged };
    savePipeline(updated);
    setAppState(updated);
    setFlag("triaging", false);
    setActiveStep(4);
  };
  
  const handleExtractInsights = async () => {
    setFlag("triaging", true);
    await new Promise((r) => setTimeout(r, 800));
    const { extractInsights } = await import("@/lib/loop/engine");
    
    const insights = extractInsights(appState.triage, appState.model.id);
    
    const updated = { ...appState, insights };
    savePipeline(updated);
    setAppState(updated);
    setFlag("triaging", false);
    setActiveStep(5);
  };

  const handleGeneratePlan = async () => {
    setFlag("planning", true);
    await new Promise((r) => setTimeout(r, 800));
    const { recommendCombos } = await import("@/lib/loop/engine");
    const memory = loadRLMemory();
    
    const plan = [];
    for (const insight of appState.insights) {
      const insightSamples = appState.triage.filter(s => s.attrTag === insight.tag);
      const recs = recommendCombos({
        insight,
        samples: insightSamples,
        modelId: appState.model.id,
        modelState: { version: appState.version, fixes: appState.fixes, baseDelta: appState.baseDelta },
        memory,
      });
      // automatically pick the recommended one
      plan.push(recs.find(r => r.recommended) || recs[0]);
    }
    
    const updated = { ...appState, plan };
    savePipeline(updated);
    setAppState(updated);
    setFlag("planning", false);
    setActiveStep(6);
  };

  const handleAttack = async () => {
    setFlag("attacking", true);
    await new Promise((r) => setTimeout(r, 800));
    const { runAttack, summarizeAttack, recordMemory, comboSignature } = await import("@/lib/loop/engine");
    
    // Execute attacks
    const results = [];
    let memory = loadRLMemory();
    const newDbEntries = [...(appState.retrainDb || [])];
    
    for (const p of appState.plan) {
      const insightSamples = appState.triage.filter(s => s.attrTag === p.tag);
      for (const s of insightSamples) {
        const res = runAttack(s, appState.model.id, { version: appState.version, fixes: appState.fixes, baseDelta: appState.baseDelta }, p.attacks);
        results.push(res);
        // Add to retrain DB
        if (res.effective || res.newlyMissed || res.flipped) {
          newDbEntries.push({
            id: s.id,
            tag: p.tag,
            sig: comboSignature(p.attacks),
            combo: p.attacks,
            drop: res.drop,
          });
        }
      }
      const summary = summarizeAttack(results.filter(r => appState.triage.find(s => s.id === r.sampleId)?.attrTag === p.tag));
      memory = recordMemory(memory, p.tag, p.attacks, summary);
    }
    
    // Save RL memory locally
    updateGlobalStore(s => ({ ...s, memory }));
    
    const updated = { ...appState, attackRes: summarizeAttack(results), retrainDb: newDbEntries };
    savePipeline(updated);
    setAppState(updated);
    setFlag("attacking", false);
    setActiveStep(7);
  };

  const handleRetrain = async () => {
    setFlag("retraining", true);
    await new Promise((r) => setTimeout(r, 1000));
    const { simulateRetrain, planFromDb, applyRewards } = await import("@/lib/loop/engine");
    
    const dbPlan = planFromDb(appState.retrainDb);
    const retrainRes = simulateRetrain({
      modelId: appState.model.id,
      datasetId: appState.dataset.id,
      modelState: { version: appState.version, fixes: appState.fixes, baseDelta: appState.baseDelta },
      dbEntries: appState.retrainDb,
      plan: dbPlan
    });
    
    // Apply rewards
    updateGlobalStore(s => ({
      ...s,
      memory: applyRewards(s.memory || {}, retrainRes.rewards, appState.retrainDb)
    }));
    
    const updated = { ...appState, retrainRes };
    savePipeline(updated);
    setAppState(updated);
    setFlag("retraining", false);
  };

  const handleReset = () => {
    const s = {
      version: appState.retrainRes.newState.version,
      fixes: appState.retrainRes.newState.fixes,
      baseDelta: appState.retrainRes.newState.baseDelta,
      retrainDb: [],
      model: appState.model,
      dataset: appState.dataset,
      baseline: null,
      triage: null,
      insights: null,
      plan: null,
      attackRes: null,
      retrainRes: null,
    };
    savePipeline(s);
    setAppState(s);
    setActiveStep(1);
  };

  return (
    <StudioLayout 
      version={appState.version || 1} 
      dbSize={appState.retrainDb?.length || 0} 
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

