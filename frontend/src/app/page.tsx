"use client";
import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import TopNav from "@/components/layout/TopNav";
import AdversarialStudio from "@/components/modules/AdversarialStudio";
import ProofOfCure from "@/components/modules/ProofOfCure";
import TemporalInspector from "@/components/modules/TemporalInspector";
import TriageMatrix from "@/components/modules/TriageMatrix";
import { SPRING } from "@/components/ui/primitives";
import { loadWorkspace } from "@/lib/api";
import { MOCK_WORKSPACE } from "@/lib/mockData";
import type { WorkspaceData } from "@/lib/types";

/** Unified Enterprise Data Engine workspace: one page, four pipeline stages. */
export default function Workspace() {
  const [stage, setStage] = useState(0);
  const [demo, setDemo] = useState(true);
  const [data, setData] = useState<WorkspaceData>(MOCK_WORKSPACE);
  const [loading, setLoading] = useState(true);
  const [focus, setFocus] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    loadWorkspace(demo)
      .then((d) => alive && setData(d))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [demo]);

  const inspect = (id: string) => {
    setFocus(id);
    setStage(1);
  };

  return (
    <div className="relative min-h-screen bg-black">
      <div className="void-grid pointer-events-none absolute inset-x-0 top-0 h-[700px]" aria-hidden />
      <TopNav stage={stage} onStage={setStage} demo={demo} onDemo={setDemo} source={data.source} />
      <main className="relative mx-auto max-w-[1500px] px-4 py-8 md:px-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={stage}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={SPRING}
          >
            {stage === 0 && <TriageMatrix data={data} loading={loading} selected={focus} onSelect={inspect} onProceed={() => setStage(1)} />}
            {stage === 1 && <AdversarialStudio data={data} focusId={focus} />}
            {stage === 2 && <TemporalInspector data={data} />}
            {stage === 3 && <ProofOfCure data={data} />}
          </motion.div>
        </AnimatePresence>

        <nav className="mt-10 flex items-center justify-between border-t border-white/[0.08] pt-5 font-mono text-xs">
          <button disabled={stage === 0} onClick={() => setStage((s) => s - 1)} className="rounded-md border border-white/10 px-3 py-1.5 text-zinc-400 hover:bg-white/5 disabled:opacity-30">
            ← Previous stage
          </button>
          <span className="text-zinc-600">
            {stage + 1} / 4 · {data.source === "api" ? "FastAPI artifacts" : data.source === "artifact" ? "measured Colab run (static artifacts)" : "local placeholder store"}
          </span>
          <button disabled={stage === 3} onClick={() => setStage((s) => s + 1)} className="rounded-md border border-white/10 px-3 py-1.5 text-zinc-300 hover:bg-white/5 disabled:opacity-30">
            Next stage →
          </button>
        </nav>
      </main>
    </div>
  );
}
