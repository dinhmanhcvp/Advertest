"use client";
import { useState } from "react";
import AdversarialStudio from "@/components/engine/AdversarialStudio";
import AnnotationCanvas from "@/components/engine/AnnotationCanvas";
import ProofOfCure from "@/components/engine/ProofOfCure";
import TemporalInspector from "@/components/engine/TemporalInspector";
import TriageMatrix from "@/components/engine/TriageMatrix";

export default function EnginePage() {
  const [selected, setSelected] = useState(null);
  return (
    <main className="scale-grid min-h-screen bg-black px-4 py-10 font-sans text-zinc-200 md:px-10">
      <div className="mx-auto max-w-6xl space-y-6">
        <header>
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-500">AdverTest · Egocentric PII Data Engine</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight text-white">From failure to cure, with lineage.</h1>
        </header>
        <TriageMatrix onSelect={setSelected} selectedId={selected} />
        <AdversarialStudio selectedId={selected} onSelect={setSelected} />
        <AnnotationCanvas />
        <TemporalInspector />
        <ProofOfCure />
      </div>
    </main>
  );
}
