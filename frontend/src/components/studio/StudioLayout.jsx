"use client";

import { Activity, Brain, Crosshair, Database, LayoutDashboard, Settings2, Target, Workflow } from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { id: 1, name: "Data & Model", icon: Database },
  { id: 2, name: "Baseline", icon: Activity },
  { id: 3, name: "Triage", icon: Target },
  { id: 4, name: "Insights", icon: Brain },
  { id: 5, name: "Attack Plan", icon: Workflow },
  { id: 6, name: "Simulation", icon: Crosshair },
  { id: 7, name: "Retrain", icon: Settings2 },
];

export function StudioLayout({ children, version, dbSize, activeStep, setActiveStep }) {
  return (
    <div className="flex min-h-screen flex-col bg-[#050608] font-sans text-[#aeb2ba]">
      {/* Top Navbar */}
      <header className="flex h-14 items-center justify-between border-b border-white/[0.04] bg-[#0c0e11] px-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="flex h-7 w-7 items-center justify-center rounded bg-indigo-500/10 text-indigo-400">
            <Workflow className="h-4 w-4" />
          </div>
          <div className="font-mono text-sm font-semibold text-white">Closed-Loop Studio</div>
        </div>
        
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-2">
            <span className="text-white/40">Model Version:</span>
            <span className="rounded bg-white/[0.05] px-2 py-0.5 text-white/80">v{version}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-white/40">Retrain DB:</span>
            <span className="rounded bg-emerald-500/10 px-2 py-0.5 text-emerald-400">{dbSize} samples</span>
          </div>
          <Link href="/" className="ml-4 flex items-center gap-1.5 text-white/40 hover:text-white transition-colors">
            <LayoutDashboard className="h-3.5 w-3.5" /> Exit
          </Link>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-56 border-r border-white/[0.04] bg-[#090b0e] py-4">
          <div className="mb-4 px-4 text-[10px] font-bold uppercase tracking-wider text-white/30 font-mono">Pipeline</div>
          <nav className="space-y-0.5 px-2">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = activeStep === item.id;
              const isPast = activeStep > item.id;
              
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    // Allow navigating back but not forward unless unlocked
                    if (isPast || isActive) {
                      setActiveStep(item.id);
                    }
                  }}
                  disabled={!isPast && !isActive}
                  className={cn(
                    "flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-all duration-200 text-left",
                    isActive ? "bg-white/10 text-white font-medium shadow-[inset_2px_0_0_0_#8b5cf6]" : 
                    isPast ? "text-white/60 hover:bg-white/5 cursor-pointer" : "text-white/20 cursor-not-allowed opacity-50"
                  )}
                >
                  <Icon className={cn("h-4 w-4", isActive ? "text-indigo-400" : isPast ? "text-white/40" : "")} />
                  {item.name}
                </button>
              );
            })}
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto bg-[url('/grid.svg')] bg-center [background-size:40px_40px]">
          {children}
        </main>
      </div>
    </div>
  );
}
