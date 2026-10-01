import React from "react";
import { cn } from "@/lib/utils";

export default function Card({ children, className, title, subtitle, headerAction, noPadding = false, ...props }) {
  return (
    <div
      className={cn(
        "glass-panel rounded-2xl transition-all relative overflow-hidden group",
        className,
      )}
      {...props}
    >
      {(title || headerAction) && (
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/10 bg-white/[0.01]">
          <div>
            {title && <h3 className="text-sm font-semibold text-white tracking-tight">{title}</h3>}
            {subtitle && <p className="text-[11px] font-mono text-zinc-500 mt-1 uppercase tracking-widest">{subtitle}</p>}
          </div>
          {headerAction && <div className="relative z-10">{headerAction}</div>}
        </div>
      )}
      <div className={cn(!noPadding && "p-5", "relative z-10")}>{children}</div>
    </div>
  );
}
