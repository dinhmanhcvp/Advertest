import React from "react";
import { cn } from "@/lib/utils";

const BUTTON_VARIANTS = {
  primary: "bg-white text-black hover:bg-zinc-200 border-transparent shadow-[0_0_15px_rgba(255,255,255,0.15)] group relative overflow-hidden",
  secondary: "bg-white/[0.05] hover:bg-white/[0.1] text-white border border-white/10 shadow-[0_0_10px_rgba(0,0,0,0.5)]",
  outline: "bg-transparent hover:bg-white/[0.05] text-zinc-300 border border-white/20",
  ghost: "bg-transparent hover:bg-white/[0.05] text-zinc-400",
  danger: "bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/30 shadow-[0_0_10px_rgba(239,68,68,0.2)]",
  success: "bg-teal-500/10 hover:bg-teal-500/20 text-teal-400 border border-teal-500/30 shadow-[0_0_10px_rgba(20,184,166,0.2)]",
};

const BUTTON_SIZES = {
  sm: "px-2.5 py-1 text-xs rounded-md",
  md: "px-3.5 py-1.5 text-[13px] rounded-lg",
  lg: "px-5 py-2.5 text-[14px] rounded-lg font-semibold",
};

export default function Button({
  children,
  variant = "primary",
  size = "md",
  icon: Icon,
  className,
  disabled = false,
  ...props
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      className={cn(
        "inline-flex items-center justify-center gap-1.5 font-medium transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500/20 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer select-none",
        BUTTON_VARIANTS[variant] || BUTTON_VARIANTS.primary,
        BUTTON_SIZES[size] || BUTTON_SIZES.md,
        className,
      )}
      {...props}
    >
      {Icon && <Icon className="w-4 h-4 flex-shrink-0" />}
      {children}
    </button>
  );
}
