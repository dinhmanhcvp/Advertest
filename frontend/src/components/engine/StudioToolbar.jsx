"use client";
import { motion } from "framer-motion";
import { SPRING } from "./ui";

export const LAYERS = [
  { key: "augmented", label: "Raw / Augmented" },
  { key: "boxes", label: "V1 / V2 BBoxes" },
  { key: "landmarks", label: "5-Point Landmarks" },
  { key: "imu", label: "IMU Motion Vector" },
];

export default function StudioToolbar({ layers, onToggle }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={SPRING}
      className="absolute left-1/2 top-3 z-10 flex -translate-x-1/2 gap-1 rounded-lg border border-white/10 bg-black/60 p-1 backdrop-blur-xl"
      onPointerDown={(e) => e.stopPropagation()}
    >
      {LAYERS.map((l) => (
        <button
          key={l.key}
          aria-pressed={layers[l.key]}
          onClick={() => onToggle(l.key)}
          className={`rounded-md px-2.5 py-1 font-mono text-[11px] transition ${
            layers[l.key] ? "bg-white text-black" : "text-zinc-400 hover:bg-white/10"
          }`}
        >
          {l.label}
        </button>
      ))}
    </motion.div>
  );
}
