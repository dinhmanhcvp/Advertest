import type { Config } from "tailwindcss";

/** Loaded from globals.css via `@config`. Design tokens for the Enterprise Data Engine workspace. */
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)", "Inter", "system-ui", "sans-serif"],
        mono: ["var(--font-jb-mono)", "JetBrains Mono", "ui-monospace", "monospace"],
      },
      colors: {
        signal: {
          fail: "#ef4444", // Model V1 / baseline failure
          physics: "#f59e0b", // physical distortion / IMU blur
          track: "#06b6d4", // temporal tracker / target locked
          cure: "#10b981", // Model V2 / cured
          fisheye: "#8b5cf6",
        },
      },
      borderRadius: { lg: "8px", md: "6px" },
      keyframes: {
        scan: { "0%": { transform: "translateY(-100%)" }, "100%": { transform: "translateY(100%)" } },
        ping2: { "75%, 100%": { transform: "scale(2.2)", opacity: "0" } },
      },
      animation: { scan: "scan 1.4s linear infinite", ping2: "ping2 1.8s cubic-bezier(0,0,.2,1) infinite" },
    },
  },
};

export default config;
