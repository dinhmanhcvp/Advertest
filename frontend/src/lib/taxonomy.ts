import type { Box, FailureTag, Pt } from "./types";

/** Canonical failure registry — mirrors advertest/core/taxonomy.py. */
export const TAGS: Record<FailureTag, { label: string; hex: string; text: string; bar: string; mount: string }> = {
  error_motion_blur: {
    label: "Biomechanical Motion Blur",
    hex: "#f59e0b",
    text: "text-amber-400",
    bar: "bg-amber-500",
    mount: "Head-Mount (Walking)",
  },
  error_fisheye: {
    label: "Chest-Mount Fisheye Distortion",
    hex: "#8b5cf6",
    text: "text-violet-400",
    bar: "bg-violet-500",
    mount: "Chest-Mount",
  },
  error_sensor_noise: {
    label: "Extreme Sensor Exposure / ISO",
    hex: "#06b6d4",
    text: "text-cyan-400",
    bar: "bg-cyan-500",
    mount: "Smart Glasses",
  },
  error_occlusion: {
    label: "Hand/Object Occlusion",
    hex: "#71717a",
    text: "text-zinc-400",
    bar: "bg-zinc-500",
    mount: "Head-Mount (Static)",
  },
};

export const TAG_ORDER = Object.keys(TAGS) as FailureTag[];

export function asTag(value: string): FailureTag {
  return (value in TAGS ? value : "error_motion_blur") as FailureTag;
}

/** Five facial landmarks (L eye, R eye, nose, L mouth, R mouth) derived from a face box. */
export function landmarksFor(box: Box): Pt[] {
  const [x, y, w, h] = box;
  return (
    [
      [0.3, 0.36],
      [0.7, 0.36],
      [0.5, 0.56],
      [0.33, 0.78],
      [0.67, 0.78],
    ] as Pt[]
  ).map(([fx, fy]) => [Math.round((x + w * fx) * 10) / 10, Math.round((y + h * fy) * 10) / 10] as Pt);
}

export const LANDMARK_NAMES = ["L-EYE", "R-EYE", "NOSE", "L-MOUTH", "R-MOUTH"];

/** Biomechanical head-bob trajectory (sine + cubic drift), in the same form used by the backend engine. */
export function imuPath(hz: number, severity: number, pan: number, points = 90): Pt[] {
  const amp = (5 + severity * 10) * 0.4;
  const cycles = hz / 2;
  const tilt = (pan * Math.PI) / 180;
  return Array.from({ length: points }, (_, i) => {
    const t = i / (points - 1);
    const x = (t - 0.5) * amp;
    const y = amp * 0.3 * Math.sin(2 * Math.PI * cycles * t) + amp * 0.15 * (t - 0.5) ** 3 * 8;
    return [x * Math.cos(tilt) - y * Math.sin(tilt), x * Math.sin(tilt) + y * Math.cos(tilt)] as Pt;
  });
}

export function toPath(pts: Pt[], sx = 1, sy = sx, ox = 0, oy = 0): string {
  return pts.map(([x, y], i) => `${i ? "L" : "M"}${(ox + x * sx).toFixed(2)} ${(oy + y * sy).toFixed(2)}`).join(" ");
}

export const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
