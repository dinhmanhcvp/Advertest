import { FRAME_H, FRAME_W, type Box } from "./types";

/** Constant-velocity prediction step (the Kalman time-update): x̂ₜ = xₜ₋₁ + (xₜ₋₁ − xₜ₋₂). */
export function predictBox(prev2: Box, prev1: Box): Box {
  return prev1.map((v, i) => v + (v - prev2[i])) as Box;
}

/** Share of the (barrel-distorted) face box that stays inside the frame — the label-noise gate. */
export function boxIntegrity(box: Box, fisheye: number): number {
  const [x, y, w, h] = box;
  const s = 1 + fisheye * 0.12;
  const nx = FRAME_W / 2 + (x + w / 2 - FRAME_W / 2) * s - (w * s) / 2;
  const ny = FRAME_H / 2 + (y + h / 2 - FRAME_H / 2) * s - (h * s) / 2;
  const iw = Math.max(0, Math.min(nx + w * s, FRAME_W) - Math.max(nx, 0));
  const ih = Math.max(0, Math.min(ny + h * s, FRAME_H) - Math.max(ny, 0));
  return (iw * ih) / (w * s * h * s);
}
