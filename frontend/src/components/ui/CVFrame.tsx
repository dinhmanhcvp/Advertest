"use client";
import { useEffect, useId, useState, type ReactNode } from "react";
import { FRAME_H, FRAME_W, type Box, type Pt } from "@/lib/types";

export interface CVFrameProps {
  /** Face box [x,y,w,h] in 160x120 reference units; the synthetic subject is drawn inside it. */
  box: Box;
  /** Gaussian blur radius in reference units. */
  blur?: number;
  /** Offsets (reference units) along which the subject is smeared — e.g. the IMU trajectory. */
  smear?: Pt[];
  /** 0..1 barrel distortion. */
  fisheye?: number;
  /** 1 = neutral; >1 over-exposed; <1 under-exposed. */
  exposure?: number;
  /** 0..1 sensor noise grain. */
  grain?: number;
  occlusion?: boolean;
  /** Real image to use; the synthetic renderer takes over if it is missing or fails to load. */
  imageUrl?: string | null;
  hud?: string;
  className?: string;
  /** Overlays drawn in the same coordinate system (boxes, landmarks, vectors). */
  children?: ReactNode;
}

function Subject({ box }: { box: Box }) {
  const [x, y, w, h] = box;
  const cx = x + w / 2;
  const cy = y + h / 2;
  const rx = w * 0.46;
  const ry = h * 0.5;
  return (
    <g>
      {/* shoulders + neck */}
      <path
        d={`M${cx - w * 0.9} ${FRAME_H + 4} Q${cx - w * 0.75} ${y + h * 1.08} ${cx - w * 0.16} ${y + h * 0.96} L${cx + w * 0.16} ${y + h * 0.96} Q${cx + w * 0.75} ${y + h * 1.08} ${cx + w * 0.9} ${FRAME_H + 4}Z`}
        fill="#27272a"
      />
      <rect x={cx - w * 0.14} y={y + h * 0.82} width={w * 0.28} height={h * 0.2} fill="#52525b" />
      {/* hair + head */}
      <ellipse cx={cx} cy={cy - h * 0.04} rx={rx * 1.06} ry={ry * 1.04} fill="#18181b" />
      <ellipse cx={cx} cy={cy + h * 0.02} rx={rx} ry={ry} fill="url(#skin)" />
      {/* eyes, brows, nose, mouth */}
      <ellipse cx={x + w * 0.3} cy={y + h * 0.36} rx={w * 0.075} ry={h * 0.032} fill="#0a0a0b" />
      <ellipse cx={x + w * 0.7} cy={y + h * 0.36} rx={w * 0.075} ry={h * 0.032} fill="#0a0a0b" />
      <path d={`M${x + w * 0.2} ${y + h * 0.29} h${w * 0.2} M${x + w * 0.6} ${y + h * 0.29} h${w * 0.2}`} stroke="#3f3f46" strokeWidth="1.2" />
      <path d={`M${cx} ${y + h * 0.42} l${-w * 0.06} ${h * 0.16} h${w * 0.12}z`} fill="#71717a" />
      <path d={`M${x + w * 0.33} ${y + h * 0.78} q${w * 0.17} ${h * 0.07} ${w * 0.34} 0`} stroke="#27272a" strokeWidth="1.6" fill="none" strokeLinecap="round" />
    </g>
  );
}

/**
 * Synthetic computer-vision frame: procedural scene, coordinate grid, lens / motion / sensor
 * effects and camera HUD. Used whenever no real image is available, so the viewport is never empty.
 */
export default function CVFrame({
  box,
  blur = 0,
  smear,
  fisheye = 0,
  exposure = 1,
  grain = 0,
  occlusion = false,
  imageUrl,
  hud = "CAM-01 · 1920×1080 · 30FPS",
  className,
  children,
}: CVFrameProps) {
  const uid = useId().replace(/[^a-zA-Z0-9]/g, "");
  const [imgOk, setImgOk] = useState(false);

  useEffect(() => {
    setImgOk(false);
    if (!imageUrl) return;
    const probe = new Image();
    probe.onload = () => setImgOk(true);
    probe.onerror = () => setImgOk(false);
    probe.src = imageUrl;
  }, [imageUrl]);

  const [x, y, w, h] = box;
  const gridLines: number[] = [0, 20, 40, 60, 80, 100, 120, 140, 160];
  const bow = fisheye * 0.55;
  const samples = smear && smear.length > 1 ? Array.from({ length: 7 }, (_, i) => smear[Math.round((i * (smear.length - 1)) / 6)]) : null;
  const subjectBlur = blur > 0 ? blur * (samples ? 0.35 : 0.6) : 0;

  return (
    <svg
      viewBox={`0 0 ${FRAME_W} ${FRAME_H}`}
      preserveAspectRatio="xMidYMid slice"
      className={className}
      role="img"
      aria-label="Synthetic egocentric camera frame"
    >
      <defs>
        <linearGradient id={`bg${uid}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#18181b" />
          <stop offset="1" stopColor="#050506" />
        </linearGradient>
        <linearGradient id="skin" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#a1a1aa" />
          <stop offset="1" stopColor="#6b6b73" />
        </linearGradient>
        <radialGradient id={`vig${uid}`} cx="50%" cy="50%" r="75%">
          <stop offset="0.55" stopColor="#000" stopOpacity="0" />
          <stop offset="1" stopColor="#000" stopOpacity={0.35 + fisheye * 0.6} />
        </radialGradient>
        <radialGradient id={`bloom${uid}`} cx="82%" cy="14%" r="60%">
          <stop offset="0" stopColor="#fff" stopOpacity="0.9" />
          <stop offset="1" stopColor="#fff" stopOpacity="0" />
        </radialGradient>
        <filter id={`b${uid}`} x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation={subjectBlur} />
        </filter>
        <filter id={`n${uid}`}>
          <feTurbulence type="fractalNoise" baseFrequency="1.4" numOctaves="2" seed="7" />
          <feColorMatrix type="saturate" values="0" />
        </filter>
        <clipPath id={`clip${uid}`}>
          <rect width={FRAME_W} height={FRAME_H} />
        </clipPath>
      </defs>

      <g clipPath={`url(#clip${uid})`}>
        <rect width={FRAME_W} height={FRAME_H} fill={`url(#bg${uid})`} />
        {/* barrel-distorted coordinate grid */}
        <g stroke="#ffffff" strokeOpacity="0.07" strokeWidth="0.3" fill="none">
          {gridLines.map((gx) => (
            <path key={`v${gx}`} d={`M${gx} 0 Q${gx + (gx - 80) * bow} 60 ${gx} 120`} />
          ))}
          {[0, 20, 40, 60, 80, 100, 120].map((gy) => (
            <path key={`h${gy}`} d={`M0 ${gy} Q80 ${gy + (gy - 60) * bow} 160 ${gy}`} />
          ))}
        </g>
        <path d={`M0 ${FRAME_H * 0.82} L${FRAME_W} ${FRAME_H * 0.78}`} stroke="#ffffff" strokeOpacity="0.05" />

        {imgOk && imageUrl && (
          <image href={imageUrl} width={FRAME_W} height={FRAME_H} preserveAspectRatio="xMidYMid slice" />
        )}

        {!imgOk && (
          <g
            style={{
              transform: `scale(${1 + fisheye * 0.1}, ${1 + fisheye * 0.04})`,
              transformOrigin: "80px 60px",
              filter: exposure !== 1 ? `brightness(${exposure})` : undefined,
            }}
          >
            {samples ? (
              samples.map(([dx, dy], i) => (
                <g key={i} opacity={i === 3 ? 0.55 : 0.2} transform={`translate(${dx} ${dy})`} filter={`url(#b${uid})`}>
                  <Subject box={box} />
                </g>
              ))
            ) : (
              <g filter={subjectBlur ? `url(#b${uid})` : undefined}>
                <Subject box={box} />
              </g>
            )}
            {occlusion && (
              <g>
                <rect x={x + w * 0.05} y={y + h * 0.5} width={w * 0.9} height={h * 0.34} rx="6" fill="#3f3f46" />
                <rect x={x + w * 0.05} y={y + h * 0.5} width={w * 0.9} height={h * 0.34} rx="6" fill="none" stroke="#52525b" />
              </g>
            )}
          </g>
        )}

        {exposure > 1.15 && (
          <>
            <rect width={FRAME_W} height={FRAME_H} fill="#fff" opacity={Math.min(0.55, (exposure - 1) * 0.45)} />
            <rect width={FRAME_W} height={FRAME_H} fill={`url(#bloom${uid})`} opacity={0.6} />
          </>
        )}
        {exposure < 0.85 && <rect width={FRAME_W} height={FRAME_H} fill="#000" opacity={(1 - exposure) * 0.7} />}
        {grain > 0 && <rect width={FRAME_W} height={FRAME_H} filter={`url(#n${uid})`} opacity={grain * 0.5} style={{ mixBlendMode: "overlay" }} />}
        <rect width={FRAME_W} height={FRAME_H} fill={`url(#vig${uid})`} />

        {children}

        {/* camera HUD */}
        <g stroke="#ffffff" strokeOpacity="0.5" strokeWidth="0.6" fill="none">
          <path d="M4 12 V4 H12 M148 4 H156 V12 M156 108 V116 H148 M12 116 H4 V108" />
          <path d="M76 60 h8 M80 56 v8" strokeOpacity="0.3" />
        </g>
        <text x="6" y="115" fill="#a1a1aa" fontSize="3.2" fontFamily="var(--font-jb-mono), monospace">
          {hud}
        </text>
        <circle cx="150" cy="8.5" r="1.4" fill="#ef4444" />
        <text x="141" y="9.6" fill="#a1a1aa" fontSize="3" fontFamily="var(--font-jb-mono), monospace">
          REC
        </text>
      </g>
    </svg>
  );
}
