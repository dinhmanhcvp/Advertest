// Procedural scene renderer + real pixel-level attack implementations.
// Every thumbnail and before/after view in the Studio is drawn here on a
// <canvas>, so the visual degradation the user sees is a genuine image
// transform (blur / noise / fisheye / rain …) rather than a static asset.

import { clamp, hashStr, makeRng, pick } from "./rng";

export const W = 384;
export const H = 240;

const SKIN = ["#f1c9a5", "#e0ac84", "#c68a62", "#a96a45", "#7d4a30"];
const HAIR = ["#1b1512", "#3a2618", "#5a3a1f", "#8a6a3a", "#2a2a2e"];
const SHIRT = ["#3b4a6b", "#6b3b4a", "#2f5d50", "#7a6a3b", "#4a4a52"];
const PORTRAIT_BG = [
  ["#2b4162", "#12100e"],
  ["#355c7d", "#6c5b7b"],
  ["#2c3e50", "#4ca1af"],
  ["#41295a", "#2f0743"],
  ["#3a3f47", "#1c1f24"],
];
const CAR_COLORS = ["#c0392b", "#2980b9", "#d9d9de", "#27ae60", "#f1c40f", "#34495e"];

const toBox = (gt) => ({
  x: (gt.cx - gt.w / 2) * W,
  y: (gt.cy - gt.h / 2) * H,
  w: gt.w * W,
  h: gt.h * H,
  cx: gt.cx * W,
  cy: gt.cy * H,
});

// ─────────────────────────────── Scenes ────────────────────────────────────

function drawFace(ctx, box, cls, rng, skin, hair) {
  const { cx, cy, w, h } = box;
  const profile = cls === "profile";
  const shift = profile ? w * 0.14 : 0;
  ctx.fillStyle = skin;
  ctx.beginPath();
  ctx.ellipse(cx, cy, w / 2, h / 2, 0, 0, Math.PI * 2);
  ctx.fill();
  // hair
  ctx.fillStyle = hair;
  ctx.beginPath();
  ctx.ellipse(cx, cy - h * 0.18, w * 0.54, h * 0.38, 0, Math.PI, Math.PI * 2);
  ctx.fill();
  ctx.fillRect(cx - w * 0.54, cy - h * 0.18, w * 0.1, h * 0.28);
  if (!profile) ctx.fillRect(cx + w * 0.44, cy - h * 0.18, w * 0.1, h * 0.28);
  // eyes
  const ey = cy - h * 0.05;
  for (const dir of profile ? [1] : [-1, 1]) {
    const ex = cx + dir * w * 0.2 + shift;
    ctx.fillStyle = "#f7f7f5";
    ctx.beginPath();
    ctx.ellipse(ex, ey, w * 0.09, h * 0.045, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#1d1a18";
    ctx.beginPath();
    ctx.arc(ex + shift * 0.2, ey, w * 0.04, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = hair;
    ctx.lineWidth = Math.max(1, w * 0.025);
    ctx.beginPath();
    ctx.moveTo(ex - w * 0.1, ey - h * 0.09);
    ctx.lineTo(ex + w * 0.1, ey - h * 0.1);
    ctx.stroke();
  }
  // nose + mouth
  ctx.strokeStyle = "rgba(60,30,20,.45)";
  ctx.lineWidth = Math.max(1, w * 0.02);
  ctx.beginPath();
  ctx.moveTo(cx + shift, cy);
  ctx.lineTo(cx + shift + w * 0.04, cy + h * 0.14);
  ctx.stroke();
  if (cls === "masked") {
    ctx.fillStyle = "#cfe3ee";
    ctx.beginPath();
    ctx.moveTo(cx - w * 0.46, cy + h * 0.06);
    ctx.quadraticCurveTo(cx, cy + h * 0.02, cx + w * 0.46, cy + h * 0.06);
    ctx.lineTo(cx + w * 0.34, cy + h * 0.44);
    ctx.quadraticCurveTo(cx, cy + h * 0.56, cx - w * 0.34, cy + h * 0.44);
    ctx.closePath();
    ctx.fill();
  } else {
    ctx.strokeStyle = "#8a3b3b";
    ctx.lineWidth = Math.max(1.2, w * 0.03);
    ctx.beginPath();
    ctx.arc(cx + shift, cy + h * 0.24, w * 0.14, 0.15 * Math.PI, 0.85 * Math.PI);
    ctx.stroke();
  }
  void rng;
}

function drawPortrait(ctx, s, rng) {
  const [c1, c2] = pick(rng, PORTRAIT_BG);
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, c1);
  g.addColorStop(1, c2);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
  for (let i = 0; i < 7; i += 1) {
    ctx.fillStyle = `rgba(255,255,255,${0.03 + rng() * 0.05})`;
    ctx.beginPath();
    ctx.arc(rng() * W, rng() * H * 0.8, 10 + rng() * 34, 0, Math.PI * 2);
    ctx.fill();
  }
  const box = toBox(s.gt);
  const skin = pick(rng, SKIN);
  const hair = pick(rng, HAIR);
  // background people (distractors)
  for (let i = 0; i < 2; i += 1) {
    const bw = box.w * (0.45 + rng() * 0.2);
    const bx = { cx: (box.cx + (i ? 1 : -1) * (box.w * (1.3 + rng() * 0.8)) + W) % W, cy: box.cy + box.h * 0.12, w: bw, h: bw * 1.25 };
    ctx.globalAlpha = 0.55;
    ctx.fillStyle = pick(rng, SHIRT);
    ctx.beginPath();
    ctx.ellipse(bx.cx, bx.cy + bx.h * 0.95, bx.w * 0.85, bx.h * 0.5, 0, 0, Math.PI * 2);
    ctx.fill();
    drawFace(ctx, bx, "frontal", rng, pick(rng, SKIN), pick(rng, HAIR));
    ctx.globalAlpha = 1;
  }
  // shoulders + neck
  ctx.fillStyle = pick(rng, SHIRT);
  ctx.beginPath();
  ctx.ellipse(box.cx, box.cy + box.h * 1.0, box.w * 1.15, box.h * 0.62, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = skin;
  ctx.fillRect(box.cx - box.w * 0.16, box.cy + box.h * 0.3, box.w * 0.32, box.h * 0.35);
  drawFace(ctx, box, s.cls, rng, skin, hair);
}

function drawCar(ctx, b, color, truck = false) {
  const { x, y, w, h } = b;
  ctx.fillStyle = color;
  ctx.fillRect(x, y + h * 0.35, w, h * 0.5);
  ctx.beginPath();
  if (truck) {
    ctx.fillRect(x + w * 0.05, y, w * 0.9, h * 0.65);
  } else {
    ctx.moveTo(x + w * 0.16, y + h * 0.35);
    ctx.lineTo(x + w * 0.28, y + h * 0.05);
    ctx.lineTo(x + w * 0.72, y + h * 0.05);
    ctx.lineTo(x + w * 0.86, y + h * 0.35);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "rgba(180,215,235,.8)";
    ctx.beginPath();
    ctx.moveTo(x + w * 0.22, y + h * 0.33);
    ctx.lineTo(x + w * 0.31, y + h * 0.12);
    ctx.lineTo(x + w * 0.69, y + h * 0.12);
    ctx.lineTo(x + w * 0.79, y + h * 0.33);
    ctx.closePath();
  }
  ctx.fill();
  ctx.fillStyle = "#101114";
  for (const px of [0.2, 0.72]) {
    ctx.beginPath();
    ctx.arc(x + w * (px + 0.04), y + h * 0.87, h * 0.14, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = "#ffd98a";
  ctx.fillRect(x + w * 0.02, y + h * 0.45, w * 0.07, h * 0.1);
  ctx.fillStyle = "#ff5a5a";
  ctx.fillRect(x + w * 0.91, y + h * 0.45, w * 0.07, h * 0.1);
}

function drawPerson(ctx, b, color, bike = false) {
  const { x, y, w, h } = b;
  const cx = x + w / 2;
  ctx.fillStyle = "#e0ac84";
  ctx.beginPath();
  ctx.arc(cx, y + h * 0.1, Math.min(w, h) * 0.17, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.fillRect(cx - w * 0.22, y + h * 0.2, w * 0.44, h * 0.4);
  ctx.fillStyle = "#2c3e50";
  ctx.fillRect(cx - w * 0.2, y + h * 0.6, w * 0.17, h * 0.38);
  ctx.fillRect(cx + w * 0.03, y + h * 0.6, w * 0.17, h * 0.38);
  if (bike) {
    ctx.strokeStyle = "#111";
    ctx.lineWidth = Math.max(1.5, w * 0.05);
    for (const dx of [-0.32, 0.32]) {
      ctx.beginPath();
      ctx.arc(cx + w * dx, y + h * 0.84, h * 0.15, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

function drawStreet(ctx, s, rng) {
  const horizon = H * 0.42;
  const sky = ctx.createLinearGradient(0, 0, 0, horizon);
  sky.addColorStop(0, "#5b8fc4");
  sky.addColorStop(1, "#cfe1ee");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, horizon);
  // skyline
  let x = 0;
  while (x < W) {
    const bw = 22 + rng() * 38;
    const bh = 30 + rng() * 80;
    ctx.fillStyle = `hsl(${210 + rng() * 20}, 12%, ${28 + rng() * 14}%)`;
    ctx.fillRect(x, horizon - bh, bw, bh);
    ctx.fillStyle = "rgba(255,235,160,.5)";
    for (let wy = horizon - bh + 6; wy < horizon - 6; wy += 10) {
      for (let wx = x + 4; wx < x + bw - 5; wx += 9) if (rng() > 0.45) ctx.fillRect(wx, wy, 4, 5);
    }
    x += bw + 2;
  }
  // ground + road
  ctx.fillStyle = "#4b5a4a";
  ctx.fillRect(0, horizon, W, H - horizon);
  ctx.fillStyle = "#3a3d45";
  ctx.beginPath();
  ctx.moveTo(W * 0.46, horizon);
  ctx.lineTo(W * 0.54, horizon);
  ctx.lineTo(W * 1.15, H);
  ctx.lineTo(-W * 0.15, H);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,.7)";
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 12]);
  ctx.beginPath();
  ctx.moveTo(W * 0.5, horizon);
  ctx.lineTo(W * 0.5, H);
  ctx.stroke();
  ctx.setLineDash([]);
  // trees
  for (let i = 0; i < 7; i += 1) {
    const tx = rng() * W;
    const ty = horizon + rng() * 10;
    ctx.fillStyle = "rgba(30,80,40,.85)";
    ctx.beginPath();
    ctx.arc(tx, ty - 10, 10 + rng() * 8, 0, Math.PI * 2);
    ctx.fill();
  }
  // distractors
  const target = toBox(s.gt);
  for (let i = 0; i < 3; i += 1) {
    const dw = 28 + rng() * 40;
    const dx = rng() * (W - dw);
    const dy = horizon + 12 + rng() * (H - horizon - 60);
    if (Math.abs(dx + dw / 2 - target.cx) < target.w * 0.9 && Math.abs(dy - target.y) < target.h) continue;
    drawCar(ctx, { x: dx, y: dy, w: dw, h: dw * 0.5 }, pick(rng, CAR_COLORS));
  }
  // target
  const { cls } = s;
  const b = { x: target.x, y: target.y, w: target.w, h: target.h };
  if (cls === "pedestrian") drawPerson(ctx, b, pick(rng, ["#c0392b", "#16a085", "#8e44ad", "#d35400"]));
  else if (cls === "cyclist") drawPerson(ctx, b, "#2980b9", true);
  else drawCar(ctx, b, pick(rng, CAR_COLORS), cls === "truck");
}

// ───────────────────────────── Pixel helpers ───────────────────────────────

function blurPass(src, dst, w, h, r, horizontal) {
  const n = horizontal ? w : h;
  const lines = horizontal ? h : w;
  const stride = horizontal ? 4 : w * 4;
  const lineStride = horizontal ? w * 4 : 4;
  const win = 2 * r + 1;
  for (let l = 0; l < lines; l += 1) {
    const base = l * lineStride;
    for (let c = 0; c < 3; c += 1) {
      let sum = 0;
      for (let i = -r; i <= r; i += 1) sum += src[base + Math.min(n - 1, Math.max(0, i)) * stride + c];
      for (let i = 0; i < n; i += 1) {
        dst[base + i * stride + c] = sum / win;
        sum += src[base + Math.min(n - 1, i + r + 1) * stride + c] - src[base + Math.max(0, i - r) * stride + c];
      }
    }
  }
}

function boxBlur(img, rx, ry) {
  const a = img.data;
  const b = new Uint8ClampedArray(a);
  if (rx > 0) blurPass(a, b, img.width, img.height, rx, true);
  else b.set(a);
  if (ry > 0) blurPass(b, a, img.width, img.height, ry, false);
  else a.set(b);
}

function mapPixels(img, fn) {
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const [r, g, b] = fn(d[i], d[i + 1], d[i + 2], i >> 2);
    d[i] = r;
    d[i + 1] = g;
    d[i + 2] = b;
  }
}

function remap(img, sampler) {
  const { width: w, height: h } = img;
  const src = new Uint8ClampedArray(img.data);
  const d = img.data;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const o = (y * w + x) * 4;
      const p = sampler(x, y, w, h);
      if (!p) {
        d[o] = 8;
        d[o + 1] = 8;
        d[o + 2] = 10;
        continue;
      }
      const x0 = Math.floor(p[0]);
      const y0 = Math.floor(p[1]);
      const fx = p[0] - x0;
      const fy = p[1] - y0;
      const x1 = Math.min(w - 1, x0 + 1);
      const y1 = Math.min(h - 1, y0 + 1);
      for (let c = 0; c < 3; c += 1) {
        const a = src[(y0 * w + x0) * 4 + c];
        const b = src[(y0 * w + x1) * 4 + c];
        const cc = src[(y1 * w + x0) * 4 + c];
        const dd = src[(y1 * w + x1) * 4 + c];
        d[o + c] = a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + cc * (1 - fx) * fy + dd * fx * fy;
      }
    }
  }
}

// ───────────────────────────── Attack effects ──────────────────────────────

const EFFECTS = {
  motion_blur(ctx, sev) {
    const img = ctx.getImageData(0, 0, W, H);
    boxBlur(img, Math.round(2 + sev * 4.5), 0);
    ctx.putImageData(img, 0, 0);
  },
  defocus_blur(ctx, sev) {
    const img = ctx.getImageData(0, 0, W, H);
    const r = Math.max(1, Math.round(0.8 + sev * 1.3));
    boxBlur(img, r, r);
    ctx.putImageData(img, 0, 0);
  },
  gaussian_noise(ctx, sev, rng) {
    const img = ctx.getImageData(0, 0, W, H);
    const sigma = 5 + sev * 9;
    mapPixels(img, (r, g, b) => {
      const n = (rng() + rng() + rng() - 1.5) * 2 * sigma;
      return [r + n, g + n * 0.9, b + n * 1.1];
    });
    ctx.putImageData(img, 0, 0);
  },
  low_light(ctx, sev) {
    const img = ctx.getImageData(0, 0, W, H);
    const factor = clamp(1 - 0.16 * sev, 0.12, 1);
    const gamma = 1 + 0.12 * sev;
    mapPixels(img, (r, g, b) => [
      Math.pow(r / 255, gamma) * factor * 255 * 0.92,
      Math.pow(g / 255, gamma) * factor * 255,
      Math.pow(b / 255, gamma) * factor * 255 * 1.1,
    ]);
    ctx.putImageData(img, 0, 0);
  },
  overexposure(ctx, sev) {
    const img = ctx.getImageData(0, 0, W, H);
    const gain = 1 + 0.2 * sev;
    const contrast = 1 + 0.08 * sev;
    mapPixels(img, (r, g, b) => [r, g, b].map((v) => ((v / 255 - 0.5) * contrast + 0.5) * gain * 255));
    ctx.putImageData(img, 0, 0);
  },
  sun_flare(ctx, sev, rng) {
    const fx = rng() > 0.5 ? W * 0.82 : W * 0.18;
    const fy = H * (0.12 + rng() * 0.12);
    const r = W * (0.2 + 0.09 * sev);
    ctx.save();
    ctx.globalCompositeOperation = "screen";
    const g = ctx.createRadialGradient(fx, fy, 0, fx, fy, r);
    g.addColorStop(0, `rgba(255,248,225,${clamp(0.55 + 0.1 * sev, 0, 1)})`);
    g.addColorStop(0.35, "rgba(255,214,140,.35)");
    g.addColorStop(1, "rgba(255,200,120,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
    for (let i = 1; i <= 3 + Math.round(sev / 2); i += 1) {
      const t = i / 5;
      const gx = fx + (W / 2 - fx) * t * 1.6;
      const gy = fy + (H / 2 - fy) * t * 1.6;
      ctx.fillStyle = `hsla(${30 + i * 30}, 90%, 70%, ${0.1 + 0.03 * sev})`;
      ctx.beginPath();
      ctx.arc(gx, gy, 8 + i * 6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  },
  fisheye(ctx, sev) {
    const img = ctx.getImageData(0, 0, W, H);
    const k = 0.1 + 0.2 * sev;
    remap(img, (x, y, w, h) => {
      const u = (x / w) * 2 - 1;
      const v = (y / h) * 2 - 1;
      const r2 = u * u + v * v;
      const f = 1 + k * r2;
      const sx = ((u * f + 1) / 2) * w;
      const sy = ((v * f + 1) / 2) * h;
      if (sx < 0 || sy < 0 || sx >= w - 1 || sy >= h - 1) return null;
      return [sx, sy];
    });
    ctx.putImageData(img, 0, 0);
  },
  grid_distortion(ctx, sev, rng) {
    const img = ctx.getImageData(0, 0, W, H);
    const amp = sev * 1.9;
    const p1 = 18 + rng() * 14;
    const p2 = 22 + rng() * 16;
    const ph = rng() * 6;
    remap(img, (x, y, w, h) => {
      const sx = clamp(x + amp * Math.sin(y / p1 + ph), 0, w - 2);
      const sy = clamp(y + amp * Math.sin(x / p2 + ph), 0, h - 2);
      return [sx, sy];
    });
    ctx.putImageData(img, 0, 0);
  },
  rain(ctx, sev, rng) {
    ctx.save();
    ctx.fillStyle = `rgba(28,38,56,${0.06 + 0.035 * sev})`;
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(205,220,240,.42)";
    ctx.lineWidth = 1;
    const count = 40 + sev * 55;
    for (let i = 0; i < count; i += 1) {
      const x = rng() * (W + 40);
      const y = rng() * H;
      const len = 9 + rng() * 14;
      ctx.globalAlpha = 0.25 + rng() * 0.45;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x - len * 0.28, y + len);
      ctx.stroke();
    }
    ctx.restore();
  },
  fog(ctx, sev) {
    ctx.save();
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, `rgba(208,214,224,${0.1 + 0.09 * sev})`);
    g.addColorStop(0.5, `rgba(214,220,230,${0.18 + 0.12 * sev})`);
    g.addColorStop(1, `rgba(208,214,224,${0.08 + 0.07 * sev})`);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
    ctx.restore();
  },
  cutout(ctx, sev, rng, sample) {
    const t = toBox(sample.gt);
    const n = 1 + Math.floor(sev / 2);
    ctx.save();
    for (let i = 0; i < n; i += 1) {
      const cover = clamp(0.22 + sev * 0.09, 0.2, 0.7);
      const bw = i === 0 ? t.w * (0.5 + cover * 0.6) : 20 + rng() * 40;
      const bh = i === 0 ? t.h * (0.3 + cover * 0.6) : 16 + rng() * 34;
      const bx = i === 0 ? t.x + (rng() > 0.5 ? -bw * 0.25 : t.w - bw * 0.75) : rng() * (W - bw);
      const by = i === 0 ? t.y + rng() * (t.h - bh) : rng() * (H - bh);
      ctx.fillStyle = pick(rng, ["#17181c", "#2b2d33", "#4b4f58", "#6a5a48"]);
      ctx.fillRect(bx, by, bw, bh);
    }
    ctx.restore();
  },
  jpeg(ctx, sev) {
    const f = 1 + sev * 0.6;
    const tmp = document.createElement("canvas");
    tmp.width = Math.max(8, Math.round(W / f));
    tmp.height = Math.max(8, Math.round(H / f));
    const tctx = tmp.getContext("2d");
    tctx.imageSmoothingEnabled = true;
    tctx.drawImage(ctx.canvas, 0, 0, tmp.width, tmp.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, W, H);
    ctx.imageSmoothingEnabled = true;
    const img = ctx.getImageData(0, 0, W, H);
    const step = 3 + sev * 5;
    mapPixels(img, (r, g, b) => [Math.round(r / step) * step, Math.round(g / step) * step, Math.round(b / step) * step]);
    ctx.putImageData(img, 0, 0);
  },
  color_shift(ctx, sev) {
    const img = ctx.getImageData(0, 0, W, H);
    mapPixels(img, (r, g, b) => [r * (1 + 0.06 * sev), g * (1 - 0.01 * sev), b * (1 - 0.08 * sev)]);
    ctx.putImageData(img, 0, 0);
  },
};

// Inherent dataset conditions are rendered with the same primitives.
function inherentEffects(sample) {
  const out = [];
  const c = sample.conds;
  const odd = sample.seed % 2 === 1;
  if (c.motion_blur > 0.2) out.push({ id: "motion_blur", severity: c.motion_blur * 4.2 });
  if (c.low_light > 0.2) {
    out.push({ id: "low_light", severity: c.low_light * 3.6 });
    out.push({ id: "gaussian_noise", severity: c.low_light * 2.4 });
  }
  if (c.glare > 0.2) out.push({ id: odd ? "sun_flare" : "overexposure", severity: c.glare * 3.8 });
  if (c.lens_distortion > 0.2) out.push({ id: "fisheye", severity: c.lens_distortion * 4 });
  if (c.weather > 0.2) out.push({ id: odd ? "rain" : "fog", severity: c.weather * 4 });
  if (c.compression > 0.2) out.push({ id: "jpeg", severity: c.compression * 3.6 });
  if (c.occlusion > 0.2) out.push({ id: "cutout", severity: c.occlusion * 4 });
  return out;
}

// ───────────────────────────── Overlay & public API ────────────────────────

function drawBox(ctx, b, color, label, dashed = false) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  if (dashed) ctx.setLineDash([5, 4]);
  ctx.strokeRect(b.x, b.y, b.w, b.h);
  ctx.setLineDash([]);
  if (label) {
    ctx.font = "600 10px ui-monospace, SFMono-Regular, Menlo, monospace";
    const tw = ctx.measureText(label).width + 8;
    const ly = b.y > 14 ? b.y - 13 : b.y + b.h;
    ctx.fillStyle = color;
    ctx.fillRect(b.x - 1, ly, tw, 13);
    ctx.fillStyle = "#06070a";
    ctx.fillText(label, b.x + 3, ly + 10);
  }
  ctx.restore();
}

/**
 * Draw one sample.  `attacks` = [{id, severity}] applied on top of the sample's
 * inherent conditions; `pred` = {score, conf, err} draws the model prediction.
 */
export function renderSample(canvas, sample, attacks = [], opts = {}) {
  if (!canvas) return;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const sceneRng = makeRng(sample.seed);
  ctx.clearRect(0, 0, W, H);
  if (sample.scene === "portrait") drawPortrait(ctx, sample, sceneRng);
  else drawStreet(ctx, sample, sceneRng);

  const apply = (list, salt) => {
    list.forEach((e, i) => {
      const fx = EFFECTS[e.id];
      if (!fx) return;
      const rng = makeRng(hashStr(`${sample.seed}:${salt}:${e.id}:${i}`));
      fx(ctx, clamp(e.severity, 0.3, 6), rng, sample);
    });
  };
  if (!opts.clean) apply(inherentEffects(sample), "base");
  apply(attacks, "atk");

  if (opts.showGt !== false && opts.overlay !== false) {
    const gt = toBox(sample.gt);
    drawBox(ctx, gt, "#34d399", opts.pred ? "GT" : sample.cls, true);
    if (opts.pred) {
      const { score, conf, err } = opts.pred;
      if (err === "missed") {
        ctx.save();
        ctx.font = "700 11px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "rgba(251,113,133,.95)";
        const label = "NO DETECTION";
        const tw = ctx.measureText(label).width + 12;
        ctx.fillRect(gt.x, gt.y + gt.h + 4, tw, 16);
        ctx.fillStyle = "#12060a";
        ctx.fillText(label, gt.x + 6, gt.y + gt.h + 16);
        ctx.restore();
      } else {
        const pr = makeRng(`${sample.seed}:pred`);
        const sgnX = pr() > 0.5 ? 1 : -1;
        const sgnY = pr() > 0.5 ? 1 : -1;
        const off = 1 - clamp(score);
        const shrink = 1 - off * 0.4;
        const pw = gt.w * shrink;
        const ph = gt.h * shrink;
        const pb = {
          x: gt.x + (gt.w - pw) / 2 + sgnX * off * gt.w * 0.55,
          y: gt.y + (gt.h - ph) / 2 + sgnY * off * gt.h * 0.45,
          w: pw,
          h: ph,
        };
        const color = err === "ok" ? "#60a5fa" : "#fbbf24";
        drawBox(ctx, pb, color, `${conf.toFixed(2)}`);
      }
    }
  }
}
