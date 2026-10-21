# %% [markdown]
# # AdverTest — zero-local-disk pipeline (Colab GPU)
#
# Everything heavy runs **here**, never on the developer laptop:
# clone `derronqi/yolov7-face` → stream a micro-subset of WIDER FACE (HF streaming) and Ego4D clips
# (presigned URLs, decoded in RAM) → baseline inference → triage → constraint-driven attacks with a full
# audit trail → fine-tune → blind evaluation gate → export **only small JSON + thumbnails** (< 1 MB).
#
# Run top to bottom (Runtime → GPU). Drop the exported `advertest_artifacts.zip` into `data/demo_artifacts/`
# of the repo and the UI/API serve those real results.
#
# Honest scope notes
# * WIDER FACE ships boxes only, so *ground truth* is boxes; the 5 landmarks come from the model
#   (and become pseudo-labels for fine-tuning, transformed with the same geometry as the image).
# * Ego4D needs a licence: supply presigned URLs in `EGO4D_PRESIGNED_URLS`. They are decoded as streams,
#   never saved. Without them the "target domain" is WIDER FACE passed through the physical attack engine
#   (exact ground truth) and the report says so explicitly.

# %% [markdown]
# ## 0 · Config and environment

# %%
from __future__ import annotations

import base64
import calendar
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

CFG = dict(
    workdir="/content/advertest",
    repo_url="https://github.com/derronqi/yolov7-face.git",
    weights_gdrive_id="1Mona-I4PclJr5mjX1qb8dgDeMpYyBcwM",  # yolov7-tiny-face, from the repo README
    wider_hf_id="CUHK-CSE/wider_face",
    wider_dir=os.environ.get("WIDER_DIR", ""),  # optional: extracted WIDER_val + wider_face_split
    n_pool=int(os.environ.get("N_POOL", "400")),  # images streamed from WIDER (train/blind are carved out of this)
    n_blind=int(os.environ.get("N_BLIND", "150")),  # held-out images never used for synthesis or training
    img_size=640,
    det_conf=0.25,  # detector output threshold
    hard_conf=0.40,  # hard negative: best matched confidence below this ...
    hard_iou=0.50,  # ... or best IoU below this
    ego4d_urls=[u for u in os.environ.get("EGO4D_PRESIGNED_URLS", "").split(",") if u.strip()],
    ego4d_auto=os.environ.get("EGO4D_AUTO_PRESIGN", "0") == "1",  # presign clips with AWS keys kept in Colab Secrets
    ego4d_clips=int(os.environ.get("EGO4D_CLIPS", "4")),
    ego4d_bucket=os.environ.get("EGO4D_BUCKET", "ego4d-consortium-sharing"),
    # The Ego4D CLI never lists the bucket: it reads public/<version>/<dataset>/manifest.csv (GetObject only).
    ego4d_versions=[v for v in os.environ.get("EGO4D_VERSION", "v2_1,v2,v1").split(",") if v.strip()],
    ego4d_datasets=[d for d in os.environ.get("EGO4D_DATASETS", "clips,full_scale").split(",") if d.strip()],
    ego4d_benchmark=os.environ.get("EGO4D_BENCHMARK", "social"),
    ego4d_frames=48,
    ego4d_stride=2,
    epochs=int(os.environ.get("EPOCHS", "30")),
    batch=16,
    run_control=os.environ.get("RUN_CONTROL", "1") == "1",  # clean-only fine-tune with the same budget
    n_boot=int(os.environ.get("N_BOOT", "300")),
    corruption_repeats=int(os.environ.get("CORRUPTION_REPEATS", "3")),
    max_base_drop=0.01,
    min_corrupted_gain=0.05,
    seed=int(os.environ.get("SEED", "7")),
    artifact_put_url=os.environ.get("ARTIFACT_PUT_URL", ""),  # optional presigned PUT to the Core Data Lake
)

CANONICAL_TAGS = ("error_fisheye", "error_motion_blur", "error_sensor_noise", "error_occlusion")
REF_W, REF_H = 160, 120  # reference frame used by the UI


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# yolov7-face predates NumPy 2 and uses removed aliases such as `np.int`. This shim restores them in the trainer
# process (and its dataloader workers) and then runs the real script, so no file of the repo is modified.
NUMPY_COMPAT_SHIM = (
    "import sys, runpy, numpy as np\n"
    "for _n, _t in (('int', int), ('float', float), ('bool', bool), ('object', object), ('str', str), ('complex', complex)):\n"
    "    if not hasattr(np, _n):\n"
    "        setattr(np, _n, _t)\n"
    # No W&B (it needs an API key and its image logging crashes with keypoint outputs): make `import wandb` fail.
    "sys.modules['wandb'] = None\n"
    # The repo's per-epoch validation (test.py) is replaced by a no-op: its keypoint logging is broken and the real
    # accuracy numbers come from this pipeline's own blind evaluation on held-out images.
    "import types\n"
    "_stub = types.ModuleType('test')\n"
    "_stub.test = lambda *a, **k: ((0.0,) * 7, np.zeros(1), (0.0, 0.0, 0.0))\n"
    "sys.modules['test'] = _stub\n"
    "sys.argv = sys.argv[1:]\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
)


class CommandFailed(RuntimeError):
    """A subprocess failed; the message carries the tail of its output so notebook users can see why."""


def run_logged(cmd, cwd=None, tail_lines: int = 40, env: dict | None = None) -> None:
    """Run a command, streaming its output into the notebook cell (a bare subprocess.run would send it to the kernel log)."""
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env={**os.environ, **(env or {})}
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        del tail[:-tail_lines]
    code = proc.wait()
    if code:
        raise CommandFailed(f"command exited with {code}: {' '.join(map(str, cmd[:3]))} ...\n--- last output ---\n" + "".join(tail))


def setup_environment() -> Path:
    """Colab only: install light deps, shallow-clone the model repo, fetch weights into /content."""
    wd = Path(CFG["workdir"])
    wd.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "datasets", "gdown", "av", "pillow", "boto3"], check=True)
    repo = wd / "yolov7_face"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", CFG["repo_url"], str(repo)], check=True)
    weights = wd / "yolov7-tiny-face.pt"
    if not weights.exists():
        import gdown

        gdown.download(id=CFG["weights_gdrive_id"], output=str(weights), quiet=False)
    assert weights.exists() and weights.stat().st_size > 1_000_000, "weights download failed"
    return wd


# %% [markdown]
# ## 1 · Geometry, metrics, triage (pure functions — unit-tested offline)


# %%
def iou_xyxy(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Det:
    box: tuple  # x1, y1, x2, y2 in image pixels
    score: float
    lmk: np.ndarray | None = None  # (5, 2)


def average_precision(dets_per_img, gts_per_img, iou_thr: float = 0.5) -> float:
    """All-point interpolated AP for a single class (WIDER 'face'). Greedy match by descending score."""
    n_gt = sum(len(g) for g in gts_per_img)
    if n_gt == 0:
        return 0.0
    flat = [(d.score, i, d.box) for i, ds in enumerate(dets_per_img) for d in ds]
    flat.sort(key=lambda t: -t[0])
    used = [np.zeros(len(g), bool) for g in gts_per_img]
    tp = np.zeros(len(flat))
    for k, (_, i, box) in enumerate(flat):
        best, bj = 0.0, -1
        for j, g in enumerate(gts_per_img[i]):
            if used[i][j]:
                continue
            v = iou_xyxy(box, g)
            if v > best:
                best, bj = v, j
        if bj >= 0 and best >= iou_thr:
            used[i][bj] = True
            tp[k] = 1
    if not len(flat):
        return 0.0
    ctp = np.cumsum(tp)
    rec = ctp / n_gt
    prec = ctp / (np.arange(len(flat)) + 1)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def recall_at_fppi(dets_per_img, gts_per_img, targets=(0.1, 1.0), iou_thr: float = 0.5) -> dict:
    """Recall at fixed False-Positives-Per-Image — the operating-point metric for PII redaction.

    Detections are ranked by score across all images (greedy one-to-one matching per image); for each target the
    recall is read at the lowest score threshold whose cumulative FP/image stays <= target.
    """
    n_img = max(1, len(gts_per_img))
    n_gt = sum(len(g) for g in gts_per_img)
    flat = sorted(((d.score, i, d.box) for i, ds in enumerate(dets_per_img) for d in ds), key=lambda t: -t[0])
    used = [np.zeros(len(g), bool) for g in gts_per_img]
    tp = fp = 0
    curve = []  # (fp_per_image, recall) after each detection
    for _, i, box in flat:
        best, bj = 0.0, -1
        for j, g in enumerate(gts_per_img[i]):
            if used[i][j]:
                continue
            v = iou_xyxy(box, g)
            if v > best:
                best, bj = v, j
        if bj >= 0 and best >= iou_thr:
            used[i][bj] = True
            tp += 1
        else:
            fp += 1
        curve.append((fp / n_img, tp / n_gt if n_gt else 0.0))
    out = {}
    for t in targets:
        out[f"recall@fppi={t}"] = round(max([r for f, r in curve if f <= t], default=0.0), 4)
    return out


def match_gt(gt_boxes, dets):
    """For every GT: (best IoU, score of that detection or 0)."""
    out = []
    for g in gt_boxes:
        best, sc = 0.0, 0.0
        for d in dets:
            v = iou_xyxy(g, d.box)
            if v > best:
                best, sc = v, d.score
        out.append((best, sc))
    return out


def triage_image(gt_boxes, dets, iou_thr=0.5, conf_thr=0.4) -> dict:
    """Hard negative = some GT face is missed (IoU < thr) or only weakly detected (conf < thr)."""
    m = match_gt(gt_boxes, dets)
    if not m:
        return dict(hard=False, worst_iou=1.0, worst_conf=1.0, worst_idx=-1)
    idx = min(range(len(m)), key=lambda i: (m[i][0] >= iou_thr, m[i][1] if m[i][0] >= iou_thr else m[i][0]))
    wi, wc = m[idx]
    hard = any(i < iou_thr or c < conf_thr for i, c in m)
    return dict(hard=bool(hard), worst_iou=float(wi), worst_conf=float(wc), worst_idx=idx)


def classify_failure(img_bgr, box, attrs: dict | None = None) -> str:
    """Canonical failure tag from WIDER human attributes when present, otherwise measured image statistics."""
    if attrs:
        if attrs.get("blur", 0) >= 2:
            return "error_motion_blur"
        if attrs.get("illumination", 0) >= 1 and attrs.get("blur", 0) < 2:
            return "error_sensor_noise"
        if attrs.get("occlusion", 0) >= 2:
            return "error_occlusion"
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < 4:
        return "error_occlusion"
    g = cv2.cvtColor(cv2.resize(crop, (64, 64)), cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    mean = float(g.mean())
    clipped = float(((g > 250) | (g < 5)).mean())
    cx, cy = (x1 + x2) / 2 - w / 2, (y1 + y2) / 2 - h / 2
    radial = math.hypot(cx / (w / 2), cy / (h / 2)) / math.sqrt(2)
    if mean > 205 or mean < 45 or clipped > 0.25:
        return "error_sensor_noise"
    if lap < 60:
        return "error_motion_blur"
    if radial > 0.55:
        return "error_fisheye"
    return "error_occlusion"


# %% [markdown]
# ## 2 · Constraint-driven attack engine (+ landmark/bbox integrity)


# %%
def _radial_maps(w, h, k1, k2):
    f = max(w, h) / 2.0
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = (xx - cx) / f, (yy - cy) / f
    r2 = dx * dx + dy * dy
    s = 1.0 + k1 * r2 + k2 * r2 * r2
    return (cx + dx * s * f).astype(np.float32), (cy + dy * s * f).astype(np.float32)


def distort_image(img, k1, k2):
    mx, my = _radial_maps(img.shape[1], img.shape[0], k1, k2)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def distort_points(pts, w, h, k1, k2):
    """Forward map matching `distort_image`: solve r_src = r_dst (1 + k1 r_dst^2 + k2 r_dst^4) by Newton."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    f = max(w, h) / 2.0
    c = np.array([w / 2.0, h / 2.0])
    d = (pts - c) / f
    rs = np.linalg.norm(d, axis=1)
    rd = rs.copy()
    for _ in range(12):
        g = rd * (1 + k1 * rd**2 + k2 * rd**4) - rs
        dg = 1 + 3 * k1 * rd**2 + 5 * k2 * rd**4
        rd = rd - g / np.where(np.abs(dg) < 1e-9, 1e-9, dg)
    scale = np.divide(rd, rs, out=np.ones_like(rs), where=rs > 1e-9)
    return c + d * scale[:, None] * f


def box_after_distortion(box, w, h, k1, k2):
    x1, y1, x2, y2 = box
    t = np.linspace(0, 1, 9)
    edge = np.concatenate(
        [
            np.stack([x1 + (x2 - x1) * t, np.full_like(t, y1)], 1),
            np.stack([x1 + (x2 - x1) * t, np.full_like(t, y2)], 1),
            np.stack([np.full_like(t, x1), y1 + (y2 - y1) * t], 1),
            np.stack([np.full_like(t, x2), y1 + (y2 - y1) * t], 1),
        ]
    )
    p = distort_points(edge, w, h, k1, k2)
    return (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))


def inside_fraction(box, w, h) -> float:
    x1, y1, x2, y2 = box
    area = max(1e-9, (x2 - x1) * (y2 - y1))
    iw = max(0.0, min(x2, w) - max(x1, 0))
    ih = max(0.0, min(y2, h) - max(y1, 0))
    return float(iw * ih / area)


def imu_kernel(severity: int, hz: float, pan_deg: float = 0.0, n: int = 240):
    """Biomechanical blur: sine head-bob + cubic drift, rotated by the head-pan angle."""
    sev = max(1, min(5, severity))
    k = 5 + sev * 10
    ker = np.zeros((k, k), np.float32)
    c, amp, tilt = k // 2, k * 0.4, math.radians(pan_deg)
    for i in range(n):
        t = i / n
        x = (t - 0.5) * amp
        y = amp * 0.3 * math.sin(2 * math.pi * (hz / 2.0) * t) + amp * 0.15 * (t - 0.5) ** 3 * 8
        xr, yr = x * math.cos(tilt) - y * math.sin(tilt), x * math.sin(tilt) + y * math.cos(tilt)
        xi, yi = int(round(c + xr)), int(round(c + yr))
        if 0 <= xi < k and 0 <= yi < k:
            ker[yi, xi] += 1
    ker = cv2.GaussianBlur(ker, (3, 3), 0)
    s = ker.sum()
    if s <= 0:
        ker[c, c] = 1.0
        return ker
    return ker / s


CONFLICTS = [{"flare", "low_light"}, {"flare", "high_iso"}, {"flare", "heavy_rain"}]


def exclusion_ok(tags) -> bool:
    ts = set(tags)
    return not any(c <= ts for c in CONFLICTS)


def tone_and_noise(img, severity, over: bool, rng):
    sev = max(1, min(5, severity))
    f = img.astype(np.float32)
    if over:
        f = f * (1.0 + 0.35 * sev) + 8 * sev
        tags = ["tone_curve", "flare"]
        h, w = f.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w * rng.uniform(0.6, 0.9), h * rng.uniform(0.05, 0.3)
        glow = np.exp(-(((xx - cx) / (0.35 * w)) ** 2 + ((yy - cy) / (0.35 * h)) ** 2))
        f += (glow[..., None] * 60 * sev)
    else:
        f = (f / 255.0) ** (1.0 + 0.45 * sev) * 255.0 * (1.0 - 0.06 * sev)
        f += rng.normal(0, 3.5 * sev, f.shape)  # ISO noise
        tags = ["low_light", "high_iso"]
    assert exclusion_ok(tags), "exclusion matrix violated"
    return np.clip(f, 0, 255).astype(np.uint8), tags


def hand_occlusion(img, box, severity, rng):
    out = img.copy()
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pw, ph = w * (0.25 + 0.12 * severity), h * (0.3 + 0.07 * severity)
    cx, cy = x1 + w * rng.uniform(0.3, 0.7), y1 + h * rng.uniform(0.55, 0.85)
    p1, p2 = (int(cx - pw / 2), int(cy - ph / 2)), (int(cx + pw / 2), int(cy + ph / 2))
    cv2.rectangle(out, p1, p2, (70, 85, 120), -1)
    return out


TAG_CHAIN = {
    "error_fisheye": ["OpticalDistortion"],
    "error_motion_blur": ["OpticalDistortion", "BiomechanicalBlur"],
    "error_sensor_noise": ["ToneCurve", "ISONoise"],
    "error_occlusion": ["HandOcclusion"],
}


@dataclass
class Attacked:
    image: np.ndarray
    boxes: list  # transformed GT boxes (xyxy)
    lmk: list  # transformed pseudo-landmarks per box (5,2) or None
    chain: list
    severity: int
    params: dict
    integrity: float
    discarded: bool


def synthesize(img, boxes, lmks, tag: str, severity: int, rng, hz: float | None = None) -> Attacked:
    """Apply the attack chain for `tag`; discard if any box loses > 50% of its area outside the frame."""
    h, w = img.shape[:2]
    sev = max(1, min(5, severity))
    hz = float(hz if hz is not None else rng.uniform(2.0, 5.0))
    out, nb, nl = img, [tuple(b) for b in boxes], [None if l is None else np.array(l, float) for l in lmks]
    params: dict = {"severity": sev}
    if tag == "error_fisheye" or tag == "error_motion_blur":
        k1 = (0.12 if tag == "error_motion_blur" else 0.10) * sev * (0.6 if tag == "error_motion_blur" else 1.0)
        k2 = 0.04 * sev * (0.5 if tag == "error_motion_blur" else 1.0)
        out = distort_image(out, k1, k2)
        nb = [box_after_distortion(b, w, h, k1, k2) for b in nb]
        nl = [None if l is None else distort_points(l, w, h, k1, k2) for l in nl]
        params.update(k1=round(k1, 3), k2=round(k2, 3))
    if tag == "error_motion_blur":
        pan = float(rng.uniform(0, 30))
        out = cv2.filter2D(out, -1, imu_kernel(sev, hz, pan))
        params.update(imu_hz=round(hz, 2), pan_deg=round(pan, 1), kernel_px=5 + sev * 10)
    elif tag == "error_sensor_noise":
        over = bool(rng.random() < 0.5)
        out, tags = tone_and_noise(out, sev, over, rng)
        params.update(mode="over_exposure" if over else "low_light_high_iso", tags=tags)
    elif tag == "error_occlusion":
        for b in nb:
            out = hand_occlusion(out, b, sev, rng)
    frac = min((inside_fraction(b, w, h) for b in nb), default=1.0)
    return Attacked(out, nb, nl, TAG_CHAIN[tag], sev, params, float(frac), frac < 0.5)


def explain(tag: str, dist: dict, n_hard: int, med_conf: float, att: Attacked) -> str:
    """Human-readable, data-grounded justification stored in the audit trail."""
    share = dist.get(tag, 0.0)
    p = att.params
    extra = {
        "error_fisheye": f"radial distortion k1={p.get('k1')}, k2={p.get('k2')}",
        "error_motion_blur": f"{p.get('imu_hz')} Hz biomechanical trajectory, {p.get('pan_deg')} deg head pan",
        "error_sensor_noise": f"{p.get('mode')} tone curve + ISO noise (exclusion matrix: flare != low-light/high-ISO)",
        "error_occlusion": "hand/strap occluder over the lower face",
    }[tag]
    return (
        f"{share * 100:.1f}% of {n_hard} baseline hard negatives were tagged {tag} "
        f"(median V1 confidence {med_conf:.2f}); applied {extra} at severity {att.severity}; "
        f"bbox integrity {att.integrity * 100:.0f}% inside frame."
    )


PIPELINE_VERSION = "2026-10-21.3"


def rng_for(seed: int, *key) -> np.random.Generator:
    """Independent, reproducible RNG for one (seed, key) pair — lets any single sample be regenerated exactly."""
    digest = hashlib.sha256(repr((int(seed), *key)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def draw_attack(seed: int, key, weights=None, tags=CANONICAL_TAGS):
    """Plan an attack for `key`: returns (tag, severity). Uses its own RNG stream ("plan")."""
    r = rng_for(seed, *key, "plan")
    if weights is None:
        tag = tags[int(r.integers(0, len(tags)))]
    else:
        p = np.asarray(weights, float)
        tag = tags[int(r.choice(len(tags), p=p / p.sum()))]
    return tag, int(r.integers(2, 6))


def apply_attack(img, boxes, lmks, seed: int, key, tag: str, severity: int) -> Attacked:
    """Apply a planned attack with its own RNG stream ("apply") — pure function of (inputs, seed, key, tag, severity)."""
    return synthesize(img, boxes, lmks, tag, severity, rng_for(seed, *key, "apply"))


def reproduce_sample(img, boxes, lmks, record: dict) -> Attacked:
    """Rebuild a synthesized image from its audit record + the source image (no stored pixels needed)."""
    rp = record["reproduce"]
    return apply_attack(img, boxes, lmks, rp["seed"], rp["key"], rp["tag"], rp["severity"])


def dhash(img, size: int = 8) -> int:
    """64-bit difference hash used to detect near-duplicate images across splits."""
    g = cv2.cvtColor(cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def find_leaks(train_imgs, blind_imgs, max_dist: int = 6) -> set:
    """Indices of train images that are near-duplicates of any blind image (Hamming distance <= max_dist)."""
    bh = [dhash(b) for b in blind_imgs]
    leaks = set()
    for i, t in enumerate(train_imgs):
        h = dhash(t)
        if any(bin(h ^ x).count("1") <= max_dist for x in bh):
            leaks.add(i)
    return leaks


def env_versions() -> dict:
    out = {"python": sys.version.split()[0], "numpy": np.__version__, "opencv": cv2.__version__}
    try:
        import torch

        out["torch"] = torch.__version__
    except Exception:
        pass
    return out


# %% [markdown]
# ## 3 · Data: stream WIDER FACE, stream Ego4D (presigned, in RAM)


# %%
def stream_wider(n: int):
    """Yield (image_bgr, gt_xyxy list, attrs list) for n WIDER FACE validation images.

    Order: user-provided WIDER_DIR → Hugging Face parquet streaming → official archive on the Colab VM disk.
    """
    if CFG["wider_dir"]:
        yield from _wider_from_dir(Path(CFG["wider_dir"]), n)
        return
    try:
        it = _hf_wider()
    except Exception as e:  # script-based HF datasets are no longer loadable with recent `datasets`
        log(f"HF streaming unavailable ({type(e).__name__}); using the official WIDER archive on the Colab VM disk")
        yield from _wider_from_dir(download_wider_val(), n)
        return
    yield from _convert_hf(it, n)


def _hf_wider():
    import itertools

    from datasets import load_dataset  # streaming=True never materialises the dataset

    last = None
    for kw in ({"revision": "refs/convert/parquet"}, {}):
        try:
            ds = load_dataset(CFG["wider_hf_id"], split="validation", streaming=True, **kw)
            it = iter(ds)
            first = next(it)  # force the request now so a failure surfaces here, not mid-pipeline
            return itertools.chain([first], it)
        except Exception as e:
            last = e
            log(f"HF load {kw or 'default'} failed: {type(e).__name__}: {str(e)[:120]}")
    raise RuntimeError("no loadable WIDER FACE on the Hub") from last


def _convert_hf(it, n: int):
    emitted = 0
    for ex in it:
        if emitted >= n:
            break
        pil = ex["image"].convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        boxes, attrs = [], []
        faces = ex["faces"]
        invalid = faces.get("invalid", [0] * len(faces["bbox"]))
        for j, (x, y, bw, bh) in enumerate(faces["bbox"]):
            if bw < 16 or bh < 16 or invalid[j]:
                continue  # tiny/invalid faces are excluded from the evaluation protocol
            boxes.append((float(x), float(y), float(x + bw), float(y + bh)))
            attrs.append({k: int(faces[k][j]) for k in ("blur", "illumination", "occlusion") if k in faces})
        if boxes:
            emitted += 1
            yield img, boxes, attrs


def download_wider_val(dest: Path | None = None) -> Path:
    """Fetch WIDER_val (~360 MB) + annotations into the Colab VM disk (never the laptop)."""
    import gdown

    dest = dest or Path(CFG["workdir"]) / "wider"
    if list(dest.rglob("wider_face_val_bbx_gt.txt")) and list(dest.rglob("WIDER_val")):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    val_zip, split_zip = dest / "WIDER_val.zip", dest / "wider_face_split.zip"
    if not val_zip.exists():
        gdown.download(id="0B6eKvaijfFUDd3dIRmpvSk8tLUk", output=str(val_zip), quiet=False)
    if not split_zip.exists():
        urllib.request.urlretrieve("http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip", split_zip)
    for z in (val_zip, split_zip):
        if not zipfile.is_zipfile(z):
            raise RuntimeError(f"{z.name} is not a valid zip (Drive quota page?). Download it manually from the WIDER FACE site and set WIDER_DIR.")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
    return dest


def _wider_from_dir(root: Path, n: int):
    ann = next(root.rglob("wider_face_val_bbx_gt.txt"))
    lines = ann.read_text().splitlines()
    img_root = next(root.rglob("WIDER_val")) / "images"
    i = emitted = 0
    while i < len(lines) and emitted < n:
        name, cnt = lines[i], int(lines[i + 1])
        rows = [list(map(int, l.split())) for l in lines[i + 2 : i + 2 + max(cnt, 1)]]
        i += 2 + max(cnt, 1)
        img = cv2.imread(str(img_root / name))
        boxes = [(r[0], r[1], r[0] + r[2], r[1] + r[3]) for r in rows if r[2] >= 16 and r[3] >= 16 and r[7] == 0]
        attrs = [{"blur": r[4], "illumination": r[6], "occlusion": r[8]} for r in rows if r[2] >= 16 and r[3] >= 16 and r[7] == 0]
        if img is not None and boxes:
            emitted += 1
            yield img, boxes, attrs


def url_seconds_remaining(url: str, now: float | None = None):
    q = parse_qs(urlparse(url).query)
    now = time.time() if now is None else now
    try:
        if "X-Amz-Date" in q and "X-Amz-Expires" in q:
            return calendar.timegm(time.strptime(q["X-Amz-Date"][0], "%Y%m%dT%H%M%SZ")) + int(q["X-Amz-Expires"][0]) - now
        if "Expires" in q:
            return int(q["Expires"][0]) - now
    except (ValueError, IndexError):
        return None
    return None


def stream_ego4d(urls, frames_per_clip: int, stride: int):
    """Decode frames straight from presigned URLs (ffmpeg HTTP range reads). Nothing is written to disk."""
    for u in urls:
        rem = url_seconds_remaining(u)
        if rem is not None and rem <= 0:
            log("skip expired presigned URL (re-sign it on the Data Block platform)")
            continue
        cap = cv2.VideoCapture(u)
        seq, k = [], 0
        while len(seq) < frames_per_clip and cap.isOpened():
            ok, fr = cap.read()
            if not ok:
                break
            if k % stride == 0:
                seq.append(fr)
            k += 1
        cap.release()
        if seq:
            yield u, seq
        else:
            log(f"could not read frames from {urlparse(u).path.rsplit('/', 1)[-1]} (expired link or access denied?)")


def aws_credentials() -> dict:
    """AWS keys from the environment or Colab Secrets (key icon, left bar). Keys are never printed or exported."""
    ak, sk = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not (ak and sk):
        try:
            from google.colab import userdata  # type: ignore

            ak, sk = userdata.get("AWS_ACCESS_KEY_ID"), userdata.get("AWS_SECRET_ACCESS_KEY")
        except Exception as exc:
            raise RuntimeError(
                "No AWS keys: add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in Colab Secrets (key icon) and enable "
                "'Notebook access', or set them as environment variables."
            ) from exc
    return dict(aws_access_key_id=ak, aws_secret_access_key=sk, region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))


def pick_keys(keys, n: int) -> list:
    """Deterministic, evenly spread choice of n items (sorted so the same run picks the same clips)."""
    keys = sorted(k for k in keys if k.lower().endswith(".mp4"))
    if len(keys) <= n:
        return keys
    return keys[:: max(1, len(keys) // n)][:n]


def split_s3_path(path: str):
    """'s3://bucket/some/key.mp4' -> ('bucket', 'some/key.mp4')."""
    m = re.match(r"^s3://([^/]+)/(.+)$", path.strip())
    if not m:
        raise ValueError(f"not an s3:// path: {path[:60]!r}")
    return m.group(1), m.group(2)


def parse_ego_manifest(csv_text: str, n: int, benchmark: str | None = "social") -> list:
    """Pick n (bucket, key) videos from an Ego4D manifest.csv (columns: s3_path | canonical_s3_location, type, benchmarks).

    If `benchmark` matches no row (or the manifest has no such column) all videos are considered instead.
    """
    import csv
    import io

    rows = list(csv.DictReader(io.StringIO(csv_text)))
    videos = []
    for r in rows:
        path = (r.get("canonical_s3_location") or r.get("s3_path") or "").strip()
        kind = (r.get("type") or "").strip().lower()
        if path.lower().endswith(".mp4") and kind in ("", "mp4", "video"):
            videos.append((path, re.sub(r"\s+", "", (r.get("benchmarks") or "").lower())))
    chosen = [p for p, b in videos if benchmark and benchmark.lower() in b] or [p for p, _ in videos]
    return [split_s3_path(p) for p in pick_keys(chosen, n)]


def ego4d_presigned_urls(n: int = 4, expires: int = 6 * 3600, client=None) -> list:
    """Presign `n` Ego4D videos using the manifest the Ego4D CLI itself reads (needs GetObject only, not ListBucket).

    Tries each EGO4D_VERSION x EGO4D_DATASETS until a manifest is readable and yields videos we may fetch.
    Only object names are logged — never the signed URLs.
    """
    if client is None:
        try:
            import boto3
        except ImportError:  # not preinstalled on every Colab image
            subprocess.run([sys.executable, "-m", "pip", "-q", "install", "boto3"], check=True)
            import boto3

        client = boto3.client("s3", **aws_credentials())
    problems = []
    for version in CFG["ego4d_versions"]:
        for dataset in CFG["ego4d_datasets"]:
            key = f"public/{version}/{dataset}/manifest.csv"
            try:
                text = client.get_object(Bucket=CFG["ego4d_bucket"], Key=key)["Body"].read().decode("utf-8", "replace")
            except Exception as exc:
                problems.append(f"{version}/{dataset}: {type(exc).__name__}")
                continue
            picks = parse_ego_manifest(text, n * 3, CFG["ego4d_benchmark"])  # over-pick, then keep what is readable
            usable = []
            for bucket, k in picks:
                try:
                    client.head_object(Bucket=bucket, Key=k)
                    usable.append((bucket, k))
                except Exception as exc:
                    problems.append(f"{bucket}/{k.rsplit('/', 1)[-1]}: {type(exc).__name__}")
                if len(usable) >= n:
                    break
            if usable:
                log(f"Ego4D manifest {version}/{dataset}: presigning {len(usable)} videos:", [k.rsplit("/", 1)[-1] for _, k in usable])
                return [client.generate_presigned_url("get_object", Params={"Bucket": b, "Key": k}, ExpiresIn=expires) for b, k in usable]
    raise RuntimeError("no readable Ego4D video: " + "; ".join(problems[:6]))


def ego_stats(clips: list) -> dict | None:
    """Statistics only (the Ego4D licence forbids redistributing frames).

    `clips`: list of dict(name, wh, v1=[dets per frame], v2=[...], control=[...] | None).
    Reports confidence, detector dropout (frames with no face), how many V1 dropouts V2 fixes, and tracker coasting.
    """
    n = sum(len(c["v1"]) for c in clips)
    if not n:
        return None

    def conf(key):
        return sum(max((d.score for d in f), default=0.0) for c in clips for f in c[key]) / n

    def drop(key):
        return sum(1 for c in clips for f in c[key] if not f) / n

    v1_drop = sum(1 for c in clips for f in c["v1"] if not f)
    fixed = sum(1 for c in clips for a, b in zip(c["v1"], c["v2"]) if not a and b)
    out = dict(
        clips=len(clips),
        frames=n,
        mean_conf_v1=round(conf("v1"), 3),
        mean_conf_v2=round(conf("v2"), 3),
        dropout_v1=round(drop("v1"), 3),
        dropout_v2=round(drop("v2"), 3),
        v1_dropouts_fixed_by_v2=round(fixed / v1_drop, 3) if v1_drop else None,
        tracker_coasted_frames_v1=sum(1 for c in clips for t in track_sequence(c["v1"], c["wh"]) if t["recovered"]),
        tracker_coasted_frames_v2=sum(1 for c in clips for t in track_sequence(c["v2"], c["wh"]) if t["recovered"]),
        per_clip=[
            dict(clip=c["name"], frames=len(c["v1"]), dropout_v1=round(sum(1 for f in c["v1"] if not f) / len(c["v1"]), 3),
                 dropout_v2=round(sum(1 for f in c["v2"] if not f) / len(c["v1"]), 3))
            for c in clips
        ],
    )
    if all(c.get("control") for c in clips):
        out["mean_conf_control"] = round(conf("control"), 3)
        out["dropout_control"] = round(drop("control"), 3)
    return out


# %% [markdown]
# ## 4 · Detector (real `derronqi/yolov7-face` inference with 5 landmarks)


# %%
def letterbox_square(img, size):
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py : py + nh, px : px + nw] = resized
    return canvas, s, px, py


class FaceDetector:
    def __init__(self, weights: str, repo: str, img_size=640, conf=0.25, nms_iou=0.45):
        import torch

        sys.path.insert(0, repo)
        _load = torch.load  # official checkpoints are pickled models; torch>=2.6 defaults to weights_only=True
        torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})
        from models.experimental import attempt_load

        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = attempt_load(weights, map_location=self.dev).eval()
        if self.dev.type == "cuda":
            self.model.half()
        self.size, self.conf, self.nms_iou = img_size, conf, nms_iou
        self.weights = weights

    def predict(self, bgr) -> list[Det]:
        from utils.general import non_max_suppression  # derronqi/yolov7-face: keypoint-aware NMS

        t = self.torch
        sq, s, px, py = letterbox_square(bgr, self.size)
        x = torch_from(sq, self.dev, t, half=self.dev.type == "cuda")
        with t.no_grad():
            pred = self.model(x)[0]
        det = non_max_suppression(pred, self.conf, self.nms_iou, kpt_label=5)[0]
        if det is None or len(det) == 0:
            return []
        return decode_detections(det.float().cpu().numpy(), s, px, py, bgr.shape[1], bgr.shape[0])


def decode_detections(det: np.ndarray, s: float, px: float, py: float, w: int, h: int) -> list[Det]:
    """Rows from yolov7-face NMS: [x1,y1,x2,y2,conf,cls, (x,y,conf) * 5] in letterboxed pixels → image pixels."""
    if det.ndim != 2 or det.shape[1] < 21:
        raise RuntimeError(f"unexpected NMS output {det.shape}; expected 6 + 5*3 columns (kpt_label=5)")
    out: list[Det] = []
    for row in det:
        x1, y1, x2, y2 = (row[0] - px) / s, (row[1] - py) / s, (row[2] - px) / s, (row[3] - py) / s
        lm = np.array([[(row[6 + 3 * i] - px) / s, (row[7 + 3 * i] - py) / s] for i in range(5)])
        lm[:, 0] = lm[:, 0].clip(0, w)
        lm[:, 1] = lm[:, 1].clip(0, h)
        out.append(Det((max(0.0, float(x1)), max(0.0, float(y1)), min(float(w), float(x2)), min(float(h), float(y2))), float(row[4]), lm))
    return out


def torch_from(sq_bgr, dev, torch, half=False):
    x = torch.from_numpy(np.ascontiguousarray(sq_bgr[:, :, ::-1].transpose(2, 0, 1))).to(dev)
    x = x.half() if half else x.float()
    return (x / 255.0).unsqueeze(0)


# %% [markdown]
# ## 5 · Temporal persistence tracker (constant velocity + IoU association)


# %%
class VelocityTracker:
    """Single-target tracker: persists the box via velocity extrapolation while the detector drops out."""

    def __init__(self, iou_gate=0.2, max_age=4, damping=0.7, frame_wh=None):
        """`damping` shrinks the velocity on every coasted frame (a lost face must not keep sliding on inertia);
        the track ends after `max_age` coasted frames or when its centre leaves `frame_wh` (the person left the view)."""
        self.box, self.vel, self.age, self.iou_gate, self.max_age = None, np.zeros(4), 0, iou_gate, max_age
        self.damping, self.frame_wh = damping, frame_wh

    def update(self, dets: list[Det]):
        """Returns (box, recovered_flag) or (None, False)."""
        if self.box is not None:
            pred = tuple(np.array(self.box) + self.vel)
        else:
            pred = None
        pick = None
        if dets:
            if pred is None:
                pick = max(dets, key=lambda d: d.score)
            else:
                cand = max(dets, key=lambda d: iou_xyxy(pred, d.box))
                if iou_xyxy(pred, cand.box) >= self.iou_gate:
                    pick = cand
        if pick is not None:
            nb = np.array(pick.box, float)
            if self.box is not None:
                self.vel = 0.6 * (nb - np.array(self.box)) + 0.4 * self.vel
            self.box, self.age = tuple(nb), 0
            return self.box, False
        if self.box is not None and self.age < self.max_age:
            self.age += 1
            self.box = tuple(np.array(self.box) + self.vel)
            self.vel = self.vel * self.damping
            if self.frame_wh is not None:
                cx, cy = (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2
                if not (0 <= cx <= self.frame_wh[0] and 0 <= cy <= self.frame_wh[1]):
                    self.box = None  # left the frame: terminate instead of chasing a ghost
                    return None, False
            return self.box, True
        self.box = None
        return None, False


def track_sequence(dets_per_frame, frame_wh=None) -> list[dict]:
    tr = VelocityTracker(frame_wh=frame_wh)
    out = []
    for i, ds in enumerate(dets_per_frame):
        box, rec = tr.update(ds)
        out.append(dict(frame=i, detected=bool(ds), tracked=box is not None, recovered=rec, box=box))
    return out


def yolo_label_rows(boxes, lmks, w, h) -> list[str]:
    """derronqi/yolov7-face label rows: `0 cx cy w h` + 5 x (x y visibility) = 20 columns, all in [0, 1].

    The loader rejects negative values, so a missing/out-of-frame landmark is written as `0 0 0`.
    """
    rows, seen = [], set()
    for b, lm in zip(boxes, lmks):
        x1, y1, x2, y2 = max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        vals = [0.0, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]
        for i in range(5):
            if lm is not None and 0 <= lm[i][0] <= w and 0 <= lm[i][1] <= h:
                vals += [lm[i][0] / w, lm[i][1] / h, 2.0]
            else:
                vals += [0.0, 0.0, 0.0]
        is_vis = lambda k: k >= 5 and (k - 5) % 3 == 2  # noqa: E731  (visibility is not a coordinate)
        row = " ".join("0" if k == 0 else f"{v:.6f}" if is_vis(k) else f"{min(1.0, max(0.0, v)):.6f}" for k, v in enumerate(vals))
        if row not in seen:  # the loader asserts there are no duplicate rows
            seen.add(row)
            rows.append(row)
    return rows


# %% [markdown]
# ## 6 · Orchestration: baseline → triage → synthesis → fine-tune → blind gate → export


# %%
def evaluate(detector, items):
    """items: list of (img, gt boxes). Returns (AP, dets per image)."""
    dets = [detector.predict(img) for img, _ in items]
    return average_precision(dets, [g for _, g in items], CFG["hard_iou"]), dets


def bootstrap_diff_ci(dets_a, dets_b, gts, n_boot: int = 300, seed: int = 0, iou_thr: float = 0.5, alpha: float = 0.05) -> dict:
    """Paired bootstrap over images of AP(b) - AP(a): mean and (1 - alpha) percentile interval, in AP units."""
    rng = np.random.default_rng(seed)
    n = len(gts)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        g = [gts[i] for i in idx]
        diffs.append(
            average_precision([dets_b[i] for i in idx], g, iou_thr) - average_precision([dets_a[i] for i in idx], g, iou_thr)
        )
    d = np.asarray(diffs)
    return dict(
        mean=round(float(d.mean()), 4),
        lo=round(float(np.percentile(d, 100 * alpha / 2)), 4),
        hi=round(float(np.percentile(d, 100 * (1 - alpha / 2))), 4),
        n_boot=n_boot,
        n_images=n,
    )


def files_signature(files) -> str:
    """Stable fingerprint of a training set (names + sizes) so a stale checkpoint is never silently reused."""
    h = hashlib.sha256()
    for p in sorted(Path(f) for f in files):
        h.update(f"{p.name}:{p.stat().st_size}\n".encode())
    return h.hexdigest()


def finetune(repo: str, weights: str, data_yaml, name: str, data_sig: str | None = None) -> Path:
    """Run yolov7-face's own trainer (keypoint loss on) and return the resulting checkpoint.

    If a checkpoint with the same recipe (epochs, batch, size, seed, training-set fingerprint) already
    exists it is reused, so a crash after training costs minutes instead of a re-train. A checkpoint without a stamp is
    reused only when REUSE_CHECKPOINTS=1 (you assert the configuration is unchanged).
    """
    wdir = Path(repo, "runs", "train", name, "weights")
    stamp = Path(repo, "runs", "train", name, "advertest_stamp.json")
    want = dict(epochs=CFG["epochs"], batch=CFG["batch"], img=CFG["img_size"], seed=CFG["seed"], data=data_sig)
    existing = next((c for c in (wdir / "best.pt", wdir / "last.pt") if c.exists()), None)
    if existing is not None:
        have = json.loads(stamp.read_text()) if stamp.exists() else None
        if have is not None:
            have.pop("version", None)  # older stamps recorded the pipeline version; it does not change the recipe
        if have == want:
            log(f"reusing the matching checkpoint for {name} (same recipe and data)")
            return existing
        if have is None and os.environ.get("REUSE_CHECKPOINTS") == "1":
            log(f"REUSE_CHECKPOINTS=1: reusing the unstamped checkpoint for {name}; make sure the configuration is unchanged")
            return existing
    cfg_yaml, hyp_yaml = Path(repo, "cfg", "yolov7-tiny-face.yaml"), Path(repo, "data", "hyp.scratch.tiny.yaml")
    for need in (cfg_yaml, hyp_yaml):
        assert need.exists(), f"{need} missing; repo layout changed: {[p.name for p in need.parent.glob('*')]}"
    cmd = [
        sys.executable, "-c", NUMPY_COMPAT_SHIM, "train.py", "--data", str(data_yaml), "--cfg", str(cfg_yaml), "--hyp", str(hyp_yaml),
        "--weights", weights, "--epochs", str(CFG["epochs"]), "--batch-size", str(CFG["batch"]),
        "--img-size", str(CFG["img_size"]), str(CFG["img_size"]), "--kpt-label", "5",
        "--name", name, "--workers", "2", "--exist-ok", "--notest",
    ]
    # The repo's train.py runs check_requirements() against requirements.txt, which pins versions that no longer
    # exist for Colab's Python (e.g. onnxruntime==1.10.0). Dependencies are preinstalled, so neutralise the pins
    # in the Colab clone only. Official checkpoints are pickled models, so allow full unpickling for the trainer.
    Path(repo, "requirements.txt").write_text("# pins removed on the Colab worker (deps preinstalled)\n")
    # W&B would demand an API key; experiment tracking is not needed on the worker.
    run_logged(cmd, cwd=repo, env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "WANDB_MODE": "disabled", "WANDB_DISABLED": "true"})
    ckpt = next((c for c in (wdir / "best.pt", wdir / "last.pt") if c.exists()), None)  # --notest may skip best.pt
    assert ckpt is not None, f"trainer produced no checkpoint in {wdir}: {[p.name for p in wdir.glob('*')]}"
    stamp.write_text(json.dumps(want))
    return ckpt


def to_ref(box, w, h):
    """Map an image-pixel xyxy box to [x,y,w,h] in the UI's 160x120 letterboxed reference frame."""
    s = min(320 / w, 240 / h)
    px, py = (320 - w * s) / 2, (240 - h * s) / 2
    x1, y1, x2, y2 = [(box[0] * s + px) / 2, (box[1] * s + py) / 2, (box[2] * s + px) / 2, (box[3] * s + py) / 2]
    return [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]


def crop_around(img, box, factor: float = 5.0, min_side: int = 96):
    """Crop a square-ish window centred on `box` (clipped to the image). Returns (crop, x0, y0)."""
    h, w = img.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    side = max(min_side, factor * max(bw, bh))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0, y0 = int(max(0, min(w - 1, cx - side / 2))), int(max(0, min(h - 1, cy - side / 2 * 0.75)))
    x1, y1 = int(min(w, x0 + side)), int(min(h, y0 + side * 0.75))
    return img[y0:y1, x0:x1], x0, y0


def shift_box(box, x0, y0):
    return (box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0)


def lmk_to_ref(lm, w, h):
    s = min(320 / w, 240 / h)
    px, py = (320 - w * s) / 2, (240 - h * s) / 2
    return [[round((float(x) * s + px) / 2, 1), round((float(y) * s + py) / 2, 1)] for x, y in lm]


def thumb_data_uri(img) -> str:
    h, w = img.shape[:2]
    s = min(320 / w, 240 / h)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    canvas = np.full((240, 320, 3), 0, np.uint8)
    py, px = (240 - r.shape[0]) // 2, (320 - r.shape[1]) // 2
    canvas[py : py + r.shape[0], px : px + r.shape[1]] = r
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_weights(src, workdir, accepted: bool, drive_dir: str = "advertest", put_url: str | None = None,
                   drive_root: str = "/content/drive/MyDrive") -> dict:
    """Hand the V2 checkpoint over without ever touching the laptop.

    Copies it to the user's Google Drive when running in Colab (and/or PUTs it to a presigned URL from
    WEIGHTS_PUT_URL) and returns {name, bytes, sha256, accepted_by_gate, location} for the manifest.
    Any failure is recorded in `location` and never raised: the metrics and artifacts must not be lost over a mount.
    """
    src = Path(src)
    info = dict(
        name="yolov7-face-v2.pt",
        bytes=src.stat().st_size,
        sha256=sha256_file(src),
        accepted_by_gate=bool(accepted),
        location=f"worker-only (ephemeral): {src}",
    )
    put_url = put_url or os.environ.get("WEIGHTS_PUT_URL", "")
    try:
        if put_url:
            urllib.request.urlopen(urllib.request.Request(put_url, data=src.read_bytes(), method="PUT"), timeout=120).read()
            info["location"] = "uploaded via presigned PUT (WEIGHTS_PUT_URL)"
        elif "google.colab" in sys.modules:
            root = Path(drive_root)
            if not root.exists():  # skip the interactive mount when Drive is already attached
                from google.colab import drive  # type: ignore

                drive.mount(str(root.parent))
            dest = root / drive_dir
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / info["name"])
            info["location"] = f"Google Drive: MyDrive/{drive_dir}/{info['name']}"
    except Exception as exc:
        info["location"] = (
            f"worker-only (copy failed: {type(exc).__name__}); copy {src} manually before the session ends "
            "(mount Drive in a cell first, then run: !cp <path> /content/drive/MyDrive/advertest/yolov7-face-v2.pt)"
        )
        log("weights export failed, continuing:", type(exc).__name__, str(exc)[:120])
    log("weights:", info["location"], info["sha256"][:12])
    return info


def blind_gate(base_v1, base_v2, corr_v1, corr_v2, max_drop, min_gain) -> dict:
    drop, gain = base_v1 - base_v2, corr_v2 - corr_v1
    return dict(
        max_base_drop=max_drop,
        min_corrupted_gain=min_gain,
        base_drop=round(drop, 4),
        corrupted_gain=round(gain, 4),
        accepted=bool(drop <= max_drop + 1e-9 and gain >= min_gain),
    )


def best_conf_for(gt_box, dets):
    iou, sc = (match_gt([gt_box], dets) or [(0.0, 0.0)])[0]
    best = max(dets, key=lambda d: iou_xyxy(gt_box, d.box), default=None)
    return iou, sc, best


def run_pipeline() -> Path:
    log(f"AdverTest pipeline version {PIPELINE_VERSION} (if this is not the latest, reopen the notebook from GitHub)")
    random.seed(CFG["seed"])
    wd = Path(CFG["workdir"])
    repo, weights = str(wd / "yolov7_face"), str(wd / "yolov7-tiny-face.pt")
    out_dir = wd / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    log("streaming WIDER FACE micro-subset ...")
    pool = list(stream_wider(CFG["n_pool"]))
    assert len(pool) >= CFG["n_blind"] + 20, f"only {len(pool)} images streamed"
    blind, train_all = pool[: CFG["n_blind"]], pool[CFG["n_blind"] :]
    # Leakage guard: WIDER images of one event can be near-identical; drop any train image that duplicates a blind one.
    leaks = find_leaks([t[0] for t in train_all], [b[0] for b in blind])
    train_idx = [i for i in range(len(train_all)) if i not in leaks]  # position in train_all (source_index = n_blind + i)
    train = [train_all[i] for i in train_idx]
    log(f"{len(train)} train / {len(blind)} blind images; {len(leaks)} near-duplicate train images removed")

    v1 = FaceDetector(weights, repo, CFG["img_size"], CFG["det_conf"])
    v1_sha = sha256_file(weights)

    # ---- baseline + triage on the TRAIN split (clean and physically corrupted) -------------------
    clean_dets = [v1.predict(im) for im, _, _ in train]
    clean_triage = [triage_image(g, d, CFG["hard_iou"], CFG["hard_conf"]) for (_, g, _), d in zip(train, clean_dets)]
    log(f"clean hard negatives: {sum(t['hard'] for t in clean_triage)}/{len(train)}")

    # Corrupt every train image with a random tag to expose the egocentric domain gap, then triage again.
    insight_rows, corrupted_train = [], []
    for k, ((img, gts, attrs), dets) in enumerate(zip(train, clean_dets)):
        lm = []
        for g in gts:
            iou, sc, best = best_conf_for(g, dets)
            lm.append(best.lmk if best is not None and iou >= 0.5 and sc >= CFG["hard_conf"] else None)
        tag, sev = draw_attack(CFG["seed"], ("corrupt", k))
        att = apply_attack(img, gts, lm, CFG["seed"], ("corrupt", k), tag, sev)
        if att.discarded:
            continue
        d2 = v1.predict(att.image)
        t2 = triage_image(att.boxes, d2, CFG["hard_iou"], CFG["hard_conf"])
        corrupted_train.append((k, tag, att, t2))
        if t2["hard"]:
            insight_rows.append((k, tag, att, t2))
    n_hard = len(insight_rows)
    assert n_hard > 0, "baseline never fails on the corrupted split — raise severity"
    counts = {t: sum(1 for r in insight_rows if r[1] == t) for t in CANONICAL_TAGS}
    dist = {t: counts[t] / n_hard for t in CANONICAL_TAGS}
    med_conf = float(np.median([r[3]["worst_conf"] for r in insight_rows]))
    log("failure distribution:", {t: f"{v * 100:.1f}%" for t, v in dist.items()})

    # ---- targeted synthesis: sample attack tags in proportion to the measured failure distribution --
    ds = wd / "ds"
    for sp in ("train", "val"):
        (ds / "images" / sp).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / sp).mkdir(parents=True, exist_ok=True)
    audit, n_written = [], 0

    def write_sample(name, split, image, boxes, lmks):
        h, w = image.shape[:2]
        cv2.imwrite(str(ds / "images" / split / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (ds / "labels" / split / f"{name}.txt").write_text("\n".join(yolo_label_rows(boxes, lmks, w, h)))

    # clean replay protects the base domain (catastrophic-forgetting guard)
    for k, ((img, gts, _), dets) in enumerate(zip(train, clean_dets)):
        lm = []
        for g in gts:
            iou, sc, best = best_conf_for(g, dets)
            lm.append(best.lmk if best is not None and iou >= 0.5 and sc >= CFG["hard_conf"] else None)
        write_sample(f"clean_{k:04d}", "train", img, gts, lm)
        n_written += 1
    hard_ids = [r[0] for r in insight_rows]
    tags_w = [max(dist[t], 1e-3) for t in CANONICAL_TAGS]
    for j in range(2 * n_hard):
        k = random.choice(hard_ids)
        tag, sev = draw_attack(CFG["seed"], ("adv", j), weights=tags_w)
        img, gts, _ = train[k]
        lm = []
        for g in gts:
            iou, sc, best = best_conf_for(g, clean_dets[k])
            lm.append(best.lmk if best is not None and iou >= 0.5 and sc >= CFG["hard_conf"] else None)
        att = apply_attack(img, gts, lm, CFG["seed"], ("adv", j), tag, sev)
        if att.discarded:
            continue
        write_sample(f"adv_{j:04d}", "train", att.image, att.boxes, att.lmk)
        audit.append(
            dict(
                id=f"wider_adv_{j:04d}",
                image_url="",
                iou=round(float(insight_rows[hard_ids.index(k)][3]["worst_iou"]), 2),
                confidence=round(float(insight_rows[hard_ids.index(k)][3]["worst_conf"]), 2),
                audit_trail=dict(
                    insight_source=tag,
                    applied_chain=att.chain,
                    severity_level=att.severity,
                    justification=explain(tag, dist, n_hard, med_conf, att),
                    parameters=att.params,
                ),
                reproduce=dict(
                    seed=CFG["seed"], key=["adv", j], tag=tag, severity=att.severity,
                    source=dict(split="wider_val_stream", index=CFG["n_blind"] + train_idx[k]),
                    v1_sha256=v1_sha, pipeline_version=PIPELINE_VERSION,
                ),
            )
        )
    # small validation split for the trainer
    for k, (img, gts, _) in enumerate(blind[:8]):
        write_sample(f"val_{k:02d}", "val", img, gts, [None] * len(gts))
    (wd / "advertest.yaml").write_text(
        f"train: {ds / 'images' / 'train'}\nval: {ds / 'images' / 'val'}\nnc: 1\nnames: ['face']\n"
    )

    # ---- fine-tune V2 (clean replay + targeted adversarial samples) ---------------------------------
    log(f"fine-tuning V2: {CFG['epochs']} epochs on {n_written + len(audit)} images ...")
    v2_w = finetune(repo, weights, wd / "advertest.yaml", "advertest_v2", files_signature((ds / "images" / "train").glob("*.jpg")))
    v2 = FaceDetector(str(v2_w), repo, CFG["img_size"], CFG["det_conf"])

    # ---- control: same init, same epochs, same clean images, but NO adversarial samples ---------------
    # If V2 only beats V1 because it saw more WIDER data, the control improves just as much.
    ctrl = None
    if CFG["run_control"]:
        clean_list = sorted((ds / "images" / "train").glob("clean_*.jpg"))
        (wd / "train_clean.txt").write_text("\n".join(str(p) for p in clean_list))
        (wd / "advertest_clean.yaml").write_text(
            f"train: {wd / 'train_clean.txt'}\nval: {ds / 'images' / 'val'}\nnc: 1\nnames: ['face']\n"
        )
        log(f"fine-tuning clean-only control: {CFG['epochs']} epochs on {len(clean_list)} images ...")
        ctrl_w = finetune(repo, weights, wd / "advertest_clean.yaml", "advertest_ctrl", files_signature(clean_list))
        ctrl = FaceDetector(str(ctrl_w), repo, CFG["img_size"], CFG["det_conf"])

    # ---- blind evaluation: images never used above -------------------------------------------------
    log("blind evaluation ...")
    clean_items = [(im, g) for im, g, _ in blind]
    base1, bdets1 = evaluate(v1, clean_items)
    base2, bdets2 = evaluate(v2, clean_items)
    blind_corr, blind_skipped = [], []
    for im, g, attrs in blind:
        key = ("blind", len(blind_corr) + len(blind_skipped))
        tag, sev = draw_attack(CFG["seed"], key)
        att = apply_attack(im, g, [None] * len(g), CFG["seed"], key, tag, sev)
        if att.discarded:
            blind_skipped.append(key)
        else:
            blind_corr.append((tag, im, att))
    corr_items = [(a.image, a.boxes) for _, _, a in blind_corr]
    corr1, cdets1 = evaluate(v1, corr_items)
    corr2, cdets2 = evaluate(v2, corr_items)
    gate = blind_gate(base1, base2, corr1, corr2, CFG["max_base_drop"], CFG["min_corrupted_gain"])
    gts_clean, gts_corr = [g for _, g in clean_items], [g for _, g in corr_items]
    base3 = corr3 = None
    stats = dict(
        corrupted=dict(v2_minus_v1=bootstrap_diff_ci(cdets1, cdets2, gts_corr, CFG["n_boot"], CFG["seed"])),
        clean=dict(v2_minus_v1=bootstrap_diff_ci(bdets1, bdets2, gts_clean, CFG["n_boot"], CFG["seed"])),
    )
    control_report = None
    if ctrl is not None:
        base3, bdets3 = evaluate(ctrl, clean_items)
        corr3, cdets3 = evaluate(ctrl, corr_items)
        stats["corrupted"]["v2_minus_control"] = bootstrap_diff_ci(cdets3, cdets2, gts_corr, CFG["n_boot"], CFG["seed"])
        stats["clean"]["v2_minus_control"] = bootstrap_diff_ci(bdets3, bdets2, gts_clean, CFG["n_boot"], CFG["seed"])
        control_report = dict(base_map=round(base3, 4), corrupted_map=round(corr3, 4),
                              note="same init, epochs and clean images as V2, without the adversarial samples")
        log(f"control: base {base3:.3f}  corrupted {corr3:.3f}  (V2 corrupted {corr2:.3f})")
    ci = stats["corrupted"]["v2_minus_v1"]
    gate["corrupted_gain_ci95"] = [ci["lo"], ci["hi"]]
    gate["gain_statistically_supported"] = bool(ci["lo"] > 0)
    if ctrl is not None:
        gate["beats_clean_only_control"] = bool(stats["corrupted"]["v2_minus_control"]["lo"] > 0)

    # Sensitivity to the specific random corruption: repeat the corrupted blind set with fresh draws.
    repeats = [dict(rep=0, v2_minus_v1=round(corr2 - corr1, 4), **({"v2_minus_control": round(corr2 - corr3, 4)} if ctrl else {}))]
    for rep_i in range(1, CFG["corruption_repeats"]):
        items = []
        for idx, (im, g, _) in enumerate(blind):
            key = ("blind", rep_i, idx)
            t_, sv_ = draw_attack(CFG["seed"], key)
            a_ = apply_attack(im, g, [None] * len(g), CFG["seed"], key, t_, sv_)
            if not a_.discarded:
                items.append((a_.image, a_.boxes))
        m1, _ = evaluate(v1, items)
        m2, _ = evaluate(v2, items)
        row = dict(rep=rep_i, v2_minus_v1=round(m2 - m1, 4))
        if ctrl is not None:
            m3, _ = evaluate(ctrl, items)
            row["v2_minus_control"] = round(m2 - m3, 4)
        repeats.append(row)
    gains = [r["v2_minus_v1"] for r in repeats]
    repeats_summary = dict(draws=repeats, v2_minus_v1_mean=round(float(np.mean(gains)), 4), v2_minus_v1_std=round(float(np.std(gains)), 4))
    log("repeated corruption draws:", repeats_summary)
    fppi = dict(
        clean=dict(v1=recall_at_fppi(bdets1, [g for _, g in clean_items]), v2=recall_at_fppi(bdets2, [g for _, g in clean_items])),
        corrupted=dict(v1=recall_at_fppi(cdets1, [g for _, g in corr_items]), v2=recall_at_fppi(cdets2, [g for _, g in corr_items])),
    )
    log("recall @ fixed FPPI:", fppi)
    log("gate:", gate, f"base {base1:.3f}->{base2:.3f}  corrupted {corr1:.3f}->{corr2:.3f}")

    # hard-negative recovery on the blind corrupted set
    rec_den = rec_num = 0
    cases = []
    for (tag, im, att), d1, d2 in zip(blind_corr, cdets1, cdets2):
        for gb in att.boxes:
            i1, c1, b1 = best_conf_for(gb, d1)
            i2, c2, b2 = best_conf_for(gb, d2)
            if i1 < CFG["hard_iou"] or c1 < CFG["hard_conf"]:
                rec_den += 1
                if i2 >= CFG["hard_iou"] and c2 >= CFG["hard_conf"]:
                    rec_num += 1
                    cases.append((c2 - c1, tag, att, gb, (i1, c1, b1), (i2, c2, b2)))
    recovery = rec_num / rec_den if rec_den else 0.0

    # ---- real Ego4D frames (stream only) --------------------------------------------------------------
    ego, ego_error = None, None
    try:
        ego_urls = CFG["ego4d_urls"] or (ego4d_presigned_urls(CFG["ego4d_clips"]) if CFG["ego4d_auto"] else [])
        if ego_urls:
            clips = []
            for url, seq in stream_ego4d(ego_urls, CFG["ego4d_frames"], CFG["ego4d_stride"]):
                clips.append(dict(
                    name=urlparse(url).path.rsplit("/", 1)[-1],  # object key only; the signed query string is dropped
                    wh=(seq[0].shape[1], seq[0].shape[0]),
                    v1=[v1.predict(f) for f in seq],
                    v2=[v2.predict(f) for f in seq],
                    control=[ctrl.predict(f) for f in seq] if ctrl is not None else None,
                ))
            ego = ego_stats(clips)
            if ego:
                log("ego4d (real, streamed, statistics only):", {k: v for k, v in ego.items() if k != "per_clip"})
    except Exception as exc:  # Ego4D is optional: never lose the trained models and metrics because of it
        ego_error = re.sub(r"\?\S*", "?<redacted>", f"{type(exc).__name__}: {exc}")[:300]  # strip any signed query string
        log("Ego4D stage failed; the rest of the run is kept:", ego_error)

    # ---- export: small JSON + thumbnails only ---------------------------------------------------------------
    cases.sort(key=lambda c: -c[0])
    seen, poc = set(), []
    for gain, tag, att, gb, (i1, c1, b1), (i2, c2, b2) in cases:
        if tag in seen or len(poc) >= 3:
            continue
        seen.add(tag)
        # Zoom on the face: WIDER faces are tiny in crowd scenes, so crop a window around the ground-truth box.
        crop, x0, y0 = crop_around(att.image, gb)
        h, w = crop.shape[:2]
        sh = lambda b: shift_box(b, x0, y0)  # noqa: E731
        v1_hit = b1 is not None and i1 >= 0.1  # a detection on a *different* face is not a V1 result for this face
        poc.append(dict(
            case_id=f"eval_wider_{len(poc) + 1:03d}",
            condition={"error_fisheye": "Chest-Mount Lens Distortion", "error_motion_blur": "Walking Head-Bob Blur",
                       "error_sensor_noise": "Extreme Exposure / ISO", "error_occlusion": "Hand Occlusion"}[tag],
            failure_tag=tag,
            delta=f"+{c2 - c1:.2f} Confidence Gain",
            image=thumb_data_uri(crop),
            model_v1=dict(status="Failed" if c1 < CFG["hard_conf"] else "Missed", confidence=round(c1, 2),
                          bbox=to_ref(sh(b1.box if v1_hit else gb), w, h),
                          landmarks=lmk_to_ref(np.asarray(b1.lmk) - [x0, y0], w, h) if (v1_hit and b1.lmk is not None) else None),
            model_v2=dict(status="Success", confidence=round(c2, 2), bbox=to_ref(sh(b2.box), w, h),
                          landmarks=lmk_to_ref(np.asarray(b2.lmk) - [x0, y0], w, h) if b2.lmk is not None else None),
        ))
    report = dict(
        base_model=dict(base_map=round(base1, 4), corrupted_map=round(corr1, 4)),
        advertest_model=dict(base_map=round(base2, 4), corrupted_map=round(corr2, 4)),
        deltas=dict(base_drop=gate["base_drop"], corrupted_improvement=gate["corrupted_gain"]),
        hard_negative_recovery=round(recovery, 4),
        recall_at_fppi=fppi,
        control_model=control_report,
        stats=stats,
        corruption_repeats=repeats_summary,
        gate=gate,
        provenance=dict(
            source="colab_zero_disk_pipeline", model="derronqi/yolov7-face (yolov7-tiny-face)",
            base_dataset="WIDER FACE (validation, streamed)", n_train=len(train), n_blind=len(blind),
            target_domain="WIDER FACE through the physical attack engine (proxy)" + ("; plus real Ego4D frames, statistics only" if ego else "; no Ego4D supplied"),
            ego4d_real=ego, ego4d_error=ego_error, seed=CFG["seed"], epochs=CFG["epochs"], pipeline_version=PIPELINE_VERSION,
            v1_sha256=v1_sha, versions=env_versions(), leaks_removed=len(leaks),
            split="image-level, near-duplicate guarded (dHash); blind images are never used for synthesis or training", generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
    )
    tri = sorted(
        [dict(id=f"wider_{k:04d}", iou=round(t["worst_iou"], 2), confidence=round(t["worst_conf"], 2),
              audit_trail=dict(insight_source=tag, applied_chain=att.chain, severity_level=att.severity,
                               justification=explain(tag, dist, n_hard, med_conf, att), parameters=att.params),
              reproduce=dict(seed=CFG["seed"], key=["corrupt", k], tag=tag, severity=att.severity,
                             source=dict(split="wider_val_stream", index=CFG["n_blind"] + train_idx[k]),
                             v1_sha256=v1_sha, pipeline_version=PIPELINE_VERSION),
              image_url="")
         for k, tag, att, t in insight_rows],
        key=lambda r: r["iou"],
    )
    # audit_trail.json = the hard negatives (what the UI triages) followed by synthesized samples
    audit_out = tri + audit[:200]
    weights_info = export_weights(v2_w, wd, report["gate"]["accepted"])
    manifest = dict(files=["audit_trail.json", "proof_of_cure.json", "eval_report.json", "manifest.json"],
                    failure_distribution={t: counts[t] for t in CANONICAL_TAGS}, hard_negatives=n_hard,
                    synthesized=len(audit), schema="ENTERPRISE_PII_V2",
                    weights=weights_info, note="weights are never bundled in the zip; see weights.location")
    for name, obj in [("audit_trail.json", audit_out), ("proof_of_cure.json", poc), ("eval_report.json", report), ("manifest.json", manifest)]:
        (out_dir / name).write_text(json.dumps(obj, indent=1))
    zpath = wd / "advertest_artifacts.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.glob("*.json"):
            z.write(p, p.name)
    log(f"exported {zpath} ({zpath.stat().st_size / 1024:.0f} KB)")
    if CFG["artifact_put_url"]:
        req = urllib.request.Request(CFG["artifact_put_url"], data=zpath.read_bytes(), method="PUT")
        urllib.request.urlopen(req, timeout=60).read()
        log("uploaded artifact bundle to the presigned destination")
    return zpath


# %% [markdown]
# ## 7 · Run

# %%
IN_COLAB = "google.colab" in sys.modules
if __name__ == "__main__" and (IN_COLAB or os.environ.get("ADVERTEST_RUN") == "1"):
    setup_environment()
    zip_path = run_pipeline()
    if IN_COLAB:
        from google.colab import files  # type: ignore

        files.download(str(zip_path))
