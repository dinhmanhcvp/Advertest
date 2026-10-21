# AdverTest — Handover Notes

Last updated: 2026-10-19. This document says what exists, what was **measured**, what is a **proxy**, and what is **still pending**.

## 1. What this is

An end-to-end data engine for egocentric face detection (`derronqi/yolov7-face`: box + 5 landmarks).
It diagnoses where a baseline fails, synthesizes those failures with an auditable attack engine, retrains, and accepts the new
model only if it passes a blind gate (base mAP drop ≤ 1 pt **and** corrupted mAP gain ≥ 5 pts).

| Piece | Where |
|---|---|
| GPU pipeline (the real work) | `colab/colab_zero_disk_pipeline.py` / `.ipynb` — Colab only, nothing heavy on the laptop |
| Measured results (served by the UI) | `data/demo_artifacts/*.json` (copied to `frontend/public/demo/`, a test keeps them identical) |
| Web console (Next.js, 4 stages) | `frontend/` — deployed on Vercel |
| API (FastAPI) | `backend/app/` — `GET /api/v1/demo/audit-trail`, `/demo/proof-of-cure`, `/eval-report`, `POST /api/v1/triage/analyze`, `/attack/generate` |
| Canonical failure taxonomy | `advertest/core/taxonomy.py` (`error_fisheye`, `error_motion_blur`, `error_sensor_noise`, `error_occlusion`) |
| Presigned-URL ingestion | `src/services/data_connector.py` (TTL check + one re-sign, retries, cache purge) |
| Documents | `docs/advertest_executive_report.html` (Ctrl+P → PDF), `docs/beautiful_architecture.html`, `docs/MENTOR_QA.md` (hard questions, answers, evidence, gaps), this file |

## 2. Measured results (one run)

WIDER FACE validation, streamed: 160 images for training, **80 held-out images** for the blind evaluation, yolov7-tiny-face, 30 epochs, seed 7.

| | V1 | V2 | Change |
|---|---|---|---|
| Base mAP (clean) | 71.1% | 73.5% | +2.4 pts (gate: drop ≤ 1 pt → pass) |
| Corrupted mAP (proxy) | 55.0% | 60.7% | +5.8 pts (gate: gain ≥ 5 pts → pass) |
| Hard negatives recovered | | 16.5% | |

91 of 160 images were baseline hard negatives (fisheye 24, motion blur 22, sensor/ISO 21, occlusion 24).

## 3. Honest limitations

1. **Target domain is a proxy, not Ego4D.** Ego4D needs a licence and AWS keys (pending). "Corrupted" = WIDER FACE through the physical attack engine, which has exact ground truth. The report and UI label it as such (`provenance.target_domain`).
2. **Thin evidence.** 80 blind images, one seed; the gain clears the gate by < 1 pt and has no confidence interval.
3. **Low recovery.** 16.5% of baseline hard negatives are recovered.
4. **Landmarks are pseudo-labels** (WIDER FACE has boxes only), taken from the baseline and transformed with the image geometry.
5. **The Studio view draws synthetic frames** (procedural SVG) to show the physics controls; only Stage 4 shows real images (face crops from the measured run).
6. **Per-epoch validation inside the repo trainer is disabled** (its keypoint logging crashes); all accuracy numbers come from the pipeline's own blind evaluation.
7. **Not exercised in the measured run:** Kafka streaming, Ray training, 3D Gaussian Splatting, RL policy, Label Studio round-trip (code and unit tests exist; see `SYSTEM_ARCHITECTURE.md`).

## 4. Reproduce the measured run (Colab, no local disk)

1. Colab → File → Open notebook → GitHub → `dinhmanhcvp/Advertest` → `colab/colab_zero_disk_pipeline.ipynb`; Runtime → T4 GPU.
2. Optional env before running: `EPOCHS`, `N_POOL` (default 400 images streamed), `N_BLIND` (default 150 held-out), `RUN_CONTROL` (1 = also train the clean-only control), `N_BOOT` (bootstrap resamples), `CORRUPTION_REPEATS` (independent corruption draws of the blind set), `WIDER_DIR` (if Hugging Face streaming is unavailable the notebook downloads the official WIDER archive to the VM disk), `EGO4D_PRESIGNED_URLS` (comma-separated), `ARTIFACT_PUT_URL`, `WEIGHTS_PUT_URL`.
3. Run all (estimated 30–60 min with the control model; set `RUN_CONTROL=0` and `EPOCHS=10` for a quick check). Download `advertest_artifacts.zip` (< 1 MB).
4. Unzip into `data/demo_artifacts/`, copy the three JSON files to `frontend/public/demo/`, commit, push → Vercel redeploys.
5. The V2 checkpoint is copied to Google Drive (`MyDrive/advertest/yolov7-face-v2.pt`) or PUT to `WEIGHTS_PUT_URL`; its SHA-256 is in `manifest.json` → `weights`.

### Adding real Ego4D frames (keys stay in Colab)
Put `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (from the Ego4D licence email) in **Colab Secrets** (key icon, enable *Notebook access*) — never in a cell, the repo or chat. Then set `EGO4D_AUTO_PRESIGN=1` (optional: `EGO4D_CLIPS`, `EGO4D_PREFIX`, `EGO4D_BUCKET`); the pipeline lists the clips, presigns a few for 6 h, streams them and logs only object names. Licence note: Ego4D frames must not be redistributed, so only statistics are exported (confidence, detector dropout, dropouts fixed by V2, tracker coasting) — no thumbnails.

Manual alternative:
Request access at ego4d-data.org, then create short-lived URLs for a few clips (`aws s3 presign … --expires-in 7200`) and set `EGO4D_PRESIGNED_URLS`. The pipeline decodes them as streams (never saved) and reports confidence, detector dropout and tracker-recovered frames for V1/V2. It does **not** compute mAP on Ego4D (no box labels in that path).

## 5. Run locally

```bash
# API in demo mode
ENABLE_DEMO_MODE=True uvicorn backend.app.main:app --port 8000
# UI
cd frontend && npm ci && API_PROXY_TARGET=http://localhost:8000 npm run dev   # http://localhost:3000
```
Without an API the UI reads `frontend/public/demo/*.json` (nav shows `ARTIFACTS: MEASURED RUN`); with neither it falls back to a built-in placeholder store (`DATA BLOCK: LOCAL CACHE` — not real data).
Docker: `docker compose up --build` (see `.env.example`).

## 6. Deploy settings

- **Vercel:** Root Directory `frontend`. Optional `API_PROXY_TARGET` = backend URL; not required for the static artifacts.
- **Render (API):** uses `Dockerfile.api`; set `ENABLE_DEMO_MODE=True` (no weights on the server).
- **GitHub Actions:** `ci.yml` (ruff, plugin catalog, pytest, frontend `npm ci`/test/lint/build). `production.yml` AWS build/deploy jobs are opt-in: set the repo variable `ENABLE_AWS_DEPLOY=true` and AWS secrets.

## 7. Compliance notes

- No company names in code, docs or UI; generic terms only ("Enterprise Data Platform", "Data Block", "Core Data Lake").
- No datasets, weights or large binaries in git (`tests/test_repo_footprint.py`: < 5 MB, no `.pt/.zip/.jpg/...`).
- Secrets belong in Colab/Vercel/Render settings, never in the repo. Rotate any token that was ever embedded in a git remote URL.

## 8. Acceptance checklist

| Item | Status |
|---|---|
| Pipeline runs end to end on Colab (stream → triage → attacks → fine-tune → blind gate → export) | Done (one run) |
| Failure taxonomy + audit trail per sample | Done |
| Blind gate (≤ 1 pt base drop, ≥ 5 pts corrupted gain) | Done, passed narrowly |
| 5-point landmarks in A/B evidence | Done (V2); V1 shown as "no detection" when it finds nothing |
| Temporal tracker | Implemented + unit-tested; **not evaluated on real sequences** |
| UI: 4 stages, runs without any backend | Done |
| Ego4D real-data validation | **Pending** (licence keys) |
| V2 weights saved with a verifiable hash | Done: `MyDrive/advertest/yolov7-face-v2.pt`, 15.99 MB, SHA-256 `8c6461c2fd9e9a7320fa75df26e632129ab316be1cce6b804a5709f97b6ec133` |
| Weights transferred to the Core Data Lake | **Pending** (copy from Drive, or rerun with `WEIGHTS_PUT_URL`) |
| Statistical confidence (more seeds / larger blind set) | **Pending** |
| Per-sample reproducibility record, leak guard, Recall@FPPI | Implemented and unit-tested; **artifacts need one Colab rerun to contain them** |
| Clean-only control model, 95% bootstrap CI, repeated corruption draws, larger blind set (150) | Implemented; the whole pipeline is covered by an offline dry-run test (fake detector); **needs one Colab rerun to produce numbers** — the UI then shows a "Validity checks" panel |
| Official WIDER Easy/Medium/Hard, event-level split, training-seed variance | **Pending** (see `docs/MENTOR_QA.md`) |
