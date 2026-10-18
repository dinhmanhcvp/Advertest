<div align="center">
  <h1>🛡️ AdverTest</h1>
  <p><b>Decoupled Egocentric PII Engine via Active Learning</b></p>
  
  <p>
    <img src="https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge" alt="Build Status" />
    <img src="https://img.shields.io/badge/python-3.12-blue?style=for-the-badge" alt="Python Version" />
    <img src="https://img.shields.io/badge/docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
    <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License" />
  </p>
</div>

## Overview

AdverTest acts as a "Red Team" AI pipeline that solves the PII Domain Gap by automatically auditing vision models, synthesizing physics-accurate adversarial edge cases (e.g., motion blur, fisheye), and running distributed retraining loops—without exposing raw PII data.

## Prerequisites

- **Docker** and **Docker Compose** installed (v2.0+)
- **Git**
- Node.js & npm (if running frontend locally outside of Docker)
- Python 3.12 (if running backend locally outside of Docker)

## Data Security & Compliance (Enterprise Grade)

AdverTest is engineered with strict Data Governance and PII protection mechanisms:
- **Zero-Persistence PII:** Raw images (which may contain PII) are streamed in memory for Triage. They are NEVER persisted locally longer than the inference cycle unless they are strictly identified as hard negatives.
- **Containerized Artifacts:** All synthesized datasets and metadata are securely containerized or uploaded to enterprise Object Storage (S3/R2) via encrypted channels. No raw data leaks to the local developer machine.

## Quick Start (Local Demo Mode)

This mode allows you to spin up the UI and FastAPI backend using pre-computed `audit_trail.json` and `proof_of_cure.json` artifacts. It requires **zero GPU compute** and is perfect for a fast mentor demonstration.

**Hybrid Decoupling Advantage:** This Local Demo Mode perfectly illustrates the decoupling of our architecture. The "Heavy ML Engine" (Adversarial synthesis and Ray Retraining) runs remotely (e.g., Colab GPU or AWS ECS), while the lightweight "API/UI Layer" runs seamlessly on a standard laptop.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/advertest.git
   cd advertest
   ```

2. **Configure Environment:**
   ```bash
   cp .env.example .env
   ```
   *(The default `.env` is pre-configured to enable Demo Mode).*

3. **Spin up the stack via Docker:**
   ```bash
   docker-compose up --build
   ```

4. **Access the Demo Dashboard:**
   - Frontend: [http://localhost:3000](http://localhost:3000)
   - Backend API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Open **[http://localhost:3000/engine](http://localhost:3000/engine)** for the five-module console: Triage Matrix → Adversarial Studio → Visual Inspection (V1/V2 diff slider) → Temporal Inspector → Proof of Cure.

Without Docker: `ENABLE_DEMO_MODE=True uvicorn backend.app.main:app --port 8000` and, in `frontend/`, `API_PROXY_TARGET=http://localhost:8000 npm run dev`.

### Demo artifacts & taxonomy

- Pre-computed artifacts live in `data/demo_artifacts/` (`audit_trail.json`, `proof_of_cure.json`, `eval_report.json`); regenerate deterministically with `python scripts/build_demo_artifacts.py`. Live runs write to `data/demo_output/*.live.json` and never overwrite them.
- Failure tags are canonical (`error_fisheye`, `error_motion_blur`, `error_sensor_noise`, `error_occlusion`) and defined once in `advertest/core/taxonomy.py`; legacy engine tags are mapped at the API boundary.
- The GPU stage (triage, synthesis, retrain, blind gate) is packaged in `colab/colab_yolov7_advertest.ipynb`. Model V2 is accepted only if corrupted mAP improves and base mAP drops by ≤ 1%.
- Data Block ingestion (`src/services/data_connector.py`) handles presigned-URL expiry (re-sign once), retries with backoff, and purges raw files from the temporary cache.

## Real Pipeline — zero local disk (Colab GPU worker)

The laptop never stores datasets, weights, PyTorch images or generated binaries (tracked repo ≈ 3.4 MB; CI enforces < 5 MB).
All heavy work is one script, [`colab/colab_zero_disk_pipeline.py`](colab/colab_zero_disk_pipeline.py) (also as `.ipynb`):

1. shallow-clone `derronqi/yolov7-face`, fetch the official tiny-face weights **into the Colab VM**;
2. stream a micro-subset of WIDER FACE (Hugging Face streaming) and, if you provide `EGO4D_PRESIGNED_URLS`, decode Ego4D clips straight from presigned URLs in RAM;
3. baseline inference → triage (IoU < 0.5 or conf < 0.4) → measured failure distribution;
4. constraint-driven attacks (radial distortion, biomechanical IMU blur, sensor degradation, occlusion; exclusion matrix; bbox/landmark integrity) with a data-grounded `audit_trail` per sample;
5. fine-tune with the repo's trainer (clean replay + targeted adversarial samples), then the **blind gate** on held-out images: accept V2 only if base mAP drops ≤ 1% and corrupted mAP gains ≥ 5 points;
6. export < 1 MB of JSON + thumbnails (`advertest_artifacts.zip`). Unzip into `data/demo_artifacts/` and the API/UI serve those *measured* results.

Notes: WIDER FACE has boxes only (landmarks come from the model and become pseudo-labels). Ego4D needs a licence, so without URLs the target domain is WIDER passed through the attack engine; the report's `provenance.target_domain` says which was used. The pure logic (AP, triage, geometry, tracker, gate) is unit-tested offline in `tests/test_colab_pipeline.py`.

*Refer to the `docs/beautiful_architecture.html` for a comprehensive system diagram and handover report.*
