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

Navigate to the **"E2E Demo Workflow"** button in the dashboard to see the active learning loop in action!

## Production Mode (GPU Enabled)

To run the full adversarial engine and actually mutate data on the fly, you will need a CUDA-enabled GPU and a Ray Cluster configured.

1. Set `ENABLE_DEMO_MODE=False` in your `.env` file.
2. Ensure you have your `yolov7-tiny-face.pt` weights placed in the `weights/` directory.
3. Deploy the API via the optimized `Dockerfile.api` (Railway/AWS ECS compatible).
4. Spin up the orchestrator to listen for RabbitMQ/Kafka streaming inputs.

*Refer to the `docs/beautiful_architecture.html` for a comprehensive system diagram and handover report.*
