
# NoPickles.ai

_A modular, AI-first POS system for fast food chains – blending real-time personalization, order intelligence, and multimodal interaction._
(Under-construction)
---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Directory Structure](#directory-structure)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [License](#license)

---
## Overview

NoPickles is a production-grade conversational kiosk platform for fast food environments. It is designed to simulate a human-like customer service experience using advanced agentic workflows, real-time personalization, and voice/face recognition — all while maintaining full compatibility with traditional PoS and analytics systems.

The system includes:
- Multimodal customer interaction (voice, face, touch)
- Personalized upsell/cross-sell suggestions using retrieval-augmented generation
- Fine-tunable avatar models for conversational embeddings
- Manager-side dashboards for analytics, voice ops, and staff/inventory tracking
- Plug-and-play edge deployment across multiple kiosks

---

## Architecture

```text
[C# Windows IoT Frontend]
        │
        ▼
[FastAPI Gateway] ──▶ [LangChain Agent (RAG + LLMs)]
        │                          │
        ▼                          ▼
[Auth / Session Layer]      [PyTorch Avatar Models]
        │                          │
        ▼                          ▼
     [MySQL DB]              [Vector DB (FAISS)]
````

Kiosks interact via a lightweight FastAPI backend, supported by PyTorch-based models for real-time inference. All services are containerized for IoT or server-based deployment. Realtime event propagation (planned) will use Redis/MQTT. Infrastructure is defined in Terraform and K8s manifests.

---
## Tech Stack

**Frontend**

- C# (.NET) on Windows IoT Core for kiosk hardware integration
    
- WPF/XAML for customer-side UI
    

**Backend**

- FastAPI (Python 3.12+)
    
- PyTorch (transformer-based models, personalization)
    
- LangChain for orchestration and agent workflows
    
- Hugging Face Transformers (quantized + fine-tuned variants)
    
- FAISS (planned) for vector search in RAG
    
- MySQL for structured persistent data
    
- Docker + Kubernetes + Terraform for deployment
    
- GitHub Actions for CI
    

**Data Engineering**

- Redis Pub/Sub or MQTT for cross-kiosk messaging
    
- dbt + Airflow (under `data-engineering/`) for analytics pipelines
    
- Sentry + Prometheus/Grafana for observability
    

---
## Directory Structure

```text
.
├── apps/                 # Customer, Manager, and PoS interface apps
├── services/             # FastAPI gateway, agent orchestration, auth
├── ml/                   # PyTorch models for personalization, avatars
├── data/                 # Raw, mock, and processed datasets
├── data-engineering/     # DE pipelines: ingest, transform, warehouse (TBD)
├── infra/                # Docker, K8s, Terraform
├── scripts/              # One-off utilities, data migration, etc.
├── tests/                # Unit, integration, e2e tests
├── docs/                 # Markdown docs and model cards
├── all_legacy_code/      # Archived prototype codebase (non-critical)
├── notebooks/            # Experimental and training workflows
├── .github/              # Actions workflows, issue templates
```

---
## Development Setup

### Requirements

- Python 3.12+
    
- CUDA 12.8+ (for GPU acceleration)
    
- Docker & Docker Compose
    
- .NET SDK (for frontend C# development)
    
### Install Backend

```bash
git clone https://github.com/nopickles/nopickles.git
cd nopickles
python -m venv venv && source venv/bin/activate
pip install -r services/agent-core/requirements.txt
```
### Run All Services

```bash
docker-compose up --build
```

> Note: For local dev, start services manually via uvicorn / dotnet run. For production, use the `infra/` stack.

---
## Testing

```bash
pytest tests/unit
pytest tests/integration
```

E2E tests are under development for the full kiosk-to-server flow.

---
## License

This project is licensed under the MIT License.
