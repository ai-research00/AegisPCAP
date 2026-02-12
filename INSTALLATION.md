# AegisPCAP Installation Guide

## Quick Start

### Prerequisites

- Python 3.9 or higher
- PostgreSQL 14+
- Redis 7+
- Node.js 18+ (for frontend)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/ai-research00/AegisPCAP.git
cd AegisPCAP
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your database and Redis settings
```

5. **Initialize database**
```bash
python -m src.db.manager --init
```

6. **Run tests**
```bash
pytest tests/ --cov=src
```

7. **Start backend API**
```bash
python -m src.api.main
```

8. **Install and start frontend** (in another terminal)
```bash
cd frontend
npm install
npm run build
npm start
```

9. **Access the application**
- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Docker Installation (Recommended)

```bash
# Start all services
docker-compose up -d

# Access dashboard
open http://localhost:3000
```

## Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n aegispcap
```

## Verify Installation

```bash
# Run test suite
pytest tests/

# Check API health
curl http://localhost:8000/health

# Analyze sample PCAP
python -m src.ingest.pcap_loader --input sample.pcap
```

## Troubleshooting

See [docs/troubleshooting/common-issues.md](docs/troubleshooting/common-issues.md) for common issues and solutions.

## Next Steps

- Read the [Quick Start Guide](docs/getting-started/quick-start.md)
- Explore the [User Guide](docs/user-guide/analyzing-pcaps.md)
- Learn about [Plugin Development](docs/developer-guide/plugin-development.md)
- Join the community on [GitHub Discussions](https://github.com/ai-research00/AegisPCAP/discussions)
