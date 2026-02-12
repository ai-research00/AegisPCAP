# Installation Guide

## Prerequisites

- Python 3.9+
- PostgreSQL 14+
- Redis 7+
- Node.js 18+ (for frontend)

## Quick Install

```bash
# Clone repository
git clone https://github.com/ai-research00/AegisPCAP.git
cd AegisPCAP

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -m src.db.manager --init

# Run tests
pytest tests/

# Start services
python -m src.api.main
```

## Docker Installation

```bash
docker-compose up -d
```

See [Quick Start](quick-start.md) for next steps.
