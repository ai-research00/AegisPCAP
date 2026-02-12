# AegisPCAP

**AI-Driven Network Security Intelligence Platform**

AegisPCAP is an enterprise-grade, AI-powered network traffic analysis system that transforms raw PCAP files into actionable security insights using advanced behavioral analysis, machine learning, and automated threat detection.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/typescript-5.0%2B-blue)](https://www.typescriptlang.org/)
[![Tests](https://img.shields.io/badge/tests-69%20passing-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen)](htmlcov/)

## 🚀 Features

### Core Capabilities
- **PCAP Processing**: Multi-protocol packet analysis (TCP, UDP, ICMP, IPv4, IPv6)
- **Feature Engineering**: 50+ behavioral indicators across 5 domains
- **ML Detection**: Ensemble models with 5 specialized threat detectors
- **AI Reasoning**: MITRE ATT&CK mapping with confidence scoring
- **Real-time Analysis**: WebSocket-based live threat monitoring
- **Enterprise Integration**: SOAR, SIEM, firewall, and ticketing system connectors

### Advanced Features
- **Behavioral Analysis**: DNS, TLS, timing, and statistical pattern detection
- **Threat Intelligence**: Community-driven indicator sharing
- **Plugin System**: Extensible architecture for custom analyzers
- **Model Registry**: Share and discover trained ML models
- **Compliance**: GDPR, HIPAA, CCPA support with anonymization
- **Distributed Processing**: Kubernetes-ready with auto-scaling

## 📊 Project Status

**Version**: 1.0.0 (Production Ready)  
**Completion**: Phases 1-14 ✅ Complete | Phase 15 🚧 In Progress (33%)

| Phase | Component | Status | LOC |
|-------|-----------|--------|-----|
| 1-5 | Core Platform | ✅ Complete | 4,500+ |
| 6 | Dashboards | ✅ Complete | 6,440+ |
| 7 | Integrations | ✅ Complete | 2,620+ |
| 8-9 | Testing & DevOps | ✅ Complete | 1,393+ |
| 10-14 | Advanced Features | ✅ Complete | 13,540+ |
| 15 | Community Ecosystem | 🚧 In Progress | 1,200+ |

**Total**: ~30,000 lines of production code

## 🎯 Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL 14+
- Redis 7+
- Node.js 18+ (for frontend)

### Installation

```bash
# Clone repository
git clone https://github.com/ai-research00/AegisPCAP.git
cd AegisPCAP

# Backend setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd frontend
npm install
npm run build
cd ..

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run tests
pytest tests/

# Start services
docker-compose up -d  # PostgreSQL, Redis, monitoring
python -m src.api.main  # Backend API
```

### Docker Deployment

```bash
# Quick start with Docker Compose
docker-compose up -d

# Access dashboard
open http://localhost:3000
```

## 📖 Documentation

- **[Getting Started](docs/getting-started/)** - Installation and first analysis
- **[User Guide](docs/user-guide/)** - Analyzing PCAPs, investigating alerts
- **[Developer Guide](docs/developer-guide/)** - Architecture, plugin development
- **[API Reference](docs/reference/api/)** - REST API documentation
- **[Contributing](CONTRIBUTING.md)** - How to contribute
- **[Roadmap](ROADMAP.md)** - Future plans

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React + TypeScript)             │
│  Dashboard │ Alerts │ Incidents │ Network Viz │ Analytics   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                       │
│  REST Endpoints │ WebSocket │ Authentication │ Rate Limiting│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Analysis Engine                      │
│  PCAP Loader │ Flow Builder │ Feature Extraction │ ML Models│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data & Storage Layer                      │
│  PostgreSQL │ Redis │ Prometheus │ Elasticsearch (optional) │
└─────────────────────────────────────────────────────────────┘
```

## 🔬 Analysis Pipeline

1. **Ingest**: Parse PCAP packets (scapy/tshark)
2. **Aggregate**: Build bidirectional flows (5-tuple)
3. **Extract**: Compute 50+ behavioral features
4. **Detect**: Run ensemble ML models
5. **Reason**: AI agent correlates findings with MITRE ATT&CK
6. **Alert**: Generate actionable security insights

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Ways to Contribute
- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🔧 Submit pull requests
- 🧪 Share trained models
- 🔍 Contribute threat intelligence

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MITRE ATT&CK framework for threat taxonomy
- CICIDS datasets for training and benchmarking
- Open source community for tools and libraries

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/ai-research00/AegisPCAP/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ai-research00/AegisPCAP/discussions)
- **Email**: support@aegispcap.org

## ⭐ Star History

If you find AegisPCAP useful, please consider giving it a star! ⭐

---

**Built with ❤️ by the AegisPCAP community**
