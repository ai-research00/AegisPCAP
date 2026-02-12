# AegisPCAP

AegisPCAP is an AI-agent-driven network traffic intelligence system that converts raw PCAP files into actionable security insights using flow-based behavioral analysis and machine learning.

## Core Capabilities
- PCAP → Flow → Feature extraction pipeline
- Behavioral anomaly detection (Isolation Forest baseline)
- DNS & TLS heuristic analysis (no decryption required)
- AI agent orchestration for reasoning & explanation
- Offline, analyst-first design

## Analysis Pipeline
1. Parse PCAP packets
2. Aggregate packets into bidirectional flows
3. Extract statistical, timing, DNS, and TLS features
4. Detect anomalies using classical ML
5. AI agent correlates findings and generates explanations

## Initial ML Strategy
- Isolation Forest for unsupervised anomaly detection
- Designed to evolve into hybrid ML + deep learning

## Status
Active PoC development (research & engineering focused)
