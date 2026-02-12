# Quick Start Guide

## First Analysis

1. Start AegisPCAP services
2. Upload a PCAP file
3. View threat detections in dashboard

```bash
# Analyze PCAP
python -m src.ingest.pcap_loader --input sample.pcap

# View results
open http://localhost:3000/flows
```

## Next Steps

- [User Guide](../user-guide/analyzing-pcaps.md)
- [API Reference](../developer-guide/api-reference.md)
- [Tutorials](../tutorials/)
