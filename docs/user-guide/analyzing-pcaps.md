# Analyzing PCAPs

## Overview

AegisPCAP analyzes network traffic from PCAP files to detect threats.

## Steps

1. **Upload PCAP**: Use CLI or API to upload PCAP file
2. **Processing**: System extracts flows and features
3. **Detection**: ML models analyze for threats
4. **Review**: View alerts in dashboard

## CLI Usage

```bash
python -m src.ingest.pcap_loader --input traffic.pcap
```

## API Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/pcap/upload",
    files={"file": open("traffic.pcap", "rb")}
)
```

See [API Reference](../developer-guide/api-reference.md) for details.
