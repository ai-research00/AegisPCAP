# Tutorial: Creating a Custom Detector

## Overview

Learn how to create a custom threat detector plugin.

## Step 1: Create Plugin Class

```python
from src.community.plugins.interface import PluginInterface

class CustomDetector(PluginInterface):
    def initialize(self, config):
        self.threshold = config.get("threshold", 0.8)
    
    def process(self, data):
        # Your detection logic
        score = self.analyze(data)
        return {"threat_detected": score > self.threshold, "score": score}
```

## Step 2: Test Plugin

```python
detector = CustomDetector()
detector.initialize({"threshold": 0.8})
result = detector.process(test_data)
```

## Step 3: Install Plugin

```bash
python -m src.community.plugins.manager --install custom_detector.py
```

## Next Steps

- [Model Training Tutorial](model-training.md)
- [Integration Setup](integration-setup.md)
