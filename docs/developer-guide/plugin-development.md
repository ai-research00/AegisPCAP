# Plugin Development Guide

## Creating a Plugin

Plugins extend AegisPCAP with custom analyzers and detectors.

## Plugin Interface

```python
from src.community.plugins.interface import PluginInterface, PluginMetadata

class MyPlugin(PluginInterface):
    def initialize(self, config):
        # Setup plugin
        pass
    
    def process(self, data):
        # Process data
        return result
    
    def cleanup(self):
        # Cleanup resources
        pass
    
    def get_metadata(self):
        return PluginMetadata(
            plugin_id="my-plugin",
            name="My Plugin",
            version="1.0.0",
            author="me@example.com"
        )
```

## Installation

```bash
python -m src.community.plugins.manager --install my_plugin.py
```

See [Contributing Guide](../../CONTRIBUTING.md) for submission process.
