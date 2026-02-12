"""
Plugin System for AegisPCAP

Enables third-party extensions without modifying core code.
Provides plugin interface, lifecycle management, and sandboxed execution.
"""

from .interface import PluginInterface, PluginMetadata, PluginData, PluginResult, PluginType
from .manager import PluginManager
from .sandbox import PluginSandbox

__all__ = [
    "PluginInterface",
    "PluginMetadata",
    "PluginData",
    "PluginResult",
    "PluginType",
    "PluginManager",
    "PluginSandbox",
]
