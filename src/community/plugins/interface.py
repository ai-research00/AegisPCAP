"""
Plugin Interface and Data Models

Defines the base interface that all plugins must implement,
along with data structures for plugin metadata and execution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class PluginType(Enum):
    """Types of plugins supported by AegisPCAP."""
    ANALYZER = "analyzer"  # Custom feature extractors and analyzers
    DETECTOR = "detector"  # Custom threat detection algorithms
    INTEGRATION = "integration"  # Connectors to external systems
    VISUALIZATION = "visualization"  # Custom dashboards and charts


@dataclass
class Dependency:
    """Represents a plugin dependency."""
    name: str
    version: str
    optional: bool = False


@dataclass
class PluginMetadata:
    """Metadata describing a plugin."""
    plugin_id: str
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    capabilities: List[str]
    dependencies: List[Dependency]
    min_aegis_version: str
    max_aegis_version: Optional[str] = None
    license: str = "MIT"
    homepage: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "plugin_type": self.plugin_type.value,
            "capabilities": self.capabilities,
            "dependencies": [
                {"name": d.name, "version": d.version, "optional": d.optional}
                for d in self.dependencies
            ],
            "min_aegis_version": self.min_aegis_version,
            "max_aegis_version": self.max_aegis_version,
            "license": self.license,
            "homepage": self.homepage,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class PluginData:
    """Data passed to plugins for processing."""
    flow_data: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    ml_predictions: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plugin data to dictionary."""
        return {
            "flow_data": self.flow_data,
            "features": self.features,
            "ml_predictions": self.ml_predictions,
            "context": self.context,
        }


@dataclass
class PluginResult:
    """Result returned by plugin execution."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plugin result to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metrics": self.metrics,
            "visualizations": self.visualizations,
        }


class PluginInterface(ABC):
    """
    Base interface that all plugins must implement.
    
    Plugins extend AegisPCAP functionality without modifying core code.
    They run in sandboxed environments with controlled API access.
    """
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize plugin with configuration.
        
        Called once when the plugin is loaded. Use this to set up
        any resources, connections, or state needed by the plugin.
        
        Args:
            config: Plugin-specific configuration dictionary
            
        Raises:
            Exception: If initialization fails
        """
        pass
    
    @abstractmethod
    def process(self, data: PluginData) -> PluginResult:
        """
        Process data and return results.
        
        This is the main entry point for plugin execution.
        Receives flow data, features, and ML predictions, and
        returns analysis results, metrics, or visualizations.
        
        Args:
            data: Input data including flows, features, predictions
            
        Returns:
            PluginResult with success status, data, and optional metrics
            
        Raises:
            Exception: If processing fails
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup resources before unload.
        
        Called when the plugin is being unloaded. Use this to
        close connections, release resources, and save state.
        
        Raises:
            Exception: If cleanup fails
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """
        Return plugin metadata.
        
        Provides information about the plugin including name,
        version, author, capabilities, and dependencies.
        
        Returns:
            PluginMetadata describing this plugin
        """
        pass
    
    def validate_data(self, data: PluginData) -> bool:
        """
        Validate input data before processing.
        
        Override this method to add custom validation logic.
        Default implementation checks for required fields.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        # Basic validation - can be overridden by plugins
        return data is not None
    
    def get_capabilities(self) -> List[str]:
        """
        Get list of capabilities this plugin provides.
        
        Returns:
            List of capability strings
        """
        return self.get_metadata().capabilities
