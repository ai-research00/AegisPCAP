"""
Plugin Manager

Manages plugin lifecycle including loading, validation, execution, and unloading.
Provides plugin discovery and registry management.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .interface import PluginInterface, PluginMetadata, PluginData, PluginResult

logger = logging.getLogger(__name__)


class PluginValidationError(Exception):
    """Raised when plugin validation fails."""
    pass


class PluginLoadError(Exception):
    """Raised when plugin loading fails."""
    pass


class PluginManager:
    """
    Manages plugin lifecycle and execution.
    
    Responsibilities:
    - Load and validate plugins
    - Maintain plugin registry
    - Execute plugins with error handling
    - Unload plugins and cleanup resources
    """
    
    def __init__(self, plugin_dir: Optional[Path] = None):
        """
        Initialize plugin manager.
        
        Args:
            plugin_dir: Directory containing plugins (default: ./plugins)
        """
        self.plugin_dir = plugin_dir or Path("./plugins")
        self.plugins: Dict[str, PluginInterface] = {}
        self.metadata: Dict[str, PluginMetadata] = {}
        self._aegis_version = "1.0.0"  # TODO: Get from config
        
        logger.info(f"PluginManager initialized with plugin_dir: {self.plugin_dir}")
    
    def load_plugin(self, plugin_path: str) -> PluginInterface:
        """
        Load and validate a plugin.
        
        Steps:
        1. Import plugin module
        2. Find PluginInterface implementation
        3. Validate interface conformance
        4. Check version compatibility
        5. Validate dependencies
        6. Initialize plugin
        7. Register in plugin registry
        
        Args:
            plugin_path: Path to plugin file (relative or absolute)
            
        Returns:
            Loaded and initialized plugin instance
            
        Raises:
            PluginLoadError: If plugin cannot be loaded
            PluginValidationError: If plugin validation fails
        """
        try:
            # Convert to Path object
            path = Path(plugin_path)
            if not path.is_absolute():
                path = self.plugin_dir / path
            
            if not path.exists():
                raise PluginLoadError(f"Plugin file not found: {path}")
            
            # Import plugin module
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Cannot load plugin spec from: {path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = module
            spec.loader.exec_module(module)
            
            # Find PluginInterface implementation
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, PluginInterface) and 
                    attr is not PluginInterface):
                    plugin_class = attr
                    break
            
            if plugin_class is None:
                raise PluginValidationError(
                    f"No PluginInterface implementation found in: {path}"
                )
            
            # Instantiate plugin
            plugin = plugin_class()
            
            # Validate interface conformance
            self._validate_interface(plugin)
            
            # Get metadata
            metadata = plugin.get_metadata()
            
            # Check version compatibility
            self._check_version_compatibility(metadata)
            
            # Validate dependencies
            self._validate_dependencies(metadata)
            
            # Initialize plugin with empty config (can be extended)
            plugin.initialize({})
            
            # Register plugin
            self.plugins[metadata.plugin_id] = plugin
            self.metadata[metadata.plugin_id] = metadata
            
            logger.info(
                f"Successfully loaded plugin: {metadata.name} "
                f"v{metadata.version} ({metadata.plugin_id})"
            )
            
            return plugin
            
        except (PluginLoadError, PluginValidationError):
            raise
        except Exception as e:
            raise PluginLoadError(f"Failed to load plugin from {plugin_path}: {e}")
    
    def unload_plugin(self, plugin_id: str) -> None:
        """
        Unload a plugin and cleanup resources.
        
        Args:
            plugin_id: ID of plugin to unload
            
        Raises:
            KeyError: If plugin not found
        """
        if plugin_id not in self.plugins:
            raise KeyError(f"Plugin not found: {plugin_id}")
        
        plugin = self.plugins[plugin_id]
        metadata = self.metadata[plugin_id]
        
        try:
            # Cleanup plugin resources
            plugin.cleanup()
            logger.info(f"Plugin cleanup successful: {metadata.name}")
        except Exception as e:
            logger.error(f"Plugin cleanup failed for {metadata.name}: {e}")
        
        # Remove from registry
        del self.plugins[plugin_id]
        del self.metadata[plugin_id]
        
        logger.info(f"Unloaded plugin: {metadata.name} ({plugin_id})")
    
    def execute_plugin(self, plugin_id: str, data: PluginData) -> PluginResult:
        """
        Execute plugin in isolated environment with error handling.
        
        Args:
            plugin_id: ID of plugin to execute
            data: Input data for plugin
            
        Returns:
            PluginResult with execution results
            
        Raises:
            KeyError: If plugin not found
        """
        if plugin_id not in self.plugins:
            raise KeyError(f"Plugin not found: {plugin_id}")
        
        plugin = self.plugins[plugin_id]
        metadata = self.metadata[plugin_id]
        
        try:
            # Validate input data
            if not plugin.validate_data(data):
                return PluginResult(
                    success=False,
                    error="Input data validation failed"
                )
            
            # Execute plugin
            logger.debug(f"Executing plugin: {metadata.name}")
            result = plugin.process(data)
            
            logger.debug(f"Plugin execution completed: {metadata.name}")
            return result
            
        except Exception as e:
            logger.error(f"Plugin execution failed for {metadata.name}: {e}")
            return PluginResult(
                success=False,
                error=f"Plugin execution error: {str(e)}"
            )
    
    def list_plugins(self) -> List[PluginMetadata]:
        """
        List all loaded plugins.
        
        Returns:
            List of plugin metadata for all loaded plugins
        """
        return list(self.metadata.values())
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginInterface]:
        """
        Get plugin by ID.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(plugin_id)
    
    def get_metadata(self, plugin_id: str) -> Optional[PluginMetadata]:
        """
        Get plugin metadata by ID.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            Plugin metadata or None if not found
        """
        return self.metadata.get(plugin_id)
    
    def _validate_interface(self, plugin: PluginInterface) -> None:
        """
        Validate that plugin implements required interface methods.
        
        Args:
            plugin: Plugin instance to validate
            
        Raises:
            PluginValidationError: If interface validation fails
        """
        required_methods = ['initialize', 'process', 'cleanup', 'get_metadata']
        
        for method in required_methods:
            if not hasattr(plugin, method):
                raise PluginValidationError(
                    f"Plugin missing required method: {method}"
                )
            if not callable(getattr(plugin, method)):
                raise PluginValidationError(
                    f"Plugin method not callable: {method}"
                )
    
    def _check_version_compatibility(self, metadata: PluginMetadata) -> None:
        """
        Check if plugin is compatible with current AegisPCAP version.
        
        Args:
            metadata: Plugin metadata
            
        Raises:
            PluginValidationError: If version incompatible
        """
        # Simple version check (can be enhanced with semantic versioning)
        min_version = metadata.min_aegis_version
        max_version = metadata.max_aegis_version
        
        if self._aegis_version < min_version:
            raise PluginValidationError(
                f"Plugin requires AegisPCAP >= {min_version}, "
                f"current version: {self._aegis_version}"
            )
        
        if max_version and self._aegis_version > max_version:
            raise PluginValidationError(
                f"Plugin requires AegisPCAP <= {max_version}, "
                f"current version: {self._aegis_version}"
            )
    
    def _validate_dependencies(self, metadata: PluginMetadata) -> None:
        """
        Validate plugin dependencies are available.
        
        Args:
            metadata: Plugin metadata
            
        Raises:
            PluginValidationError: If required dependencies missing
        """
        for dep in metadata.dependencies:
            if dep.optional:
                continue
            
            try:
                importlib.import_module(dep.name)
            except ImportError:
                raise PluginValidationError(
                    f"Required dependency not found: {dep.name} {dep.version}"
                )
