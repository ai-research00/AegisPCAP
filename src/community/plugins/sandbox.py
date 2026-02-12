"""
Plugin Sandbox

Provides isolated execution environment for plugins with resource limits,
capability-based permissions, and error isolation.
"""

import logging
import multiprocessing
import signal
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .interface import PluginInterface, PluginData, PluginResult

logger = logging.getLogger(__name__)


class Capability(Enum):
    """Capabilities that can be granted to plugins."""
    READ_FLOWS = "read_flows"
    READ_FEATURES = "read_features"
    READ_PREDICTIONS = "read_predictions"
    WRITE_METRICS = "write_metrics"
    WRITE_VISUALIZATIONS = "write_visualizations"
    NETWORK_ACCESS = "network_access"
    FILE_SYSTEM_READ = "file_system_read"
    FILE_SYSTEM_WRITE = "file_system_write"


@dataclass
class ResourceLimits:
    """Resource limits for plugin execution."""
    max_cpu_percent: float = 50.0  # Maximum CPU usage percentage
    max_memory_mb: int = 512  # Maximum memory in MB
    timeout_seconds: int = 30  # Execution timeout in seconds


@dataclass
class SandboxConfig:
    """Configuration for plugin sandbox."""
    resource_limits: ResourceLimits
    capabilities: Set[Capability]
    allow_network: bool = False
    allow_file_system: bool = False


class PluginSandbox:
    """
    Isolates plugin execution for security.
    
    Provides:
    - Resource limits (CPU, memory, timeout)
    - Capability-based permissions
    - Error isolation and recovery
    - Execution monitoring
    """
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        Initialize plugin sandbox.
        
        Args:
            config: Sandbox configuration (uses defaults if None)
        """
        self.config = config or SandboxConfig(
            resource_limits=ResourceLimits(),
            capabilities={
                Capability.READ_FLOWS,
                Capability.READ_FEATURES,
                Capability.READ_PREDICTIONS,
                Capability.WRITE_METRICS,
                Capability.WRITE_VISUALIZATIONS,
            }
        )
        logger.info("PluginSandbox initialized with resource limits")
    
    def execute_isolated(
        self,
        plugin: PluginInterface,
        data: PluginData,
        timeout: Optional[int] = None
    ) -> PluginResult:
        """
        Execute plugin with resource limits and timeout.
        
        Uses multiprocessing to isolate plugin execution and enforce
        resource limits. Handles timeouts and errors gracefully.
        
        Args:
            plugin: Plugin to execute
            data: Input data for plugin
            timeout: Execution timeout in seconds (overrides config)
            
        Returns:
            PluginResult with execution results or error
        """
        timeout = timeout or self.config.resource_limits.timeout_seconds
        
        try:
            # Validate capabilities before execution
            if not self._check_capabilities(data):
                return PluginResult(
                    success=False,
                    error="Plugin lacks required capabilities for data access"
                )
            
            # Execute in separate process with timeout
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=self._execute_in_process,
                args=(plugin, data, result_queue)
            )
            
            process.start()
            process.join(timeout=timeout)
            
            if process.is_alive():
                # Timeout occurred
                logger.warning(f"Plugin execution timeout after {timeout}s")
                process.terminate()
                process.join(timeout=5)
                
                if process.is_alive():
                    # Force kill if still alive
                    process.kill()
                    process.join()
                
                return PluginResult(
                    success=False,
                    error=f"Plugin execution timeout after {timeout} seconds"
                )
            
            # Get result from queue
            if not result_queue.empty():
                result = result_queue.get()
                return result
            else:
                return PluginResult(
                    success=False,
                    error="Plugin execution failed without result"
                )
                
        except Exception as e:
            logger.error(f"Sandbox execution error: {e}")
            return PluginResult(
                success=False,
                error=f"Sandbox execution error: {str(e)}"
            )
    
    def _execute_in_process(
        self,
        plugin: PluginInterface,
        data: PluginData,
        result_queue: multiprocessing.Queue
    ) -> None:
        """
        Execute plugin in separate process.
        
        This runs in a child process and puts the result in the queue.
        
        Args:
            plugin: Plugin to execute
            data: Input data
            result_queue: Queue to put result in
        """
        try:
            # Set resource limits (Unix-like systems)
            try:
                import resource
                
                # Set memory limit
                max_memory = self.config.resource_limits.max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
                
                # Set CPU time limit
                cpu_limit = self.config.resource_limits.timeout_seconds
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
                
            except (ImportError, AttributeError):
                # resource module not available (Windows)
                logger.warning("Resource limits not available on this platform")
            
            # Execute plugin
            result = plugin.process(data)
            result_queue.put(result)
            
        except Exception as e:
            logger.error(f"Plugin execution error in process: {e}")
            result_queue.put(PluginResult(
                success=False,
                error=f"Plugin execution error: {str(e)}"
            ))
    
    def _check_capabilities(self, data: PluginData) -> bool:
        """
        Check if plugin has required capabilities for data access.
        
        Args:
            data: Input data to check
            
        Returns:
            True if plugin has required capabilities
        """
        required_caps = set()
        
        if data.flow_data is not None:
            required_caps.add(Capability.READ_FLOWS)
        
        if data.features is not None:
            required_caps.add(Capability.READ_FEATURES)
        
        if data.ml_predictions is not None:
            required_caps.add(Capability.READ_PREDICTIONS)
        
        # Check if plugin has all required capabilities
        return required_caps.issubset(self.config.capabilities)
    
    def grant_capability(self, capability: Capability) -> None:
        """
        Grant a capability to the sandbox.
        
        Args:
            capability: Capability to grant
        """
        self.config.capabilities.add(capability)
        logger.info(f"Granted capability: {capability.value}")
    
    def revoke_capability(self, capability: Capability) -> None:
        """
        Revoke a capability from the sandbox.
        
        Args:
            capability: Capability to revoke
        """
        self.config.capabilities.discard(capability)
        logger.info(f"Revoked capability: {capability.value}")
    
    def has_capability(self, capability: Capability) -> bool:
        """
        Check if sandbox has a capability.
        
        Args:
            capability: Capability to check
            
        Returns:
            True if capability is granted
        """
        return capability in self.config.capabilities
    
    def update_resource_limits(self, limits: ResourceLimits) -> None:
        """
        Update resource limits for sandbox.
        
        Args:
            limits: New resource limits
        """
        self.config.resource_limits = limits
        logger.info(
            f"Updated resource limits: "
            f"CPU={limits.max_cpu_percent}%, "
            f"Memory={limits.max_memory_mb}MB, "
            f"Timeout={limits.timeout_seconds}s"
        )
