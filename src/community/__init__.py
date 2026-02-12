"""
AegisPCAP Community & Ecosystem Module

This module provides infrastructure for community collaboration, including:
- Plugin system for extensibility
- Model registry for sharing trained models
- Research API for academic access
- Contribution framework for open source collaboration
- Threat intelligence feed for community-driven detection
- Extension marketplace for discovering and installing plugins
- Documentation portal and community forum
- Analytics and telemetry for understanding usage
"""

from src.community.plugins.interface import PluginInterface, PluginMetadata
from src.community.plugins.manager import PluginManager
from src.community.plugins.sandbox import PluginSandbox
from src.community.models.registry import ModelRegistry
from src.community.models.validator import ModelValidator
from src.community.threat_intel.feed import ThreatIntelligenceFeed
from src.community.threat_intel.stix_converter import STIXConverter
from src.community.research.api import CommunityResearchAPI
from src.community.research.anonymizer import DataAnonymizer

__version__ = "1.0.0"
__all__ = [
    # Plugin System
    "PluginInterface",
    "PluginMetadata",
    "PluginManager",
    "PluginSandbox",
    
    # Model Registry
    "ModelRegistry",
    "ModelValidator",
    
    # Threat Intelligence
    "ThreatIntelligenceFeed",
    "STIXConverter",
    
    # Research API
    "CommunityResearchAPI",
    "DataAnonymizer",
    
    # Module names
    "plugins",
    "models",
    "research",
    "contributions",
    "threat_intel",
    "marketplace",
    "docs",
    "forum",
    "analytics",
]

