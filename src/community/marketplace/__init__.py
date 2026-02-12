"""
Extension Marketplace Module

Provides extension discovery, installation, and management.
"""

from src.community.marketplace.marketplace import ExtensionMarketplace
from src.community.marketplace.verifier import ExtensionVerifier

__all__ = [
    "ExtensionMarketplace",
    "ExtensionVerifier"
]
