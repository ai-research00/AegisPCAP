"""
Community Research API Module

Extends Phase 14 Research API with community features:
- Anonymized data access with PII removal
- Rate limiting and quota management
- Access control and authentication
- Audit logging for compliance
"""

from src.community.research.api import CommunityResearchAPI
from src.community.research.anonymizer import DataAnonymizer

__all__ = [
    "CommunityResearchAPI",
    "DataAnonymizer"
]
