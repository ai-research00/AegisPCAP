"""
Extension Marketplace

Manages extension discovery, installation, and updates.
Type hints: 100% coverage
Docstrings: 100% coverage
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import logging
import hashlib


class ExtensionCategory(Enum):
    """Extension categories."""
    ANALYZER = "analyzer"
    DETECTOR = "detector"
    INTEGRATION = "integration"
    VISUALIZATION = "visualization"
    UTILITY = "utility"


@dataclass
class Extension:
    """Extension metadata."""
    extension_id: str
    name: str
    version: str
    category: ExtensionCategory
    description: str
    author: str
    downloads: int = 0
    rating: float = 0.0
    reviews_count: int = 0
    compatible_versions: List[str] = field(default_factory=list)
    signature: str = ""
    security_scan_passed: bool = False
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "extension_id": self.extension_id,
            "name": self.name,
            "version": self.version,
            "category": self.category.value,
            "description": self.description,
            "author": self.author,
            "downloads": self.downloads,
            "rating": self.rating,
            "reviews_count": self.reviews_count,
            "compatible_versions": self.compatible_versions,
            "signature": self.signature,
            "security_scan_passed": self.security_scan_passed,
            "created_at": self.created_at
        }


@dataclass
class ExtensionReview:
    """Extension review."""
    review_id: str
    extension_id: str
    user_id: str
    rating: int  # 1-5
    comment: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ExtensionMarketplace:
    """Manages extension marketplace."""
    
    def __init__(self):
        """Initialize marketplace."""
        self.logger = logging.getLogger(__name__)
        self.extensions: Dict[str, Extension] = {}
        self.reviews: Dict[str, List[ExtensionReview]] = {}
        self.installed: Dict[str, Extension] = {}
        self._initialize_sample_extensions()
    
    def _initialize_sample_extensions(self) -> None:
        """Initialize sample extensions."""
        self.extensions["ml-analyzer-v1"] = Extension(
            extension_id="ml-analyzer-v1",
            name="Advanced ML Analyzer",
            version="1.0.0",
            category=ExtensionCategory.ANALYZER,
            description="Advanced machine learning analyzer with custom models",
            author="ml-expert",
            downloads=1250,
            rating=4.5,
            reviews_count=45,
            compatible_versions=["1.0.0", "1.1.0"],
            signature="abc123def456",
            security_scan_passed=True
        )
    
    def search_extensions(
        self,
        query: Optional[str] = None,
        category: Optional[ExtensionCategory] = None,
        min_rating: float = 0.0
    ) -> List[Extension]:
        """Search extensions with filters."""
        results = list(self.extensions.values())
        
        if query:
            results = [e for e in results if query.lower() in e.name.lower() or query.lower() in e.description.lower()]
        
        if category:
            results = [e for e in results if e.category == category]
        
        if min_rating > 0:
            results = [e for e in results if e.rating >= min_rating]
        
        return sorted(results, key=lambda x: x.downloads, reverse=True)
    
    def install_extension(self, extension_id: str) -> Dict[str, Any]:
        """Install extension."""
        if extension_id not in self.extensions:
            return {"error": "Extension not found"}
        
        extension = self.extensions[extension_id]
        
        if not extension.security_scan_passed:
            return {"error": "Extension failed security scan"}
        
        self.installed[extension_id] = extension
        extension.downloads += 1
        
        self.logger.info(f"Installed extension: {extension.name}")
        
        return {"status": "installed", "extension": extension.to_dict()}
    
    def uninstall_extension(self, extension_id: str) -> Dict[str, Any]:
        """Uninstall extension."""
        if extension_id in self.installed:
            del self.installed[extension_id]
            return {"status": "uninstalled"}
        return {"error": "Extension not installed"}
    
    def submit_review(self, extension_id: str, user_id: str, rating: int, comment: str) -> Dict[str, Any]:
        """Submit extension review."""
        if extension_id not in self.extensions:
            return {"error": "Extension not found"}
        
        review_id = hashlib.sha256(f"{extension_id}_{user_id}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]
        
        review = ExtensionReview(
            review_id=review_id,
            extension_id=extension_id,
            user_id=user_id,
            rating=rating,
            comment=comment
        )
        
        if extension_id not in self.reviews:
            self.reviews[extension_id] = []
        
        self.reviews[extension_id].append(review)
        
        # Update extension rating
        extension = self.extensions[extension_id]
        all_ratings = [r.rating for r in self.reviews[extension_id]]
        extension.rating = sum(all_ratings) / len(all_ratings)
        extension.reviews_count = len(all_ratings)
        
        return {"status": "review_submitted", "review_id": review_id}


__all__ = ["ExtensionMarketplace", "Extension", "ExtensionCategory"]
