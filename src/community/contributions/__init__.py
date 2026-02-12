"""
Community Contributions Module

Manages contribution workflow automation:
- Contribution validation
- CI/CD pipeline execution
- Reviewer assignment
- Changelog automation
"""

from src.community.contributions.manager import ContributionManager

__all__ = [
    "ContributionManager"
]
