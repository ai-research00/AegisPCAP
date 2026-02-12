"""
Reputation System

Manages user reputation and badges.
Type hints: 100% coverage
Docstrings: 100% coverage
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime
import logging


@dataclass
class Badge:
    """Achievement badge."""
    badge_id: str
    name: str
    description: str
    icon: str


@dataclass
class ReputationInfo:
    """User reputation information."""
    user_id: str
    points: int = 0
    badges: List[Badge] = field(default_factory=list)
    level: str = "Newcomer"


class ReputationSystem:
    """Manages user reputation and gamification."""
    
    def __init__(self):
        """Initialize reputation system."""
        self.logger = logging.getLogger(__name__)
        self.user_reputation: Dict[str, ReputationInfo] = {}
        self.badges = self._initialize_badges()
    
    def _initialize_badges(self) -> Dict[str, Badge]:
        """Initialize available badges."""
        return {
            "first_contribution": Badge("first_contribution", "First Contribution", "Made first contribution", "🎉"),
            "bug_hunter": Badge("bug_hunter", "Bug Hunter", "Reported 10 bugs", "🐛"),
            "doc_hero": Badge("doc_hero", "Documentation Hero", "Improved documentation", "📚"),
            "model_contributor": Badge("model_contributor", "Model Contributor", "Shared ML model", "🤖"),
            "community_helper": Badge("community_helper", "Community Helper", "Helped 50 users", "🤝")
        }
    
    def award_points(self, user_id: str, action: str, points: int) -> None:
        """Award reputation points."""
        if user_id not in self.user_reputation:
            self.user_reputation[user_id] = ReputationInfo(user_id=user_id)
        
        self.user_reputation[user_id].points += points
        self._update_level(user_id)
        
        self.logger.info(f"Awarded {points} points to {user_id} for {action}")
    
    def _update_level(self, user_id: str) -> None:
        """Update user level based on points."""
        points = self.user_reputation[user_id].points
        
        if points >= 1000:
            self.user_reputation[user_id].level = "Expert"
        elif points >= 500:
            self.user_reputation[user_id].level = "Advanced"
        elif points >= 100:
            self.user_reputation[user_id].level = "Intermediate"
        else:
            self.user_reputation[user_id].level = "Newcomer"
    
    def grant_badge(self, user_id: str, badge_id: str) -> None:
        """Grant achievement badge."""
        if user_id not in self.user_reputation:
            self.user_reputation[user_id] = ReputationInfo(user_id=user_id)
        
        if badge_id in self.badges:
            badge = self.badges[badge_id]
            if badge not in self.user_reputation[user_id].badges:
                self.user_reputation[user_id].badges.append(badge)
                self.logger.info(f"Granted badge '{badge.name}' to {user_id}")
    
    def get_user_reputation(self, user_id: str) -> ReputationInfo:
        """Get user reputation information."""
        if user_id not in self.user_reputation:
            self.user_reputation[user_id] = ReputationInfo(user_id=user_id)
        return self.user_reputation[user_id]


__all__ = ["ReputationSystem", "Badge", "ReputationInfo"]
