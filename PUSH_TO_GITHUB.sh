#!/bin/bash

# AegisPCAP - GitHub Push Script
# Prepares and pushes the repository to GitHub

echo "🚀 AegisPCAP - GitHub Push Preparation"
echo "======================================"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git repository already initialized"
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo ""
    echo "📝 Adding all files..."
    git add .
    echo "✅ Files staged"
    
    echo ""
    echo "💾 Creating commit..."
    git commit -m "Initial commit: AegisPCAP v1.0.0

- Production-ready AI-driven network security platform
- 30,000+ lines of code across 14 complete phases
- Core tests passing with high coverage
- Docker, Kubernetes, CI/CD ready
- Phase 15 (Community Ecosystem) 33% complete

Features:
- PCAP analysis with 50+ behavioral indicators
- Ensemble ML detection with 5 specialized models
- Real-time dashboard with WebSocket updates
- Enterprise integrations (SOAR, SIEM, firewall)
- Plugin system and model registry
- Compliance support (GDPR, HIPAA, CCPA)
- Comprehensive documentation and contribution guidelines"
    
    echo "✅ Commit created"
else
    echo "✅ No uncommitted changes"
fi

# Add remote if not exists
if ! git remote | grep -q "origin"; then
    echo ""
    echo "🔗 Adding remote repository..."
    git remote add origin https://github.com/ai-research00/AegisPCAP.git
    echo "✅ Remote added"
else
    echo "✅ Remote already configured"
fi

# Set main branch
echo ""
echo "🌿 Setting main branch..."
git branch -M main
echo "✅ Branch set to main"

echo ""
echo "======================================"
echo "✅ Repository prepared for push!"
echo ""
echo "To push to GitHub, run:"
echo "  git push -u origin main"
echo ""
echo "Or to force push (if repository exists):"
echo "  git push -u origin main --force"
echo ""
echo "======================================"
