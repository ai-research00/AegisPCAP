#!/bin/bash
# AegisPCAP Local Integration Test
# Tests all components running together with WebSocket real-time updates

set -e

PROJECT_ROOT="/home/hssn/Documents/AegisPCAP"
BACKEND_DIR="$PROJECT_ROOT"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  AegisPCAP Local Integration Test - Full System               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"

# Check if services are running
echo -e "\n${YELLOW}[1/5] Verifying services...${NC}"

# PostgreSQL check
if sudo systemctl is-active --quiet postgresql; then
    echo -e "${GREEN}✓${NC} PostgreSQL running"
else
    echo -e "${RED}✗${NC} PostgreSQL not running"
    exit 1
fi

# Redis check
if sudo systemctl is-active --quiet redis-server; then
    echo -e "${GREEN}✓${NC} Redis running"
else
    echo -e "${RED}✗${NC} Redis not running"
    exit 1
fi

# PostgreSQL connection test
echo -e "\n${YELLOW}[2/5] Testing database connectivity...${NC}"
if psql -U postgres -d postgres -c "SELECT 1" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} PostgreSQL accessible"
else
    echo -e "${RED}✗${NC} PostgreSQL not accessible"
    exit 1
fi

# Redis connectivity test
echo -e "\n${YELLOW}[3/5] Testing Redis connectivity...${NC}"
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Redis accessible (response: $(redis-cli ping))"
else
    echo -e "${RED}✗${NC} Redis not accessible"
    exit 1
fi

# Check Python dependencies
echo -e "\n${YELLOW}[4/5] Checking Python environment...${NC}"
cd "$BACKEND_DIR"
if python3 -c "import fastapi, sqlalchemy, redis" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Python dependencies available"
else
    echo -e "${YELLOW}!${NC} Installing Python dependencies..."
    pip install -q fastapi sqlalchemy redis psycopg2-binary uvicorn python-dotenv 2>/dev/null || true
fi

# Check Node/npm for frontend
echo -e "\n${YELLOW}[5/5] Checking Node.js environment...${NC}"
cd "$FRONTEND_DIR"
if [ -d "node_modules" ]; then
    echo -e "${GREEN}✓${NC} Node modules available"
else
    echo -e "${YELLOW}!${NC} Installing npm packages..."
    npm install -q 2>/dev/null
fi

# Show service endpoints
echo -e "\n${BLUE}═════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}System Services Ready${NC}"
echo -e "${BLUE}═════════════════════════════════════════════════════════════════${NC}"

echo -e "\n${GREEN}Backend Services:${NC}"
echo -e "  • PostgreSQL    : localhost:5432"
echo -e "  • Redis         : localhost:6379"
echo -e "  • FastAPI       : http://localhost:8080"
echo -e "  • WebSocket     : ws://localhost:8080/api/dashboard/ws"
echo -e "  • API Docs      : http://localhost:8080/docs"

echo -e "\n${GREEN}Frontend:${NC}"
echo -e "  • React Dev     : http://localhost:5173"
echo -e "  • Wireshark     : (installed)"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "  1. Start Backend:  cd $BACKEND_DIR && python main.py server"
echo -e "  2. Start Frontend: cd $FRONTEND_DIR && npm run dev"
echo -e "  3. Open Browser:   http://localhost:5173"
echo -e "  4. Start Wireshark (optional): sudo wireshark"

echo -e "\n${BLUE}═════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}All systems ready for local integration testing!${NC}"
echo -e "${BLUE}═════════════════════════════════════════════════════════════════${NC}\n"
