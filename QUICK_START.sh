#!/bin/bash
# Quick Start Script - Orpheus Hindi TTS
# Complete setup and deployment in one script

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════╗"
echo "║   Orpheus Hindi TTS - Quick Start                     ║"
echo "║   Production Banking Voice AI                        ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check prerequisites
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker found: $(docker --version)${NC}"

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}✗ Docker Compose is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose found: $(docker compose version)${NC}"

if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ NVIDIA GPU tools not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ GPU found:${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Clone or update repository
echo -e "\n${YELLOW}[2/6] Setting up repository...${NC}"
if [ ! -d ".git" ]; then
    echo "Cloning repository..."
    # git clone would go here if needed
fi
echo -e "${GREEN}✓ Repository ready${NC}"

# Create environment file
echo -e "\n${YELLOW}[3/6] Configuring environment...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ Environment file created (.env)${NC}"
else
    echo -e "${GREEN}✓ Environment file already exists${NC}"
fi

# Create necessary directories
echo -e "\n${YELLOW}[4/6] Creating directories...${NC}"
mkdir -p models outputs static templates
echo -e "${GREEN}✓ Directories created${NC}"

# Download model
echo -e "\n${YELLOW}[5/6] Downloading model (one-time, ~3.5GB)...${NC}"
echo "This may take 5-10 minutes depending on your internet speed..."
if docker compose -f docker-compose-gpu.yml --profile init run model-init; then
    echo -e "${GREEN}✓ Model downloaded successfully${NC}"
else
    echo -e "${RED}✗ Model download failed${NC}"
    echo "Try manual download:"
    echo "  mkdir -p models"
    echo "  cd models"
    echo "  huggingface-cli download lex-au Orpheus-3b-Hindi-FT-Q8_0.gguf"
    exit 1
fi

# Start services
echo -e "\n${YELLOW}[6/6] Starting services...${NC}"
echo "Starting inference server and FastAPI..."
docker compose -f docker-compose-gpu.yml up -d

echo -e "${GREEN}✓ Services starting${NC}"

# Wait for services to be ready
echo -e "\n${YELLOW}Waiting for services to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:5005/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Services are ready!${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

if ! curl -s http://localhost:5005/health > /dev/null 2>&1; then
    echo -e "${RED}✗ Services failed to start${NC}"
    echo "Check logs with: docker compose logs -f"
    exit 1
fi

# Display summary
echo -e "\n${GREEN}"
echo "╔═══════════════════════════════════════════════════════╗"
echo "║   ✓ Setup Complete!                                   ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}API Endpoints:${NC}"
echo "  Web UI:  ${GREEN}http://localhost:5005${NC}"
echo "  Docs:    ${GREEN}http://localhost:5005/docs${NC}"
echo "  Health:  ${GREEN}http://localhost:5005/health${NC}"

echo -e "\n${BLUE}Quick Test:${NC}"
echo -e "${YELLOW}Generate Hindi speech:${NC}"
echo "  curl -X POST http://localhost:5005/v1/audio/speech \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"input\": \"नमस्ते\", \"voice\": \"ऋतिका\"}' \\"
echo "    --output test.wav"

echo -e "\n${BLUE}Run Examples:${NC}"
echo "  python examples/basic_usage.py"

echo -e "\n${BLUE}View Logs:${NC}"
echo "  docker compose -f docker-compose-gpu.yml logs -f"

echo -e "\n${BLUE}Stop Services:${NC}"
echo "  docker compose -f docker-compose-gpu.yml down"

echo -e "\n${YELLOW}Documentation:${NC}"
echo "  - README.md - Complete documentation"
echo "  - SETUP_GUIDE.md - Detailed setup"
echo "  - BANKING_EXAMPLES.md - Hindi banking examples"

echo -e "\n${GREEN}Ready to use Orpheus Hindi TTS!${NC}\n"
