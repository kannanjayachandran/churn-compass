#!/bin/bash
# Churn Compass - Automated Setup Script

set -e

echo "=================================="
echo "Churn Compass - Setup Script"
echo "=================================="
echo ""

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
REQUIRED_MAJOR=3
REQUIRED_MINOR=13

MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if (( MAJOR < REQUIRED_MAJOR || (MAJOR == REQUIRED_MAJOR && MINOR < REQUIRED_MINOR) )); then
    if command -v uv &>/dev/null; then
        echo -e "${YELLOW}⚠ System Python ($PYTHON_VERSION) is older than recommended (3.13).${NC}"
        echo -e "${YELLOW}  uv will be used to manage the correct Python version.${NC}"
    else
        echo -e "${RED}❌ Python 3.13+ required. Found: $PYTHON_VERSION${NC}"
        echo "Please install uv or update Python."
        exit 1
    fi
else
    echo -e "${GREEN}✅ Python $PYTHON_VERSION${NC}"
fi

# Create core directories
echo "Creating project directories..."
mkdir -p data/{raw,interim,processed,scored}
mkdir -p logs mlruns
touch data/raw/.gitkeep
touch data/interim/.gitkeep
touch data/processed/.gitkeep
touch data/scored/.gitkeep
touch logs/.gitkeep
echo -e "${GREEN}✅ Project directories ensured${NC}"

# Setup Backend (Python)
echo ""
echo "Setting up Backend (Python)..."
if command -v uv &>/dev/null; then
    echo "Using uv for dependency management..."
    uv sync
    echo -e "${GREEN}✅ Backend setup complete with uv${NC}"
else
    echo -e "${YELLOW}⚠ uv not found. Falling back to standard venv (not recommended)${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    echo -e "${GREEN}✅ Backend setup complete with venv${NC}"
fi

# Setup Frontend (React)
echo ""
echo "Setting up Frontend (React)..."
if [ -d "frontend" ]; then
    cd frontend
    if command -v bun &>/dev/null; then
        echo "Using bun for frontend dependencies..."
        bun install
        echo -e "${GREEN}✅ Frontend dependencies installed with bun${NC}"
    else
        echo -e "${RED}❌ bun is not installed. Frontend setup failed.${NC}"
        echo "Please install bun: https://bun.sh/"
    fi
    cd ..
else
    echo -e "${YELLOW}⚠ frontend directory not found. Skipping.${NC}"
fi

# Copy environment file
echo ""
echo "Setting up .env..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ .env created${NC}"
    else
        echo -e "${YELLOW}⚠ .env.example missing${NC}"
    fi
else
    echo -e "${YELLOW}⚠ .env already exists${NC}"
fi

# Verify core packages
echo ""
echo "Verifying Backend installation..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python3 << 'EOF'
try:
    import pandas, prefect, mlflow, pandera, duckdb, sqlalchemy, xgboost, psutil, fastapi, uvicorn
    print("✅ Package verification passed (FastAPI included)")
except ImportError as e:
    print(f"❌ Missing package: {e}")
    exit(1)
EOF

echo ""
echo "=================================="
echo -e "${GREEN}Setup Complete! 🎉${NC}"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Start the Backend API:"
echo "     uv run uvicorn churn_compass.api.main:app --reload"
echo ""
echo "  2. Start the Frontend:"
echo "     cd frontend && bun run dev"
echo ""
echo "  3. (Optional) Run demo ingestion:"
echo "     uv run python -m churn_compass.pipelines.ingest_pipeline --demo"
echo ""
echo "Happy coding 🚀"
