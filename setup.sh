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

# Check Python version (require 3.10+)
echo "Checking Python version..."
PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
REQUIRED_MAJOR=3
REQUIRED_MINOR=10

MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if (( MAJOR < REQUIRED_MAJOR || (MAJOR == REQUIRED_MAJOR && MINOR < REQUIRED_MINOR) )); then
    echo -e "${RED}❌ Python 3.10+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python $PYTHON_VERSION${NC}"

# Create core directories (only those that aren't tracked)
echo "Creating project directories..."
mkdir -p data/{raw,interim,processed}
mkdir -p logs mlflow
touch data/raw/.gitkeep
touch data/interim/.gitkeep
touch data/processed/.gitkeep
touch logs/.gitkeep
echo -e "${GREEN}✅ Data and logs directories ensured${NC}"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if command -v uv &>/dev/null; then
    echo "Using uv..."
    uv venv
    source .venv/bin/activate
else
    echo "Using Python venv..."
    python3 -m venv venv
    source venv/bin/activate
fi
echo -e "${GREEN}✅ Virtual environment ready${NC}"

# Install dependencies
echo ""
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    if command -v uv &>/dev/null; then
        uv pip install -r requirements.txt
    else
        pip install -r requirements.txt
    fi
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ requirements.txt not found. Skipping.${NC}"
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
echo "Verifying installation..."
python3 << 'EOF'
try:
    import pandas, prefect, mlflow, pandera, duckdb, sqlalchemy
    print("✅ Package verification passed")
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
echo "  source .venv/bin/activate"
echo "  python -m churn_compass.pipelines.ingest_pipeline --demo"
echo ""
echo "Happy coding 🚀"
