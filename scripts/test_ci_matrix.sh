#!/bin/bash
# Test CI/CD with Python version matrix (simulates GitHub Actions matrix)
# This requires conda or pyenv to test multiple Python versions

set -e

echo "========================================="
echo "🔧 CI/CD Matrix Test for vLLM CLI"
echo "========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Python versions to test (matching your CI/CD matrix)
PYTHON_VERSIONS=("3.9" "3.10" "3.11" "3.12")

# Check if conda is available
if command -v conda &> /dev/null; then
    PYTHON_MANAGER="conda"
    echo -e "${BLUE}Using conda for Python version management${NC}"
elif command -v pyenv &> /dev/null; then
    PYTHON_MANAGER="pyenv"
    echo -e "${BLUE}Using pyenv for Python version management${NC}"
else
    echo -e "${YELLOW}Warning: Neither conda nor pyenv found.${NC}"
    echo "Testing with current Python only: $(python --version)"
    PYTHON_VERSIONS=($(python --version | cut -d' ' -f2 | cut -d'.' -f1,2))
fi

# Track results
RESULTS=""
ALL_PASSED=true

# Test each Python version
for PY_VERSION in "${PYTHON_VERSIONS[@]}"; do
    echo ""
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}Testing with Python $PY_VERSION${NC}"
    echo -e "${BLUE}=========================================${NC}"

    # Create/activate environment based on manager
    if [ "$PYTHON_MANAGER" == "conda" ]; then
        ENV_NAME="vllm-test-py${PY_VERSION//./}"

        # Check if environment exists, create if not
        if ! conda env list | grep -q "$ENV_NAME"; then
            echo "Creating conda environment $ENV_NAME with Python $PY_VERSION..."
            conda create -n "$ENV_NAME" python="$PY_VERSION" -y -q
        fi

        # Activate and test
        echo "Activating $ENV_NAME..."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate "$ENV_NAME"

    elif [ "$PYTHON_MANAGER" == "pyenv" ]; then
        # Install Python version if not available
        if ! pyenv versions | grep -q "$PY_VERSION"; then
            echo "Installing Python $PY_VERSION with pyenv..."
            pyenv install "$PY_VERSION"
        fi
        pyenv local "$PY_VERSION"
    fi

    # Verify Python version
    ACTUAL_VERSION=$(python --version | cut -d' ' -f2)
    echo "Python version: $ACTUAL_VERSION"

    # Install package and dependencies
    echo "Installing package and dependencies..."
    pip install -q -e . || exit 1
    pip install -q pytest flake8 black || exit 1

    # Run tests
    echo -e "\n${YELLOW}Running tests...${NC}"
    if pytest tests/ -q; then
        echo -e "${GREEN}✅ Tests passed for Python $PY_VERSION${NC}"
        TEST_RESULT="PASS"
    else
        echo -e "${RED}❌ Tests failed for Python $PY_VERSION${NC}"
        TEST_RESULT="FAIL"
        ALL_PASSED=false
    fi

    # Run flake8
    echo -e "\n${YELLOW}Running flake8...${NC}"
    if flake8 . --quiet; then
        echo -e "${GREEN}✅ Linting passed for Python $PY_VERSION${NC}"
        LINT_RESULT="PASS"
    else
        echo -e "${RED}❌ Linting failed for Python $PY_VERSION${NC}"
        LINT_RESULT="FAIL"
        ALL_PASSED=false
    fi

    # Store results
    RESULTS="$RESULTS\nPython $PY_VERSION: Tests=$TEST_RESULT, Linting=$LINT_RESULT"

    # Deactivate environment if using conda
    if [ "$PYTHON_MANAGER" == "conda" ]; then
        conda deactivate
    fi
done

# Summary
echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}📊 CI/CD Matrix Test Summary${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "$RESULTS"
echo ""

if [ "$ALL_PASSED" = true ]; then
    echo -e "${GREEN}🎉 All Python versions passed CI/CD checks!${NC}"
    echo "Your code is ready for GitHub Actions."
    exit 0
else
    echo -e "${RED}⚠️  Some Python versions failed CI/CD checks.${NC}"
    echo "Please fix the issues before pushing."
    exit 1
fi
