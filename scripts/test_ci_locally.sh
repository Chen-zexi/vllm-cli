#!/bin/bash
# Local CI/CD Test Script
# This script mimics the GitHub Actions CI workflow to test locally

set -e  # Exit on any error

echo "========================================="
echo "🔧 Local CI/CD Test for vLLM CLI"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track failures
FAILED_TESTS=""

# Function to run a test and track results
run_test() {
    local test_name=$1
    local test_command=$2

    echo -e "${YELLOW}Running: $test_name${NC}"
    echo "Command: $test_command"
    echo "----------------------------------------"

    if eval $test_command; then
        echo -e "${GREEN}✅ $test_name PASSED${NC}\n"
    else
        echo -e "${RED}❌ $test_name FAILED${NC}\n"
        FAILED_TESTS="$FAILED_TESTS\n  - $test_name"
        # Don't exit immediately, continue with other tests
    fi
}

# 1. Python Version Check
echo -e "${YELLOW}📋 Environment Information${NC}"
echo "Python version: $(python --version)"
echo "Current environment: $CONDA_DEFAULT_ENV"
echo ""

# 2. Install Dependencies (optional, skip if already installed)
if [ "$1" == "--install" ]; then
    run_test "Install dependencies" "pip install -e . && pip install pytest pytest-cov flake8 black mypy"
fi

# 3. Run Tests
run_test "Unit Tests" "pytest tests/ -v --tb=short"

# 4. Check Test Coverage (optional but informative)
if python -c "import pytest_cov" 2>/dev/null; then
    run_test "Test Coverage" "pytest tests/ --cov=src/vllm_cli --cov-report=term-missing --cov-fail-under=50"
else
    echo -e "${YELLOW}⚠️  Skipping coverage (pytest-cov not installed)${NC}\n"
fi

# 5. Linting with flake8
run_test "Flake8 Linting" "flake8 ."

# 6. Code Formatting Check with Black
run_test "Black Formatting Check" "black src/ tests/ --check --diff"

# 7. Type Checking with mypy (if configured)
if [ -f "mypy.ini" ] || [ -f "pyproject.toml" ]; then
    run_test "MyPy Type Checking" "mypy src/vllm_cli --python-version 3.9 --ignore-missing-imports"
else
    echo -e "${YELLOW}⚠️  Skipping mypy (no configuration found)${NC}\n"
fi

# 8. Import Sort Check (if isort is available)
if command -v isort &> /dev/null; then
    run_test "Import Sorting" "isort src/ tests/ --check-only --diff"
else
    echo -e "${YELLOW}⚠️  Skipping isort (not installed)${NC}\n"
fi

# 9. Security Check (if bandit is available)
if command -v bandit &> /dev/null; then
    run_test "Security Check" "bandit -r src/ -ll"
else
    echo -e "${YELLOW}⚠️  Skipping bandit security check (not installed)${NC}\n"
fi

# 10. Check that the package can be imported
run_test "Package Import Test" "python -c 'import vllm_cli; print(\"Version:\", vllm_cli.__version__)'"

# 11. Check CLI help works
run_test "CLI Help Test" "python -m vllm_cli --help > /dev/null"

# 12. Validate pyproject.toml
if [ -f "pyproject.toml" ]; then
    run_test "Validate pyproject.toml" "python -c '
try:
    import tomllib
    with open(\"pyproject.toml\", \"rb\") as f:
        tomllib.load(f)
except ImportError:
    import toml
    toml.load(\"pyproject.toml\")
print(\"pyproject.toml is valid\")
'"
fi

# Summary
echo ""
echo "========================================="
echo "📊 Test Summary"
echo "========================================="

if [ -z "$FAILED_TESTS" ]; then
    echo -e "${GREEN}✅ All tests passed! Your code is ready for CI/CD.${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed:${NC}"
    echo -e "$FAILED_TESTS"
    echo ""
    echo "Please fix these issues before pushing to GitHub."
    exit 1
fi
