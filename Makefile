.PHONY: help test lint format clean install ci-test all

# Default target
help:
	@echo "vLLM CLI Development Commands"
	@echo "=============================="
	@echo "make install      - Install package in development mode"
	@echo "make test        - Run unit tests"
	@echo "make test-cov    - Run tests with coverage report"
	@echo "make lint        - Run linting checks (flake8)"
	@echo "make format      - Format code with black"
	@echo "make format-check - Check code formatting without changes"
	@echo "make type-check  - Run type checking with mypy"
	@echo "make clean       - Remove build artifacts and caches"
	@echo "make ci-test     - Run full CI/CD test suite locally"
	@echo "make ci-local    - Run comprehensive local CI script"
	@echo "make ci-matrix   - Test with multiple Python versions"
	@echo "make all         - Run format, lint, and test"
	@echo "make pre-commit  - Run pre-commit checks"

# Install the package in development mode
install:
	pip install -e .
	pip install -r requirements-dev.txt || pip install pytest flake8 black

# Run tests
test:
	pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	pytest tests/ --cov=src/vllm_cli --cov-report=term-missing --cov-report=html

# Run linting
lint:
	flake8 .

# Format code
format:
	isort src/ tests/ --profile black --line-length 88
	black src/ tests/ --line-length 88
	@echo "✅ Code formatted"

# Check formatting without making changes
format-check:
	isort src/ tests/ --profile black --line-length 88 --check --diff
	black src/ tests/ --line-length 88 --check --diff

# Type checking
type-check:
	mypy src/vllm_cli --python-version 3.9 --ignore-missing-imports || echo "⚠️  MyPy not configured or installed"

# Clean build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true
	@echo "✅ Cleaned build artifacts"

# Run the same checks as CI/CD
ci-test:
	@echo "🔧 Running CI/CD Test Suite"
	@echo "============================"
	@echo "1️⃣  Formatting check..."
	@isort src/ tests/ --profile black --line-length 88 --check || (echo "❌ Import order check failed. Run 'make format' to fix" && exit 1)
	@black src/ tests/ --line-length 88 --check || (echo "❌ Format check failed. Run 'make format' to fix" && exit 1)
	@echo "✅ Format check passed"
	@echo ""
	@echo "2️⃣  Linting..."
	@flake8 . || (echo "❌ Linting failed" && exit 1)
	@echo "✅ Linting passed"
	@echo ""
	@echo "3️⃣  Running tests..."
	@pytest tests/ -v || (echo "❌ Tests failed" && exit 1)
	@echo "✅ All tests passed"
	@echo ""
	@echo "4️⃣  Import check..."
	@python -c "import vllm_cli" || (echo "❌ Import failed" && exit 1)
	@echo "✅ Import check passed"
	@echo ""
	@echo "🎉 All CI/CD checks passed!"

# Run local CI test script
ci-local:
	@./scripts/test_ci_locally.sh

# Test with multiple Python versions
ci-matrix:
	@./scripts/test_ci_matrix.sh

# Run all checks
all: format lint test
	@echo "✅ All checks completed"

# Quick test for pre-commit
pre-commit: format-check lint test
	@echo "✅ Pre-commit checks passed"
