# Testing Guide for vLLM CLI

This guide explains how to test the vLLM CLI project locally to ensure your changes will pass CI/CD.

## Quick Start

Run all CI/CD checks locally:

```bash
make ci-test
```

This runs the same checks as GitHub Actions:
1. Code formatting check (Black)
2. Linting (Flake8)
3. Unit tests (Pytest)
4. Import verification

## Available Testing Commands

### Using Make (Recommended)

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Check code formatting (no changes)
make format-check

# Auto-format code
make format

# Run linting
make lint

# Run type checking
make type-check

# Clean build artifacts
make clean

# Run full CI/CD suite
make ci-test

# Run all checks (format, lint, test)
make all
```

### Using Test Scripts

```bash
# Run full CI/CD test suite
./scripts/test_ci_locally.sh

# Test with multiple Python versions (requires conda/pyenv)
./scripts/test_ci_matrix.sh

# Install dependencies first
./scripts/test_ci_locally.sh --install

# Or use Make commands
make ci-local    # Run comprehensive local CI script
make ci-matrix   # Test with multiple Python versions
```

## Manual Testing

### 1. Code Formatting

```bash
# Check formatting without changes
black src/ tests/ --line-length 88 --check

# Auto-format code
black src/ tests/ --line-length 88
```

### 2. Linting

```bash
# Run flake8 with project configuration
flake8 .

# Run with specific line length
flake8 . --max-line-length=88
```

### 3. Unit Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/vllm_cli --cov-report=term-missing

# Run specific test file
pytest tests/test_config_manager.py

# Run specific test
pytest tests/test_config_manager.py::TestConfigManager::test_init_creates_config_dir
```

### 4. Type Checking (Optional)

```bash
# Run mypy
mypy src/vllm_cli --python-version 3.9 --ignore-missing-imports
```

### 5. Security Checks (Optional)

```bash
# Install bandit
pip install bandit

# Run security scan
bandit -r src/ -ll
```

## Pre-Commit Hooks

Set up pre-commit hooks to automatically check your code before commits:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Run on staged files (happens automatically on commit)
git commit -m "your message"
```

## Testing Different Python Versions

The project supports Python 3.9, 3.10, 3.11, and 3.12.

### Using Conda

```bash
# Create environments for testing
conda create -n vllm-py39 python=3.9
conda create -n vllm-py310 python=3.10
conda create -n vllm-py311 python=3.11
conda create -n vllm-py312 python=3.12

# Test with specific version
conda activate vllm-py39
pip install -e .
pytest tests/
```

### Using the Matrix Test Script

```bash
# Automatically tests all Python versions
./scripts/test_ci_matrix.sh
# Or use Make command
make ci-matrix
```

## CI/CD Configuration

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs:

1. **Test Matrix**: Python 3.9, 3.10, 3.11, 3.12
2. **Linting**: Flake8 (informational only)
3. **Formatting**: Black check (informational only)
4. **Type Checking**: MyPy (if configured)

## Common Issues and Solutions

### Issue: Flake8 Line Too Long (E501)

**Solution**: The project uses 88-character line length. Long strings in specific files are ignored via `.flake8` configuration.

### Issue: Import Errors

**Solution**: Ensure the package is installed in development mode:
```bash
pip install -e .
```

### Issue: Test Failures

**Solution**: Check test output for specific errors:
```bash
pytest tests/ -v --tb=short
```

### Issue: Black Formatting Conflicts

**Solution**: Always use the project's Black configuration:
```bash
black src/ tests/ --line-length 88
```

## Development Dependencies

Install all development dependencies:

```bash
pip install -r requirements-dev.txt
```

Or install individually:

```bash
pip install pytest pytest-cov flake8 black autoflake mypy
```

## Continuous Integration

Before pushing to GitHub:

1. Run `make ci-test` to simulate CI/CD
2. Fix any issues that arise
3. Commit your changes
4. Push to GitHub

The GitHub Actions workflow will automatically run on:
- Push to main branch
- Pull requests
- Manual trigger

## Tips

1. **Use Make commands**: They ensure consistency with CI/CD
2. **Run tests frequently**: Catch issues early
3. **Enable pre-commit hooks**: Automatic checking before commits
4. **Check multiple Python versions**: Ensure compatibility
5. **Monitor CI/CD results**: Check GitHub Actions tab after pushing

## Getting Help

- Check test output carefully for error messages
- Review `.github/workflows/ci.yml` for exact CI/CD configuration
- Run `make help` for available commands
- Check `pyproject.toml` for project configuration
