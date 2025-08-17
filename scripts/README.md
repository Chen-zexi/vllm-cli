# Testing Scripts

CI/CD and testing scripts for vLLM CLI.

## Scripts

- **test_ci_locally.sh** - Run full CI/CD test suite locally
- **test_ci_matrix.sh** - Test with Python 3.9, 3.10, 3.11, 3.12

## Usage

```bash
make ci-local    # Run comprehensive CI tests
make ci-matrix   # Test all Python versions
```

See [docs/TESTING.md](../docs/TESTING.md) for detailed testing documentation.
