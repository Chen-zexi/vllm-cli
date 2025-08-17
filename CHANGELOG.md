# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-08-17

### Fixed
- **Critical**: Fixed package installation issue - setuptools now correctly includes all sub-packages

## [0.2.0] - 2025-08-17

### Added
- **LoRA Adapter Support**: Serve models with LoRA adapters - select base model and multiple LoRA adapters for serving
- **Enhanced Model List Display**: Comprehensive model listing showing HuggingFace models, LoRA adapters, and datasets with size information
- **Model Directory Management**: Configure and manage custom model directories for automatic model discovery
- **Model Caching**: Performance optimization through intelligent caching with TTL for model listings
- **Improved Model Discovery**: Integration with hf-model-tool for comprehensive model detection with fallback mechanisms
- **HuggingFace Token Support**: Authentication support for accessing gated models with automatic token validation
- **Profile Management Enhancements**:
  - View/Edit profiles in unified interface with detailed configuration display
  - Direct editing of built-in profiles with user overrides
  - Reset customized built-in profiles to defaults

### Changed
- Refactored model management system with new `models/` package structure
- Enhanced error handling with comprehensive error recovery strategies
- Improved configuration validation framework with type checking and schemas
- Updated low_memory profile to use FP8 quantization instead of bitsandbytes

### Fixed
- Better handling of model metadata extraction
- Improved error messages for better user experience

## [0.1.1] - 2025-08-15

### Added
- Display complete log viewer when server startup fails
- Enhanced error handling and recovery options

### Fixed
- Small UI fixes for better terminal display
- Improved error messages clarity

## [0.1.0] - 2025-08-14

### Added
- **Interactive Mode**: Rich terminal interface with menu-driven navigation
- **Command-Line Mode**: Direct CLI commands for automation and scripting
- **Model Management**: Automatic discovery and management of local models
- **Remote Model Support**: Serve models directly from HuggingFace Hub without pre-downloading
- **Configuration Profiles**: Pre-configured server profiles (standard, moe_optimized, high_throughput, low_memory)
- **Custom Profiles**: User-defined configuration profiles support
- **Server Monitoring**: Real-time monitoring of active vLLM servers with GPU utilization
- **System Information**: GPU, memory, and CUDA compatibility checking
- **Quick Serve**: Auto-reuse last successful configuration
- **Process Management**: Global server registry with automatic cleanup on exit
- **Schema-Driven Configuration**: JSON schemas for validation of vLLM arguments
- **ASCII Fallback**: Environment detection for terminal compatibility

### Dependencies
- vLLM
- PyTorch with CUDA support
- hf-model-tool for model discovery
- Rich for terminal UI
- Inquirer for interactive prompts
- psutil for system monitoring
- PyYAML for configuration parsing

[0.2.0]: https://github.com/Chen-zexi/vllm-cli/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Chen-zexi/vllm-cli/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Chen-zexi/vllm-cli/releases/tag/v0.1.0
