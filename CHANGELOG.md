# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.2.5] - 2025-08-25

### Added
- **Multi-Model Proxy Server (Experimental)**: Enabling multiple LLMs through a single unified API endpoint
  - Single OpenAI-compatible endpoint for all models
  - Request routing based on model name
  - Save and reuse proxy configurations
- **Dynamic Model Management**: Add or remove models at runtime without restarting the proxy
  - Live model registration and unregistration
  - Pre-registration with verification lifecycle
  - Graceful handling of model failures without affecting other models
  - Model state tracking (pending, running, sleeping, stopped)
- **Model Sleep/Wake for GPU Memory Management**: Efficient GPU resource distribution
  - Sleep Level 1: CPU offload for faster wake-up
  - Sleep Level 2: Full memory discard for maximum savings
  - Real-time memory usage tracking and reporting
  - Models maintain their ports while sleeping
- **Test Coverage**: Added comprehensive tests for multi-model proxy and model registry

### Changed
- Improved error handling with detailed logs when PyTorch is not installed
- Better server cleanup and process management

### Fixed
- UI navigation improvements and minor display fixes

## [v0.2.4] - 2025-08-20

### Added
- **Hardware-Optimized Profiles for GPT-OSS Models**: New built-in profiles optimized for different GPU architectures
  - `gpt_oss_ampere`: Optimized for NVIDIA A100 GPUs
  - `gpt_oss_hopper`: Optimized for NVIDIA H100/H200 GPUs
  - `gpt_oss_blackwell`: Optimized for NVIDIA Blackwell (B100/B200) GPUs
  - Based on official [vLLM GPT recipes](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)
- **Shortcuts System**: Save and quickly launch model + profile combinations
  - Quick launch from CLI: `vllm-cli serve --shortcut NAME`
  - Manage shortcuts through interactive mode or CLI commands
  - Import/export shortcuts for sharing configurations
- **Ollama Model Support**: Full integration with Ollama-downloaded models
  - Automatic discovery in user (`~/.ollama`) and system (`/usr/share/ollama`) directories
  - GGUF format detection and experimental serving support
- **Environment Variable Management**: Two-tier system for complete control
  - Universal environment variables for all servers
  - Profile-specific environment variables (override universal)
  - Clear indication of environment sources when launching
- **GPU Selection**: Select specific GPUs for model serving
  - CLI: `--device 0,1` to use specific GPUs
  - Interactive UI for GPU selection in advanced settings
  - Automatic tensor_parallel_size adjustment
- **Enhanced System Information**: vLLM built-in feature detection
  - Detailed attention backend availability (Flash Attention 2/3, xFormers)
  - Feature compatibility checking per backend
- **Server Cleanup Control**: Configure server behavior on CLI exit
- **Extended vLLM Arguments**: Added 16+ new arguments for v1 engine
  - Performance, optimization, API, configuration, and monitoring options

### Changed
- Enhanced Quick Serve menu shows last configuration and saved shortcuts
- Model field excluded from profiles for model-agnostic templates
- Model cache refresh properly respects TTL settings (>60s)
- Environment variables available in Custom Configuration menu

### Fixed
- Fixed manual cache refresh functionality
- Fixed profile creation inconsistency between menus
- Fixed UI consistency issues with prompt formatting


## [v0.2.3] - 2025-08-17

### Fixed
- **Critical**: Fixed missing built-in profiles when installing from PyPI - JSON schema files are now properly included in the package distribution

## [v0.2.2] - 2025-08-17

### Added
- **Model Manifest Support**: Introduced `models_manifest.json` for mapping custom models in vLLM CLI native way (see [custom-model-serving.md](docs/custom-model-serving.md) for more details)
- **Documentation**: Added [custom-model-serving.md](docs/custom-model-serving.md) for custom model serving guide

### Fixed
- Serving models from custom directories now works as expected
- Fixed some UI issues


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

[0.2.2]: https://github.com/Chen-zexi/vllm-cli/compare/0.2.1...v0.2.2
[0.2.1]: https://github.com/Chen-zexi/vllm-cli/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/Chen-zexi/vllm-cli/compare/0.1.1...0.2.0
[0.1.1]: https://github.com/Chen-zexi/vllm-cli/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/Chen-zexi/vllm-cli/releases/tag/0.1.0
