# vLLM CLI

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A command-line interface tool for serving Large Language Models using vLLM. Provides both interactive and command-line modes with features for configuration profiles, model management, and server monitoring.

**Quick Links:** [📚 Documentation](#documentation) | [🗺️ Roadmap](#roadmap)

![vLLM CLI Welcome Screen](asset/welcome-screen.png)
*Welcome screen showing GPU status and system overview*

## Features

- **Interactive Mode**: Rich terminal interface with menu-driven navigation
- **Command-Line Mode**: Direct CLI commands for automation and scripting
- **Model Management**: Automatic discovery and management of local models
- **Ollama Model Support**: Discover and serve Ollama-downloaded GGUF models (experimental)
- **Remote Model Support**: Serve models directly from HuggingFace Hub without pre-downloading
- **Configuration Profiles**: Pre-configured and custom server profiles
- **Server Monitoring**: Real-time monitoring of active vLLM servers
- **System Information**: GPU, memory, and CUDA compatibility checking
- **Log Viewer**: View the complete log file when server startup fails

## What's New in v0.2.3

- **Model Manifest Support**: Map custom models in vLLM CLI native way with `models_manifest.json`
- **Documentation**: New [custom model serving guide](docs/custom-model-serving.md) for serving models from custom directories
- **Bug Fixes**: Fixed serving models from custom directories and various UI improvements

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes and version history.

### LoRA Adapter Support
![LoRA Serving](asset/lora-serving.png)
*Serve models with LoRA adapters - select base model and multiple LoRA adapters for serving*

### Enhanced Model List Display
![Model List Display](asset/model-list-display.png)
*Comprehensive model listing showing HuggingFace models, LoRA adapters, and datasets with size information*

### Model Directory Management
![Model Directory Management](asset/model-directory-management.png)
*Configure and manage custom model directories for automatic model discovery*


## Installation

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)
- vLLM package installed

### Install from PyPI

```bash
pip install vllm-cli
```

### Build from source

```bash
# Clone the repository
git clone https://github.com/Chen-zexi/vllm-cli.git
cd vllm-cli

# Activate the environment you have vLLM installed in

# Install dependencies
pip install -r requirements.txt
pip install hf-model-tool

# Install CLI in development mode
pip install -e .
```

## Important Notice

### Model Compatibility and Troubleshooting

⚠️ **Model and GPU Compatibility**: Model support and available arguments can vary significantly depending on:
- The specific model architecture and requirements
- Your GPU device capabilities (compute capability, memory, etc.)
- vLLM version and supported features

If you encounter issues when serving a model:
1. **Check the server logs** - vLLM provides detailed error messages that indicate missing requirements or incompatible settings
2. **Consult the official vLLM documentation** - Visit [vLLM docs](https://docs.vllm.ai/) for model-specific requirements and supported features
3. **Review model requirements** - Some models require specific arguments or particular quantization methods

### Model Not Showing Up?

If your models aren't appearing in the serving menu, see our [Model Discovery Quick Reference](docs/MODEL_DISCOVERY_QUICK_REF.md) for quick fixes, or the [Model Discovery Flow](docs/MODEL_DISCOVERY_FLOW.md) for technical details.

### Model Management with hf-model-tool

vLLM CLI uses [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for local model discovery and management. This is another tool I developed for model management. It provides:
- Comprehensive model scanning across HuggingFace cache, Ollama directories, and custom locations
- Ollama model support with GGUF format detection
- Detailed model information including size, type, and quantization
- Shared configuration between vLLM CLI and hf-model-tool

**Settings are synchronized** - Any model directories configured in hf-model-tool will automatically be available in vLLM CLI, and vice versa. This includes Ollama scanning settings and custom directories.

#### Ollama Model Support

vLLM CLI can discover and serve models downloaded via Ollama:
- Models are automatically discovered from both user (`~/.ollama`) and system (`/usr/share/ollama`) directories
- GGUF format models have experimental support in vLLM 0.5.0+
- See [Ollama Integration Guide](docs/ollama-integration.md) for detailed information

We encourage you to explore hf-model-tool for advanced model management capabilities. You can also launch it directly within vLLM CLI.

```bash
# Install hf-model-tool (already included with vLLM CLI)
pip install --upgrade hf-model-tool

# Scan and manage your local models
hf-model-tool
```

## Usage

### Interactive Mode

```bash
vllm-cli
```

Launch the interactive terminal interface with menu-driven navigation for model serving, configuration, and monitoring.

#### Model Selection with Remote Support
![Model Selection](asset/model-selection-remote.png)
*Model selection interface showing both local models and HuggingFace Hub auto-download option*

#### Quick Serve with Last Configuration
![Quick Serve](asset/quick-serve-config.png)
*Quick serve feature automatically uses the last successful configuration*

#### Custom Configuration Example
![Custom Configuration](asset/custom-configuration.png)
*Advanced configuration interface with categorized vLLM options and custom arguments*

### Server Monitoring
![Server Monitoring](asset/server-monitoring.png)
*Real-time server monitoring showing GPU utilization, server status, and streaming logs*

### Command-Line Mode

```bash
# Serve a model with default settings
vllm-cli serve MODEL_NAME

# Serve with a specific profile
vllm-cli serve MODEL_NAME --profile standard

# Serve with custom parameters
vllm-cli serve MODEL_NAME --quantization awq --tensor-parallel-size 2

# List available models
vllm-cli models

# Show system information
vllm-cli info

# Check active servers
vllm-cli status

# Stop a server
vllm-cli stop --port 8000
```

## Configuration

### User Configuration Files

- **Main Config**: `~/.config/vllm-cli/config.yaml`
- **User Profiles**: `~/.config/vllm-cli/user_profiles.json`
- **Cache**: `~/.config/vllm-cli/cache.json`

### Built-in Profiles

Four carefully selected profiles cover the most common use cases. Since vLLM only uses one GPU by default, all profiles include  multi-GPU detection that automatically sets tensor parallelism to utilize all available GPUs.

#### `standard` - Minimal configuration with smart defaults
*Uses vLLM's defaults configuration. Perfect for most models and hardware setups.*

#### `moe_optimized` - Optimized for Mixture of Experts models
```json
{
  "enable_expert_parallel": true
}
```
*Enables expert parallelism for MoE models like Qwen*

#### `high_throughput` - Maximum performance configuration
```json
{
  "max_model_len": 8192,
  "gpu_memory_utilization": 0.95,
  "enable_chunked_prefill": true,
  "max_num_batched_tokens": 8192,
  "trust_remote_code": true,
  "enable_prefix_caching": true
}
```
*Aggressive settings for maximum request throughput*
#### `low_memory` - Memory-constrained environments
```json
{
  "max_model_len": 4096,
  "gpu_memory_utilization": 0.70,
  "enable_chunked_prefill": false,
  "trust_remote_code": true,
  "quantization": "fp8"
}
```
*Reduces memory usage through FP8 quantization and conservative settings*

### Error Handling and Log Viewing
![Error Handling](asset/error-handling-logs.png)
*Interactive error recovery with log viewing options when server startup fails*

## System Information

![System Information](asset/system-information.png)
*Comprehensive system information display showing GPU capabilities, memory, dependencies version, attention backends, and quantization support*

## Architecture

### Core Components

- **CLI Module**: Argument parsing and command handling
- **Server Module**: vLLM process lifecycle management
- **Config Module**: Configuration and profile management
- **Models Module**: Model discovery and metadata extraction
- **UI Module**: Rich terminal interface components
- **System Module**: GPU, memory, and environment utilities
- **Validation Module**: Configuration validation framework
- **Errors Module**: Comprehensive error handling

### Key Features

- **Automatic Model Discovery**: Integration with hf-model-tool for comprehensive model detection
- **Profile System**: JSON-based configuration with validation
- **Process Management**: Global server registry with automatic cleanup
- **Caching**: Performance optimization for model listings and system information
- **Error Handling**: Comprehensive error recovery and user feedback

## Documentation

### Model Discovery & Troubleshooting
- [**Model Discovery Quick Reference**](docs/MODEL_DISCOVERY_QUICK_REF.md) - Quick troubleshooting guide for model visibility issues
- [**Model Discovery Flow**](docs/MODEL_DISCOVERY_FLOW.md) - Technical details of how models are discovered and cached

### Integration Guides
- [**Ollama Integration**](docs/ollama-integration.md) - Guide for using Ollama-downloaded models with vLLM CLI
- [**Custom Model Serving**](docs/custom-model-serving.md) - Comprehensive guide for serving models from custom directories

### Development
- [**Testing Guide**](docs/TESTING.md) - Instructions for running tests

## Development

### Project Structure

```
src/vllm_cli/
├── cli/           # CLI command handling
├── config/        # Configuration management
├── errors/        # Error handling
├── models/        # Model management
├── server/        # Server management
├── system/        # System utilities
├── ui/            # User interface
├── validation/    # Validation framework
└── schemas/       # JSON schemas
```

## Environment Variables

- `VLLM_CLI_ASCII_BOXES`: Use ASCII box drawing characters for compatibility
- `VLLM_CLI_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)

## Requirements

### System Requirements

- Linux
- NVIDIA GPU with CUDA support (Only NVIDIA GPUs are supported right now, PRs are welcome)

### Python Dependencies

- vLLM
- PyTorch with CUDA support

Note: Following dependencies are downloaded along with vLLM CLI:
- hf-model-tool (model discovery)
- Rich (terminal UI)
- Inquirer (interactive prompts)
- psutil (system monitoring)
- PyYAML (configuration parsing)

## Roadmap

### v0.2.4 (Upcoming)

- [x] **Ollama Model Support** - Discover and serve GGUF models from Ollama directories (experimental)
- [ ] **Docker Backend Support** - Use existing vLLM Docker images as backend

### To-Do List

- [ ] **AMD GPU Support** - Add support for AMD GPUs (ROCm) in addition to NVIDIA CUDA
- [ ] **Enhanced Local Model Support** - Add support for additional local model formats:
  - [ ] Oracle Cloud Infrastructure (OCI) Registry format
  - [ ] Direct GGUF file loading without Ollama
  - [ ] Other local model formats

### Future Enhancements

Additional features and improvements planned for future releases will be added here as the project evolves.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome, please feel free to open an issue or submit a pull request.
