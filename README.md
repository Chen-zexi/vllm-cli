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

## What's New in v0.2.4rc

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

# Serve with specific GPU devices (new in v0.2.5)
vllm-cli serve MODEL_NAME --device 0,1  # Use GPU 0 and 1
vllm-cli serve MODEL_NAME --device 2    # Use only GPU 2

# Create and use shortcuts for quick launching
vllm-cli serve MODEL --profile high_throughput --save-shortcut "my-fast-model"
vllm-cli serve --shortcut "my-fast-model"

# Manage shortcuts
vllm-cli shortcuts                     # List all shortcuts
vllm-cli shortcuts --delete NAME       # Delete a shortcut
vllm-cli shortcuts --export NAME       # Export shortcut to file

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
- **Shortcuts**: `~/.config/vllm-cli/shortcuts.json`
- **Cache**: `~/.config/vllm-cli/cache.json`

### Shortcuts vs Profiles

- **Profiles**: Reusable server configuration templates (model-agnostic)
- **Shortcuts**: Saved combinations of specific model + profile for quick launching

### Built-in Profiles

Seven carefully designed profiles cover most common use cases and hardware configurations. All profiles include multi-GPU detection that automatically sets tensor parallelism to utilize all available GPUs.

#### General Purpose Profiles

##### `standard` - Minimal configuration with smart defaults
*Uses vLLM's defaults configuration. Perfect for most models and hardware setups.*

##### `moe_optimized` - Optimized for Mixture of Experts models
*Enables expert parallelism for MoE models with optimized environment variables*

##### `high_throughput` - Maximum performance configuration
*Aggressive settings for maximum request throughput with Triton flash attention*

##### `low_memory` - Memory-constrained environments
*Reduces memory usage through FP8 quantization and conservative settings*

#### Hardware-Specific Profiles for GPT-OSS Models (New in v0.2.5)

> **Note**: For the latest GPU-specific optimizations and model requirements, please refer to the [vLLM GPT recipes](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html). The built-in profiles provide a starting point that can be customized based on your specific hardware and model requirements.

##### `gpt_oss_ampere` - GPT-OSS on NVIDIA A100 (Ampere)
*Optimized for GPT-OSS models on A100 GPUs with:*
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1` for optimized attention
- `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1` for MoE activation chunking
- Async scheduling enabled for improved performance

##### `gpt_oss_hopper` - GPT-OSS on NVIDIA H100/H200 (Hopper)
*Optimized for GPT-OSS models on H100/H200 GPUs with:*
- `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1` for MoE activation chunking
- Async scheduling enabled
- Auto-detection of tensor parallelism for multi-GPU setups

##### `gpt_oss_blackwell` - GPT-OSS on NVIDIA Blackwell (B100/B200)
*Optimized for GPT-OSS models on latest Blackwell GPUs with:*
- `VLLM_USE_TRTLLM_ATTENTION=1` for TensorRT-LLM attention backend
- `VLLM_USE_FLASHINFER_MXFP4_BF16_MOE=1` for BF16 precision MoE
- Async scheduling for maximum throughput

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

### vLLM CLI Environment Variables

- `VLLM_CLI_ASCII_BOXES`: Use ASCII box drawing characters for compatibility
- `VLLM_CLI_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)

### vLLM Environment Variable Support

vLLM CLI provides comprehensive environment variable management with a three-tier system:

#### 1. Universal Environment Variables
Configure in **Settings → Universal Environment Variables** to apply to ALL servers. These provide baseline settings across all your deployments.

#### 2. Profile Environment Variables
Each profile can include specific environment variables that override universal settings. Perfect for hardware or model-specific optimizations.

#### 3. Session Environment Variables
Set environment variables for one-off server runs directly in Custom Configuration. These are applied only to the current session. **Note: Session environment variables will be set to profile environment variables if you choose to save the custom configuration as a profile.**

**Priority Order**: Universal < Profile = Session

Common vLLM environment variables supported:
- **GPU Optimization**: `VLLM_ATTENTION_BACKEND`, `VLLM_USE_TRITON_FLASH_ATTN`
- **MoE Models**: `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16`, `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING`
- **Logging**: `VLLM_LOGGING_LEVEL`, `VLLM_LOG_STATS_INTERVAL`
- **Memory**: `VLLM_CPU_KVCACHE_SPACE`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN`
- **CUDA**: `CUDA_VISIBLE_DEVICES`, `CUDA_HOME`, `TORCH_CUDA_ARCH_LIST`

**Note**: vLLM CLI only passes explicitly configured environment variables to preserve native vLLM behavior. No defaults are forced.

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

### Serving Configurations
- [x] **Environment Variable Support** - Comprehensive three-tier environment variable system for complete control
- [x] **GPU Selection Feature** - Select specific GPUs for model serving via `--device` flag or interactive UI
- [x] **Server Cleanup Control** - Configure whether servers are stopped when CLI exits
- [x] **vLLM native Arguments** - Added 16+ new critical vLLM arguments for advanced configurations
- [x] **Shortcuts System** - Save and quickly launch model+profile combinations
- [ ] **Docker Backend Support** - Use existing vLLM Docker images as backend

### UI Features
- [x] **Rich Terminal UI** - Rich terminal interface with menu-driven navigation
- [x] **Command-Line Mode** - Direct CLI commands for automation and scripting
- [x] **System Information** - GPU, memory, and CUDA compatibility checking
- [ ] **CPU Stats** - CPU usage, memory usage, and disk usage checking
- [ ] **UI Customization**
    - [x] Customize GPU stats bar
    - [x] Customize log refresh frequency
    - [ ] Customize theme
    - [ ] Multi-language support
- [ ] **Server Monitoring**
    - [x] Real-time monitoring of active vLLM servers
    - [ ] Server monitoring after exiting program
- [ ] **Log Viewer**
    - [x] View the complete log file when server startup fails
    - [ ] View logs for past runs

### Hardware Support
- [x] **NVIDIA GPUs** - Support for NVIDIA GPUs
- [ ] **AMD GPUs** - Support for AMD GPUs (ROCm)  --> Need help from AMD users! Contributions are welcome!

### Model Discovery Support
- [x] **HuggingFace Model Support** - Discover and serve models from HuggingFace Hub
- [x] **Custom Model Directories** - Support for custom model directories
- [x] **Ollama Model Support** - Discover and serve GGUF models from Ollama directories (experimental)
- [x] **GGUF file loading** - Support for direct GGUF file loading
- [x] **Model Manifest Support** - Map custom models in vLLM CLI native way with `models_manifest.json`
- [ ] **Oracle Cloud Infrastructure (OCI) Registry** - Support for OCI Registry format


### Future Enhancements

Additional features and improvements planned for future releases will be added here as the project evolves.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome, please feel free to open an issue or submit a pull request.
