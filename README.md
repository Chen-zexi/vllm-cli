# vLLM CLI

A command-line interface tool for serving Large Language Models using vLLM. Provides both interactive and command-line modes with features for configuration profiles, model management, and server monitoring.

![vLLM CLI Welcome Screen](asset/welcome-screen.png)
*Welcome screen showing GPU status and system overview*

## Features

- **Interactive Mode**: Rich terminal interface with menu-driven navigation
- **Command-Line Mode**: Direct CLI commands for automation and scripting
- **Model Management**: Automatic discovery and management of local models
- **Remote Model Support**: Serve models directly from HuggingFace Hub without pre-downloading
- **Configuration Profiles**: Pre-configured and custom server profiles
- **Server Monitoring**: Real-time monitoring of active vLLM servers
- **System Information**: GPU, memory, and CUDA compatibility checking
- **Log Viewer**: View the complete log file when server startup fails

### Server Monitoring
![Server Monitoring](asset/server-monitoring.png)
*Real-time server monitoring showing GPU utilization, server status, and streaming logs*

## Installation

### Prerequisites

- Python 3.8+
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

# Create conda environment
conda create -n vllm-cli python=3.11
conda activate vllm-cli

# Install dependencies
pip install -r requirements.txt
pip install hf-model-tool

# Install CLI in development mode
pip install -e .
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
  "quantization": "bitsandbytes"
}
```
*Reduces memory usage through quantization and conservative settings*

### Dynamic Configuration Features

- **Automatic Hardware Detection**: Profiles automatically detect and optimize for available hardware (GPU count, memory, capabilities)
- **Optimal Data Type Selection**: vLLM automatically chooses the best dtype (bfloat16, float16, float32) based on hardware support and model requirements
- **Intelligent Multi-GPU Support**: Since vLLM defaults to single GPU usage, our system automatically detects multiple GPUs and sets `tensor_parallel_size` to utilize all available hardware
- **Model-Native Context**: Profiles without explicit `max_model_len` use the model's native maximum context length
- **Quantization Compatibility**: All quantization methods (including BitsAndBytes) work seamlessly with tensor parallelism

### Custom Profiles

Create custom profiles through the interactive interface or by editing the user profiles file directly.

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

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome, please feel free to open an issue or submit a pull request.