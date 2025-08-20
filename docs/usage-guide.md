# vLLM CLI Usage Guide

Comprehensive guide for using vLLM CLI in both interactive and command-line modes.

## Table of Contents
- [Interactive Mode](#interactive-mode)
- [Command-Line Mode](#command-line-mode)
- [Configuration Management](#configuration-management)
- [Environment Variables](#environment-variables)
- [Shortcuts vs Profiles](#shortcuts-vs-profiles)
- [Advanced Usage](#advanced-usage)

## Interactive Mode

Launch the interactive terminal interface:
```bash
vllm-cli
```

The interactive mode provides a menu-driven interface for:
- Model serving with guided configuration
- Profile management and creation
- Server monitoring with real-time updates
- System information display
- Settings configuration

### Navigation
- Use arrow keys to navigate menus
- Press Enter to select options
- Use Ctrl+C to exit

## Command-Line Mode

### Basic Commands

```bash
# Serve a model with default settings
vllm-cli serve MODEL_NAME

# Serve with a specific profile
vllm-cli serve MODEL_NAME --profile standard

# Serve with custom parameters
vllm-cli serve MODEL_NAME --quantization awq --tensor-parallel-size 2
```

### GPU Selection

```bash
# Use specific GPU devices (new in v0.2.4)
vllm-cli serve MODEL_NAME --device 0,1  # Use GPU 0 and 1
vllm-cli serve MODEL_NAME --device 2    # Use only GPU 2
```

### Shortcuts Management

```bash
# Create and use shortcuts for quick launching
vllm-cli serve MODEL --profile high_throughput --save-shortcut "my-fast-model"
vllm-cli serve --shortcut "my-fast-model"

# Manage shortcuts
vllm-cli shortcuts                     # List all shortcuts
vllm-cli shortcuts --delete NAME       # Delete a shortcut
vllm-cli shortcuts --export NAME       # Export shortcut to file
vllm-cli shortcuts --import FILE       # Import shortcut from file
```

### Model Management

```bash
# List available models
vllm-cli models

# Refresh model cache
vllm-cli models --refresh

# Serve remote model from HuggingFace
vllm-cli serve "facebook/opt-125m" --remote
```

### Server Management

```bash
# Check active servers
vllm-cli status

# Stop a server
vllm-cli stop --port 8000

# Stop all servers
vllm-cli stop --all
```

### System Information

```bash
# Show system information
vllm-cli info

# Show detailed GPU information
vllm-cli info --verbose
```

## Configuration Management

### Configuration Files

All configuration files are stored in `~/.config/vllm-cli/`:

- **config.yaml** - Main configuration file
- **user_profiles.json** - Custom server profiles
- **shortcuts.json** - Saved model+profile combinations
- **cache.json** - Cached model and system information

### Built-in Profiles

#### General Purpose
- **standard** - Minimal configuration with smart defaults
- **moe_optimized** - Optimized for Mixture of Experts models
- **high_throughput** - Maximum performance configuration
- **low_memory** - Memory-constrained environments

#### Hardware-Specific (GPT-OSS)
- **gpt_oss_ampere** - Optimized for NVIDIA A100 GPUs
- **gpt_oss_hopper** - Optimized for NVIDIA H100/H200 GPUs
- **gpt_oss_blackwell** - Optimized for NVIDIA Blackwell GPUs

### Creating Custom Profiles

Custom profiles can be created through:
1. Interactive mode: Settings → Profile Management → Create Profile
2. Saving from Custom Configuration after testing
3. Manually editing `~/.config/vllm-cli/user_profiles.json`

Example profile structure:
```json
{
  "my_profile": {
    "tensor_parallel_size": 2,
    "max_model_len": 4096,
    "quantization": "awq",
    "environment_variables": {
      "VLLM_ATTENTION_BACKEND": "FLASH_ATTN"
    }
  }
}
```

## Environment Variables

### Three-Tier System

1. **Universal Variables** - Set in Settings → Universal Environment Variables
   - Apply to ALL servers
   - Provide baseline configuration

2. **Profile Variables** - Defined within each profile
   - Override universal variables
   - Hardware or model-specific optimizations

3. **Session Variables** - Set in Custom Configuration
   - One-time use for current session
   - Highest priority

### Priority Order
Universal < Profile = Session

### Common vLLM Environment Variables

#### GPU Optimization
- `VLLM_ATTENTION_BACKEND` - Backend for attention computation
- `VLLM_USE_TRITON_FLASH_ATTN` - Enable Triton flash attention

#### MoE Models
- `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16` - BF16 precision for MoE
- `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING` - Activation chunking

#### Logging
- `VLLM_LOGGING_LEVEL` - Set logging verbosity
- `VLLM_LOG_STATS_INTERVAL` - Stats logging interval

#### Memory
- `VLLM_CPU_KVCACHE_SPACE` - CPU cache space in GB
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN` - Allow extended context

#### CUDA
- `CUDA_VISIBLE_DEVICES` - Specify GPU devices
- `CUDA_HOME` - CUDA installation path
- `TORCH_CUDA_ARCH_LIST` - Target architectures

## Shortcuts vs Profiles

### Profiles
- **Purpose**: Reusable server configuration templates
- **Content**: vLLM arguments and environment variables
- **Model**: Not included (model-agnostic)
- **Use Case**: Define once, use with any compatible model

### Shortcuts
- **Purpose**: Quick launch specific model+profile combinations
- **Content**: Model name + profile reference
- **Model**: Included (model-specific)
- **Use Case**: Frequently used configurations

Example workflow:
1. Create a profile with your preferred settings
2. Test with different models
3. Save successful combinations as shortcuts
4. Launch instantly with `vllm-cli serve --shortcut NAME`

## Advanced Usage

### Serving LoRA Adapters

```bash
# Serve base model with LoRA adapters
vllm-cli serve BASE_MODEL --lora-modules adapter1=path1 adapter2=path2
```

### Custom Arguments

Pass any vLLM argument not explicitly supported:
```bash
vllm-cli serve MODEL --custom-args "--arg1 value1 --arg2 value2"
```

### Integration with Scripts

```python
import subprocess

# Launch vLLM server programmatically
result = subprocess.run(
    ["vllm-cli", "serve", "facebook/opt-125m", "--profile", "high_throughput"],
    capture_output=True,
    text=True
)
```

## Tips and Best Practices

1. **Profile Selection**: Start with `standard` profile and adjust based on needs
2. **GPU Memory**: Monitor GPU memory usage to optimize batch sizes
3. **Quantization**: Use quantization for larger models on limited hardware
4. **Environment Variables**: Test environment variables in session before adding to profiles
5. **Shortcuts**: Create shortcuts for production deployments
6. **Monitoring**: Keep server monitoring open to track performance
7. **Logs**: Check logs when servers fail to start for detailed error messages

## See Also

- [Troubleshooting Guide](troubleshooting.md)
- [Model Discovery](MODEL_DISCOVERY_FLOW.md)
- [Custom Model Serving](custom-model-serving.md)
- [Ollama Integration](ollama-integration.md)
