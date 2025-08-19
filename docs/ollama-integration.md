# Ollama Model Integration for vLLM CLI

## Overview

This document describes Ollama-specific considerations for vLLM CLI. For general model discovery and troubleshooting, see:
- **[MODEL_DISCOVERY_QUICK_REF.md](./MODEL_DISCOVERY_QUICK_REF.md)** - Quick troubleshooting guide
- **[MODEL_DISCOVERY_FLOW.md](./MODEL_DISCOVERY_FLOW.md)** - Complete technical flow

## Important Notice: Ollama Model Directories

Ollama may store models in different directories depending on how it was installed:

- **User Directory**: `~/.ollama/models/` - Default location for user installations
- **System Directory**: `/usr/share/ollama/.ollama/models/` - Used when Ollama is installed system-wide (e.g., via package managers)
- **Custom Directory**: Set via `OLLAMA_MODELS` environment variable

vLLM CLI will scan all these locations automatically through hf-model-tool integration.

## Key Differences: Ollama vs Native HuggingFace

### Ollama Models
- **Format**: GGUF (GPT-Generated Unified Format) - a quantized binary format
- **Storage**: `~/.ollama/models/blobs/` as SHA256-named blob files
- **Structure**: Manifest-based with layers stored separately
- **Quantization**: Built-in quantization (Q4_0, Q5_K_M, etc.)
- **Size**: Typically smaller due to quantization

### HuggingFace Models
- **Format**: Safetensors or PyTorch .bin files
- **Storage**: `~/.cache/huggingface/hub/` in organized directories
- **Structure**: Directory with config.json + weight files
- **Quantization**: Optional, applied separately
- **Size**: Full precision or explicitly quantized

## vLLM Compatibility

### GGUF Support Status
- **Availability**: vLLM 0.5.0+ has experimental GGUF support
- **Status**: Performance might be suboptimal
- **Limitations**: Not all GGUF architectures are supported

### Known Unsupported GGUF Architectures

The following GGUF model architectures are **NOT** supported by vLLM as of v0.8.5:

| Architecture | Example Models | Issue | Status |
|-------------|---------------|-------|--------|
| `qwen3moe` | qwen3:30b, Qwen3 MoE variants | [#18382](https://github.com/vllm-project/vllm/issues/18382) | Not supported |
| `llama4` | Llama 4 models | Not yet implemented | Not supported |

**Note**: Regular Qwen3 models (non-MoE) in GGUF format may work. Always check the latest vLLM documentation for current support status.

## Implementation Details

For complete technical details on how models are discovered and cached, see [MODEL_DISCOVERY_FLOW.md](./MODEL_DISCOVERY_FLOW.md).

### Ollama-Specific Components

1. **UI Support**
   - Ollama models appear with "ollama" provider label
   - Warning messages about experimental GGUF support
   - Confirmation prompts before using GGUF models

2. **Server Configuration**
   - Automatic `quantization="gguf"` parameter for GGUF models
   - Special handling in config builder for GGUF paths
   - `served_model_name` field correctly set for vLLM

## Usage

### Configuring Ollama Support

#### Enable/Disable Ollama Scanning
In vLLM CLI:
- Navigate to: `Settings → Model Directories → Toggle Ollama Scanning`
- `Model Management → Refresh Model Cache` after enabling

### Handling Permission Issues
When Ollama models are in system directories:
```bash
# Models in /usr/share/ollama/ may require elevated permissions
# vLLM CLI will detect but may not be able to serve without proper permissions

# Option 1: Copy model to user directory
cp -r /usr/share/ollama/.ollama/models/blobs/sha256-xxx ~/.ollama/models/blobs/

# Option 2: Adjust permissions (requires admin)
sudo chmod -R a+r /usr/share/ollama/.ollama/models/
```

### Serving Ollama Models

#### Via Interactive UI
1. Run `vllm-cli`
2. Select "Start Server"
3. Choose one of the serving options
4. Select "ollama" provider
5. Choose your model (warning will be displayed)
6. Confirm to proceed with experimental support

#### Via Command Line
```bash
# Direct path to GGUF blob
vllm-cli serve /home/user/.ollama/models/blobs/sha256-{hash} --quantization gguf
```

## Testing Recommendations

### Basic Compatibility Test
```bash
vllm serve /home/user/.ollama/models/blobs/sha256-{hash} --quantization gguf --gpu-memory-utilization 0.5 --max-model-len 512
```

## Troubleshooting

### Unsupported Architecture Error

If you see errors like:
```
The checkpoint you are trying to load has model type `qwen3moe` but Transformers does not recognize this architecture
```

Or:
```
✗ Failed to start server
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, GGUF model with architecture llama4 is not supported yet.
```

This means the GGUF model architecture is not supported by vLLM. Solutions:
1. **Use a different model**: Try the non-MoE version (e.g., use `qwen3:8b` instead of `qwen3:30b`)
2. **Use native format**: Download the model from HuggingFace instead of using GGUF
3. **Use Ollama directly**: Run `ollama serve` for unsupported GGUF models
4. **Monitor vLLM updates**: Check the GitHub issues linked above for support status

### Models Not Appearing

For comprehensive troubleshooting steps, see [MODEL_DISCOVERY_QUICK_REF.md](./MODEL_DISCOVERY_QUICK_REF.md#quick-diagnosis-tree).

Quick fix for Ollama models:
1. Go to `Settings → Model Directories → Toggle Ollama Scanning`
2. Enable Ollama scanning
3. Use `Model Management → Refresh Model Cache`

### Permission Issues
```bash
# Verify read access to Ollama directories
ls -la ~/.ollama/models/
ls -la /usr/share/ollama/.ollama/models/
```
