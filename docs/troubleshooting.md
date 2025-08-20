# Troubleshooting Guide

Common issues and solutions for vLLM CLI.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Model Not Showing Up](#model-not-showing-up)
- [GPU Compatibility Issues](#gpu-compatibility-issues)
- [Server Startup Failures](#server-startup-failures)
- [Memory Issues](#memory-issues)
- [Environment Variable Issues](#environment-variable-issues)

## Installation Issues

### "System requirements not met" Error
```
$ vllm-cli --help
System requirements not met. Please check the log for details.
```

**Common Cause:** Installing vllm-cli with pipx or in a different environment than vLLM and PyTorch

**Why This Happens:**
- Tools like pipx create isolated environments for each application
- vllm-cli is installed in its own environment, separate from your vLLM and PyTorch installation
- When vllm-cli tries to `import torch` or `import vllm`, it can't find them
- Even though you have PyTorch/vLLM installed elsewhere, the isolated environment can't access them

**Solutions:**

1. **Install in the same environment as vLLM (Recommended):**
```bash
# First, activate the environment where vLLM is installed
conda activate your_vllm_env  # or source venv/bin/activate

# Then install vllm-cli there
pip install vllm-cli
```

2. **Verify the issue:**
```bash
# Check if this is an isolation problem
python -c "import torch; print('PyTorch found')"  # Works in your environment
pipx runpip vllm-cli list  # Shows vllm-cli's isolated packages (no torch/vllm)
```

3. **Alternative: Use pip instead of pipx:**
```bash
# Uninstall from pipx
pipx uninstall vllm-cli

# Install with regular pip in your vLLM environment
pip install vllm-cli
```

**Note:** Future versions may support `pip install vllm-cli[full]` to include vLLM dependencies for pipx compatibility, but this is not recommended due to vLLM's large size and specific CUDA requirements.
```

### Dependency Conflicts
**Solutions:**
1. Create fresh virtual environment
2. Install PyTorch first with correct CUDA version
3. Then install vLLM
4. Finally install vllm-cli

## Model Not Showing Up

### Quick Checklist
1. **Refresh model cache**: `vllm-cli models --refresh`
2. **Check model directories** in Settings → Model Directories
3. **Verify hf-model-tool** is installed: `pip install hf-model-tool`
4. **Check Ollama models** are enabled in settings

### Detailed Solutions

#### HuggingFace Models
- Default location: `~/.cache/huggingface/hub/`
- Ensure models are fully downloaded (check for `.incomplete` files)
- Verify model format is supported by vLLM

#### Ollama Models
- User directory: `~/.ollama/models/`
- System directory: `/usr/share/ollama/.ollama/models/`
- Enable Ollama scanning in Settings → Model Directories
- GGUF format requires vLLM 0.5.0+

#### Custom Directories
- Add directories via Settings → Model Directories → Add Directory
- Ensure proper permissions for directory access
- Use absolute paths

See [Model Discovery Quick Reference](MODEL_DISCOVERY_QUICK_REF.md) for more details.

## GPU Compatibility Issues

### Common Errors and Solutions

#### CUDA Out of Memory
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Solutions:**
- Use quantization: `--quantization awq` or `--quantization gptq`
- Reduce max model length: `--max-model-len 2048`
- Use `low_memory` profile
- Enable CPU offloading: `--cpu-offload-gb 10`

#### Unsupported GPU Architecture
```
RuntimeError: GPU architecture not supported
```
**Solutions:**
- Check GPU compute capability: navigate to system information from main menu in vLLM CLI
- vLLM requires compute capability ≥ 7.0
- Update CUDA drivers and PyTorch

#### Multiple GPU Issues
```
RuntimeError: Tensor parallel size mismatch
```
**Solutions:**
- Ensure tensor_parallel_size matches number of GPUs
- Use `--device` flag to specify GPUs: `--device 0,1`
- Check all GPUs are same model

## Server Startup Failures

### Port Already in Use
```
OSError: [Errno 98] Address already in use
```
**Solutions:**
- Check active servers: `vllm-cli status`
- Stop conflicting server: `vllm-cli stop --port 8000`
- Use different port: `--port 8001`

### Model Loading Errors
```
ValueError: Model not found or unsupported
```
**Solutions:**
- Verify model architecture is supported by vLLM
- Check [vLLM documentation](https://docs.vllm.ai/) for supported models
- Try without quantization first
- Ensure model files are not corrupted

### Configuration Conflicts
```
ValueError: Incompatible configuration parameters
```
**Solutions:**
- Start with `standard` profile
- Remove conflicting arguments
- Check model-specific requirements

## Memory Issues

### System RAM
```
MemoryError: Unable to allocate memory
```
**Solutions:**
- Close other applications
- Reduce `--max-num-seqs`
- Use `--enable-prefix-caching false`
- Monitor with `free -h`

## Environment Variable Issues

### Variables Not Applied
**Check:**
1. Variable spelling and format

### CUDA Variables
```bash
# Common fixes
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

### vLLM Backend Variables
```bash
# Test different backends
VLLM_ATTENTION_BACKEND=FLASH_ATTN  # Flash Attention
VLLM_ATTENTION_BACKEND=XFORMERS    # xFormers
VLLM_ATTENTION_BACKEND=TRITON      # Triton
```

### Log Files
- Server logs: Check output when server fails
- vLLM logs: Set the log level in settings to DEBUG
- System logs: `dmesg | grep -i gpu` for GPU issues

### Resources
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub Issues](https://github.com/vllm-project/vllm/issues)
- [vLLM CLI Issues](https://github.com/Chen-zexi/vllm-cli/issues)
