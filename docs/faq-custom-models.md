# FAQ: Loading Custom and GGUF Models

> This guide addresses common questions about loading fine-tuned models and GGUF files in vLLM-CLI.

## Table of Contents

1. [Loading Fine-Tuned Models](#loading-fine-tuned-models)
2. [Loading GGUF Files](#loading-gguf-files)
3. [Troubleshooting](#troubleshooting)

---

## Loading Fine-Tuned Models

### Prerequisites

Your fine-tuned model directory must contain:
- `config.json` - Model configuration file
- Model weights in `.safetensors`, `.bin`, or `.pt` format
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, etc.)

### Step-by-Step Guide

#### 1. Register Your Model Directory

**Option A: Via vLLM-CLI UI**
```bash
vllm-cli
```
Then navigate to: **Settings** → **Model Directories** → **Add Model Directory**

Enter the path to your model (e.g., `/home/user/notebooks/qwen3-4b-bf16`)

**Option B: Via hf-model-tool CLI**
```bash
pip install hf-model-tool  # if not already installed
hf-model-tool -path /path/to/your/model/directory
```

#### 2. Verify Model Registration

After adding the directory, a `models_manifest.json` file is automatically created. You can verify your model is registered by:

1. In vLLM-CLI: **Model Management** → **View Model Directories**
2. Or check the manifest file directly in your model directory

#### 3. Refresh the Model Cache

In vLLM-CLI: **Model Management** → **Refresh Model Cache**

This ensures vLLM-CLI picks up the newly registered models.

#### 4. Serve Your Model

**Via UI:**
1. Select **Start Server** from the main menu
2. Choose a serving option
3. Select your custom model from the list
4. Choose a profile and start

**Via Command Line:**
```bash
vllm-cli serve /path/to/your/model --profile standard
```

---

## Loading GGUF Files

### Important: GGUF Support is Experimental

vLLM has **experimental** GGUF support (since v0.5.0). Performance may be suboptimal, and not all architectures are supported.

### Supported vs Unsupported

| Status | Architectures |
|--------|---------------|
| **Supported** | Most standard models (Llama, Qwen3 non-MoE, Mistral, etc.) |
| **Not Supported** | `qwen3moe` (MoE variants), `llama4` |

### Option 1: Use Ollama Integration (Recommended for GGUF)

If you already have Ollama installed with GGUF models:

1. **Enable Ollama Scanning:**
   - In vLLM-CLI: **Settings** → **Model Directories** → **Toggle Ollama Scanning**

2. **Refresh Cache:**
   - **Model Management** → **Refresh Model Cache**

3. **Select Ollama Model:**
   - Your Ollama models will appear under the "ollama" provider

Ollama models are typically stored in:
- User: `~/.ollama/models/`
- System: `/usr/share/ollama/.ollama/models/`

### Option 2: Direct GGUF File Serving

For standalone GGUF files (not managed by Ollama):

```bash
# Serve directly with quantization flag
vllm-cli serve /path/to/your/model.gguf --quantization gguf
```

**Example:**
```bash
vllm-cli serve /home/user/models/qwen3-4b-Q4_K_M.gguf --quantization gguf
```

### Option 3: Import GGUF into Ollama First

If vLLM has issues with your GGUF file directly, you can import it into Ollama:

```bash
# Create a Modelfile
echo "FROM /path/to/qwen3-4b-Q4_K_M.gguf" > Modelfile

# Import into Ollama
ollama create my-qwen3-4b -f Modelfile

# Now it will appear in vLLM-CLI's Ollama models
```

---

## Troubleshooting

### Model Shows in List but Won't Load

1. **Verify model structure:**
   ```bash
   ls -la /path/to/your/model/
   # Should contain: config.json, model weights, tokenizer files
   ```

2. **Check config.json is valid:**
   ```bash
   python -c "import json; json.load(open('/path/to/your/model/config.json'))"
   ```

3. **Test with transformers first:**
   ```python
   from transformers import AutoModel, AutoTokenizer
   model = AutoModel.from_pretrained("/path/to/your/model")
   tokenizer = AutoTokenizer.from_pretrained("/path/to/your/model")
   ```

4. **Check vLLM compatibility:**
   - Ensure the model architecture is supported by vLLM
   - Check GPU memory requirements

### GGUF Files Not Appearing

1. **For Ollama models:** Enable Ollama scanning in Settings
2. **For standalone GGUF:** Use direct command-line serving with `--quantization gguf`
3. **Refresh cache:** Always refresh after adding new models

### "Unsupported Architecture" Error

If you see errors about unsupported architectures:

1. **Check if it's a MoE model** - MoE variants (like `qwen3moe`) are not supported in GGUF format
2. **Use HuggingFace format instead** - Download the native format from HuggingFace
3. **Use Ollama directly** - Run `ollama serve` for unsupported GGUF models

### Permission Issues

```bash
# Check permissions
ls -la /path/to/your/model/

# Fix if needed
chmod -R u+r /path/to/your/model/
```

---

## Quick Reference Commands

```bash
# Add custom model directory
hf-model-tool -path /path/to/models

# Serve custom HuggingFace-format model
vllm-cli serve /path/to/model --profile standard

# Serve GGUF file directly
vllm-cli serve /path/to/model.gguf --quantization gguf

# Serve with specific GPU settings
vllm-cli serve /path/to/model --device 0 --gpu-memory-utilization 0.8
```

---

## Related Documentation

- [Custom Model Serving Guide](./custom-model-serving.md)
- [Ollama Integration](./ollama-integration.md)
- [Model Discovery Quick Reference](./MODEL_DISCOVERY_QUICK_REF.md)
