# Serving Models from Custom Directories

## Overview

This guide explains how to serve models from custom directories (non-HuggingFace cache locations) using vLLM-CLI. Custom directories are essential when working with:
- Fine-tuned models
- Merged models
- Models downloaded outside of HuggingFace
- LoRA adapters
- Custom model formats

## Prerequisites

1. **vLLM-CLI installed**:
   ```bash
   pip install vllm-cli
   ```

2. **HF-MODEL-TOOL installed** (for model discovery):
   ```bash
   pip install hf-model-tool
   ```

3. **Models in supported formats**:
   - Must have `config.json` file
   - Model weights in `.safetensors`, `.bin`, or `.pt` format
   - Tokenizer files if applicable

## Step-by-Step Guide

### Step 1: Add Your Custom Directory

#### Method A: During vLLM-CLI Session

1. Launch vLLM-CLI:
   ```bash
   vllm-cli
   ```

2. Navigate to **Settings** from the main menu

3. Select **Model Directories**

4. Select **Add Model Directory**

5. Enter the path to your custom model directory:
   ```
   Path: /home/user/my-models
   ```

6. Choose directory type:
   - **Auto-detect** (recommended) - Let the tool determine the type
   - **Custom** - For fine-tuned or merged models
   - **LoRA** - For directories containing LoRA adapters

7. The cli will automatically scan the directory and generate a `models_manifest.json` file.
8. You can review the manifest and edit it if needed.
9. You can also add multiple directories by clicking **Add Model Directory** again.
10. You can also remove a directory by clicking **Remove Model Directory**.

#### Method B: Using HF-MODEL-TOOL Directly

```bash
# Add directory via command line
hf-model-tool -path /home/user/my-models
```

### Step 2: Understand the Manifest System

When you add a custom directory, a `models_manifest.json` file is automatically generated in the directory root. This file:
- Contains metadata for all detected models
- Is the primary source for model information
- Can be edited to customize model names and publishers

#### Example Manifest Structure

```json
{
  "version": "1.0",
  "generated": "2025-08-17T19:21:18.530824",
  "directory": "your-model-directory-path",
  "models": [
    {
      "path": "qwen3-4b-sft",
      "name": "Qwen3 4B SFT",
      "publisher": "Qwen",
      "type": "custom_model",
      "notes": "Fine-tuned on xxx dataset"
    }
  ]
}
```

### Step 3: Review and Edit the Manifest

**Important**: Always review the auto-generated manifest to ensure accurate model information.

1. Open `models_manifest.json` in your model directory
2. Edit the following fields as needed:
   - `name`: Display name shown in vLLM-CLI
   - `publisher`: Organization or author
   - `type`: Model type (model, custom_model, lora_adapter)
   - `notes`: Optional description

### Step 4: Serve Your Custom Model

1. In vLLM-CLI, select one of the **Serving** options from the main menu

2. Your custom models will appear in the model selection list based on the provider defined in the manifest:
   ```
   [?] Select Qwen Model (2 available):
   >  Qwen3 4B SFT (1.32 GB)  # This is the model from your custom directory
      Qwen3-32B (61.04 GB)
   ← Back
   ```

3. Select your custom model

4. Choose a serving profile or configure manually

5. The model will be served with the full path to the custom directory

## Directory Structure Examples

### Example 1: Single Model Directory

```
/home/user/my-model/
├── models_manifest.json     # Auto-generated manifest
├── config.json              # Model configuration
├── model.safetensors       # Model weights
├── tokenizer.json          # Tokenizer
└── tokenizer_config.json   # Tokenizer config
```

**Usage**: Add `/home/user/my-model` as a custom directory

### Example 2: Multiple Models Directory

```
/home/user/models/
├── models_manifest.json           # Manifest for all models
├── qwen3-4b-sft/
│   ├── config.json
│   └── model.safetensors
├── qwen3-7b-sft/
│   ├── config.json
│   └── model.safetensors
└── gemma-custom/
    ├── config.json
    └── model.safetensors
```


### Example 3: LoRA Adapters Directory

```
/home/user/lora-adapters/
├── models_manifest.json
├── finance-lora/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── medical-lora/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

**Usage**: Add as custom directory, LoRAs will be auto-detected

## Serving Configuration

### Quick Serve with Default Settings

```bash
# From command line
vllm-cli serve /home/user/models/qwen3-4b-sft --profile standard
```

## Troubleshooting

### Model Not Appearing in List

1. **Check directory was added successfully**:
   - Go to Model Manager → View Model Directories
   - Verify your directory is listed

2. **Verify model structure**:
   - Ensure `config.json` exists
   - Check for model weight files
   - Confirm directory permissions

3. **Review manifest**:
   - Check `models_manifest.json` was generated
   - Verify model entry exists in manifest

4. **Clear cache**:
   - Go to **Model Management** → **Clear Cache**

### Serving Fails to Start

1. **Check model compatibility**:
   - Verify model architecture is supported by vLLM
   - Ensure CUDA/GPU requirements are met

2. **Review error logs**:
   - Check vLLM-CLI logs in the monitoring view
   - Look for specific error messages

3. **Validate model files**:
   ```python
   # Test model loading
   from transformers import AutoModel
   model = AutoModel.from_pretrained("/path/to/model")
   ```

### Manifest Issues

1. **Regenerate manifest**:
   Remove the directory from vLLM-CLI and add it again
