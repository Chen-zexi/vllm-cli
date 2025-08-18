# Model Discovery Flow Documentation

## Overview
This document explains what determines if models are displayed in different contexts within the vLLM-CLI ecosystem.

## Three Display Contexts

### 1. `hf-model-tool -l` (External Tool)
**Direct directory scanning with its own caching**

```
hf-model-tool -l
    └── hf_model_tool/__main__.py::handle_cli_list()
        └── cache.py::scan_all_directories()
            ├── ConfigManager.get_all_directories_with_types()
            │   ├── Default HF cache: ~/.cache/huggingface/hub
            │   ├── Custom directories from config
            │   └── Ollama directories (if enabled)
            └── For each directory:
                └── cache.py::get_items()
                    └── AssetDetector.detect_asset()
                        ├── Detect HF models (config.json + model files)
                        ├── Detect custom models
                        ├── Detect Ollama models (manifest files)
                        └── Detect GGUF models
```

**Key factors:**
- Scans ALL configured directories
- Has its own registry cache (~/.config/hf-model-tool/registry_cache.json)
- Shows ALL discovered assets (models, LoRA, datasets)
- Ollama scanning controlled by config.json `scan_ollama` flag

### 2. `vllm-cli models` (CLI Command)
**Uses vLLM-CLI's model manager with caching**

```
vllm-cli models
    └── cli/handlers.py::handle_models()
        └── models/manager.py::list_available_models()
            ├── Check ModelCache (30s TTL)
            │   └── If cached and valid → return cached
            └── If not cached or expired:
                └── models/discovery.py::scan_for_models()
                    └── Try: hf_model_tool.get_registry()
                        ├── registry.scan_all() [uses hf-model-tool's cache]
                        ├── Returns: models + custom_models + ollama_models + gguf_models
                        └── Filters based on hf-model-tool config
```

**Key factors:**
- Uses TWO layers of caching:
  1. vLLM-CLI's ModelCache (30 second TTL)
  2. hf-model-tool's registry cache
- Shows only model-type assets (no datasets)
- Respects hf-model-tool's Ollama scanning settings

### 3. Serving Menu Model Selection (Interactive UI)
**Same as vllm-cli models but with UI filtering**

```
Serving Menu → "Serve with Profile"
    └── ui/model_manager.py::select_model()
        └── models/manager.py::list_available_models()
            └── [Same flow as vllm-cli models]
        Then groups by provider:
            ├── Groups Ollama models together
            ├── Groups by publisher field
            └── Separates "local" models
```

**Key factors:**
- Same discovery as `vllm-cli models`
- Additional UI grouping/filtering by provider
- Special handling for Ollama models (warnings)

## Caching Hierarchy

### Level 1: hf-model-tool Registry Cache
- **Location:** `~/.config/hf-model-tool/registry_cache.json`
- **TTL:** 5 minutes (hardcoded)
- **Cleared by:**
  - Manual deletion
  - `registry.scan_all(force=True, incremental=False)`
  - Time expiration

### Level 2: vLLM-CLI Model Cache
- **Location:** In-memory (ModelCache instance)
- **TTL:** 30 seconds (configurable)
- **Cleared by:**
  - `refresh=True` parameter
  - Time expiration
  - "Refresh Model Cache" menu option

## Why Models May Not Appear

### Common Issues:

1. **Ollama Models Not Showing:**
   - Check: `hf-model-tool` config has `scan_ollama: true`
   - Check: Ollama directories are configured or defaults exist
   - Fix: Toggle Ollama scanning in Settings → Model Directories

2. **Recently Added Models Not Showing:**
   - Cause: Multiple cache layers not expired
   - Fix: Use "Refresh Model Cache" in Model Management menu
   - Fix: Wait 30s (vLLM cache) + 5min (registry cache)

3. **Models Show in `hf-model-tool -l` but Not in vLLM-CLI:**
   - Cause: Registry cache is stale
   - Fix: The improved refresh_cache() now clears both caches

4. **Custom Models Not Detected:**
   - Check: Directory is added to hf-model-tool config
   - Check: Models have proper structure (config.json, model files)
   - Fix: Add directory via Settings → Model Directories

## Configuration Files

### hf-model-tool Configuration
**Location:** `~/.config/hf-model-tool/config.json`
```json
{
  "cache_dir": "~/.cache/huggingface/hub",
  "custom_dirs": ["/path/to/models"],
  "scan_ollama": true,
  "ollama_directories": []  // Uses defaults if empty
}
```

### vLLM-CLI Configuration
**Location:** `~/.config/vllm-cli/config.yaml`
```yaml
# Doesn't control model discovery
# Only stores server configs and UI preferences
```

## Model Detection Rules

### HuggingFace Models
**Required files:**
- `config.json` (model configuration)
- One of: `*.safetensors`, `*.bin`, `pytorch_model.bin`

### Ollama Models
**Required structure:**
- Manifest: `/manifests/registry.ollama.ai/library/{model}/{tag}`
- Blobs: `/blobs/sha256:*`
- Detected only if `scan_ollama: true`

### Custom Models
**Required files:**
- `config.json` OR `params.json`
- Model weights in any format

### GGUF Models
**Required files:**
- `*.gguf` files
- Automatically detected in Ollama directories

## Refresh Flow

When "Refresh Model Cache" is clicked:

```python
ModelManager.refresh_cache()
    ├── 1. Clear hf-model-tool cache files
    │   ├── ~/.config/hf-model-tool/registry_cache.json
    │   └── Other potential cache locations
    ├── 2. Clear registry in-memory data
    │   ├── registry.models.clear()
    │   ├── registry.ollama_models.clear()
    │   └── registry._last_scan_time = 0
    ├── 3. Force registry rescan
    │   └── registry.scan_all(force=True, incremental=False)
    ├── 4. Clear vLLM-CLI cache
    │   └── self.cache.clear_cache()
    └── 5. Fetch fresh models
        └── list_available_models(refresh=True)
```

## Summary

**Model visibility is determined by:**

1. **Directory Configuration** - Which directories are scanned
2. **Asset Detection** - Whether files match detection patterns
3. **Cache State** - Whether caches are fresh or stale
4. **Ollama Settings** - Whether Ollama scanning is enabled
5. **Filter Context** - What the specific UI/command chooses to show

**To ensure all models appear:**
1. Add all model directories to hf-model-tool
2. Enable Ollama scanning if using Ollama models
3. Use "Refresh Model Cache" after adding new models
4. Wait for cache TTLs to expire (or force refresh)
