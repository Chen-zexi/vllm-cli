# Model Discovery Quick Reference

## Quick Diagnosis Tree

```
Model not showing?
│
├── Is it an Ollama model?
│   ├── YES → Check: Settings → Model Directories → Toggle Ollama Scanning
│   └── NO → Continue ↓
│
├── Does it show in `hf-model-tool -l`?
│   ├── NO → Model not detected:
│   │   ├── Add directory: Settings → Model Directories → Add Directory
│   │   └── Check model structure (needs config.json + weights)
│   └── YES → Cache issue:
│       └── Model Management → Refresh Model Cache
│
└── Still not working?
    └── Force complete refresh:
        1. Close vLLM-CLI
        2. Run: rm ~/.config/hf-model-tool/registry_cache.json
        3. Restart vLLM-CLI
        4. Model Management → Refresh Model Cache
```

## Command Comparison

| Command | Source | Cache | Shows |
|---------|--------|-------|-------|
| `hf-model-tool -l` | Direct scan | Registry (5min) | ALL assets |
| `vllm-cli models` | Via registry | Model (30s) + Registry (5min) | Models only |
| Serving menu | Via registry | Model (30s) + Registry (5min) | Models grouped by provider |

## Key Files

| File | Purpose | Clear Command |
|------|---------|--------------|
| `~/.config/hf-model-tool/config.json` | Controls what directories are scanned | Edit manually |
| `~/.config/hf-model-tool/registry_cache.json` | hf-model-tool's cache | `rm` this file |
| In-memory ModelCache | vLLM-CLI's 30s cache | "Refresh Model Cache" |


## Cache TTLs

- **hf-model-tool registry:** 5 minutes
- **vLLM-CLI model cache:** 30 seconds
- **After "Refresh Model Cache":** Immediate

## Model Detection Requirements

| Model Type | Required Files |
|------------|---------------|
| HuggingFace | `config.json` + `*.safetensors` or `*.bin` |
| Ollama | Manifest in `/manifests/` + blobs |
| GGUF | `*.gguf` file |
| Custom | `config.json` or `params.json` + weights |

## Debugging Commands

```bash
# Check what hf-model-tool sees
hf-model-tool -l | grep "model_name"

# Check Ollama models directly
ls /usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/

# Check vLLM-CLI's view
vllm-cli models | grep "model_name"

# Check hf-model-tool config
cat ~/.config/hf-model-tool/config.json

# Force clean state
rm ~/.config/hf-model-tool/registry_cache.json
```
