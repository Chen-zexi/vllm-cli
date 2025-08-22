# Multi-Model Proxy Server (Experimental)

The Multi-Model Proxy is an experimental feature that enables serving multiple language models through a single unified API endpoint, with efficient GPU resource distribution and centralized management.

## Overview

The proxy server acts as a smart router that:
- Routes requests to multiple vLLM instances based on model name
- Manages GPU allocation across different models
- Provides a single OpenAI-compatible endpoint for all models
- Tracks request statistics and health status
- Handles model lifecycle management

## Getting Started

### Interactive Mode (Recommended)

Access the proxy through the main menu:

```bash
vllm-cli
# Navigate to: Main Menu → Multi-Model Proxy (Exp)
```

You'll see two options:
- **Start from saved configuration** - Quick start with previously saved configs
- **Configure new proxy** - Create a new proxy setup

### Configuration Workflow

#### Step 1: Proxy Server Settings
- **Host**: Network interface to bind (default: 0.0.0.0)
- **Port**: Proxy server port (default: 8000)
- **CORS**: Enable/disable CORS support
- **Metrics**: Enable/disable metrics endpoint
- **Request Logging**: Enable/disable request logging

#### Step 2: Model Configuration
For each model, configure:

1. **Model Selection**
   - Use saved shortcut (if available)
   - Select from local models
   - Serve with LoRA adapters
   - Use HuggingFace model (auto-download)

2. **Model Alias** (Optional)
   - Add custom name for easier API access

3. **GPU Assignment**
   - Interactive GPU selector with space to toggle
   - Shows GPU name, memory, and current utilization
   - Warning: GPU selection overrides profile parallelism settings

4. **Port Assignment**
   - Automatic port assignment (8001, 8002, etc.)
   - Or specify custom port

5. **Profile Selection** (if not using shortcut)
   - Choose from configured profiles
   - Shortcut users skip this (profile included)

Add multiple models as needed, then select "Done configuring models".<br>
At the moment, you can only assign one model to one GPU.

#### Step 3: Save Configuration (Optional)
After configuration, choose:
- **Save and start now** - Save config and launch proxy
- **Save for later use** - Save without starting
- **Start without saving** - One-time use
- **Cancel** - Abort setup

## Managing Running Proxy

Once started, the proxy management interface offers:

### Monitoring Options
- **Monitor proxy logs** - View proxy server logs and statistics
  - Shows registered backends with status
  - Request counts per model
  - GPU utilization
  - Real-time proxy logs

- **Monitor model logs** - View individual model engine logs or overview (combined logs)
  - Select specific model to monitor
  - Real-time log streaming
  - Server health status

### Server Control
- **Stop all servers** - Gracefully shutdown proxy and all models
- **Back to main menu** - Proxy continues running in background

## Using the Proxy

Once running, access your models through proxy port (default: 8000) and make requests to the proxy endpoint.

## Key Features

### Intelligent Routing
- Automatic request routing based on model name
- Support for model aliases
- Health-aware routing (skip unhealthy backends)

### Resource Management
- Flexible GPU assignment per model
- Automatic conflict detection for GPU assignments

### Monitoring & Observability
- Real-time request tracking
- Per-model request statistics
- Unified log viewing
- GPU utilization monitoring
- Backend health status

### Configuration Management
- Save and reuse proxy configurations
- Quick start from saved configs
- Export/import configurations

## Architecture

```
    Client Applications
            │
            ▼
    ┌───────────────────┐
    │   Proxy Server    │
    │   (Port 8000)     │
    └───────┬───────────┘
            │
    ┌───────┴──────┬──────────────┐
    ▼              ▼              ▼
Model A        Model B         Model C
Port: 8001     Port: 8002      Port: 8003
GPU: 0         GPU: 1          GPU: 2,3
```

## Known Limitations
Currently, the proxy server only supports one model per GPU.

## Troubleshooting

- **Models not responding**: Check individual model logs via monitoring
- **Port conflicts**: Ensure each model has a unique port
- **Proxy not accessible**: Check firewall settings for proxy port

For additional help, please report issues on GitHub.
