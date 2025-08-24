# Multi-Model Proxy Server (Experimental)

The Multi-Model Proxy is an experimental feature that enables serving multiple language models through a single unified API endpoint, with dynamic model management, efficient GPU resource distribution through sleep/wake functionality, and centralized management.

## Overview

The proxy server acts as a smart router that:
- Routes requests to multiple vLLM instances based on model name
- **NEW**: Dynamically adds/removes models without proxy restart
- **NEW**: Manages GPU memory through model sleep/wake functionality
- Provides a single OpenAI-compatible endpoint for all models
- Tracks request statistics and health status with enhanced state monitoring

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

## Dynamic Model Management (NEW)

The proxy now supports adding and removing models at runtime without restarting:

### Adding Models Dynamically
While the proxy is running, you can:
1. From the proxy management menu, select **"Add new model"**
2. Configure the model settings (same as initial setup)
3. The model will be automatically registered and activated
4. Existing models continue serving requests uninterrupted

### Removing Models
1. From the proxy management menu, select **"Remove model"**
2. Choose the model to remove from the list
3. The model will be gracefully stopped and unregistered
4. Other models remain unaffected

### Model Registration Lifecycle
The system automatically handles:
- **Pre-registration**: Models are registered before fully ready (shown as "pending")
- **Verification**: Continuous health checks verify model availability
- **Activation**: Models become available once verified
- **Error Recovery**: Failed models can be restarted without affecting others

## Model Sleep/Wake for GPU Memory Management (NEW)

Models can now be put to "sleep" to free GPU memory while keeping their ports active:

### Sleep Modes
- **Sleep Level 1 (CPU Offload)**:
  - Moves model weights to CPU RAM
  - Frees most GPU memory
  - Faster wake-up time
  - Ideal for models that need quick reactivation

- **Sleep Level 2 (Full Discard)**:
  - Completely frees GPU memory
  - Model weights are discarded
  - Slower wake-up (full reload required)
  - Maximum memory savings

### Using Sleep/Wake

#### Interactive UI
1. While monitoring the proxy, select **"Manage models"**
2. Choose a model and select **"Put to sleep"**
3. Select the desired sleep level
4. The UI shows real-time progress (e.g., "Sleep completed in 17.89 seconds")
5. Memory savings are displayed (e.g., "Freed 90.40 GiB, 3.30 GiB still in use")


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
  - Server health status with state indicators

- **Manage models** (NEW) - Dynamic model control
  - Add new models to running proxy
  - Remove models without affecting others
  - Put models to sleep or wake them up
  - View detailed model states and memory usage

### Server Control
- **Stop all servers** - Gracefully shutdown proxy and all models
- **Back to main menu** - Proxy continues running in background

## Using the Proxy

Once running, access your models through proxy port (default: 8000) and make requests to the proxy endpoint.

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
State: Running State: Sleeping State: Running
```

### Model Lifecycle
1. **Pre-registration** → Model allocated resources, shown as "pending"
2. **Verification** → Health checks confirm model is ready
3. **Activation** → Model available for serving requests
4. **Runtime States**:
   - Running: Actively serving
   - Sleeping: GPU memory freed, port active
   - Starting: Initializing or waking
   - Stopped: Gracefully stopped
5. **Dynamic Management** → Add/remove/sleep/wake without proxy restart

## Known Limitations
Currently, the proxy server only supports one model per GPU.

## Troubleshooting

- **Models not responding**: Check individual model logs via monitoring
- **Port conflicts**: Ensure each model has a unique port
- **Proxy not accessible**: Check firewall settings for proxy port

For additional help, please report issues on GitHub.
