# Using vLLM CLI with Docker

This guide explains how to use vLLM CLI with the official vLLM Docker images.

## Prerequisites

- Docker installed with NVIDIA runtime support
- NVIDIA GPU with CUDA support
- Docker Compose (optional, for easier management)

## Quick Start

### Method 1: Using the Docker Run Script

The simplest way to run vLLM with Docker:

```bash
# Run a model with default settings
./docker-run.sh facebook/opt-125m

# Run with custom port and GPU selection
./docker-run.sh -p 8080 -g 0,1 meta-llama/Llama-2-7b-hf

# Use a custom models directory
./docker-run.sh -m /path/to/models local-model --max-model-len 4096

# Use a different Docker image
./docker-run.sh -i vllm/vllm-openai:v0.5.0 facebook/opt-125m
```

### Method 2: Using Docker Compose

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` to configure your model and settings:
```env
MODEL_NAME=meta-llama/Llama-2-7b-hf
VLLM_PORT=8000
HF_TOKEN=your_token_here
VLLM_EXTRA_ARGS=--max-model-len 4096
```

3. Start the server:
```bash
# Start vLLM server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the server
docker-compose down
```

### Method 3: Direct Docker Commands

Run vLLM directly with Docker:

```bash
docker run --rm -it \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  --shm-size=16gb \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model facebook/opt-125m \
  --host 0.0.0.0 \
  --port 8000
```

## Configuration

### Environment Variables

- `VLLM_DOCKER_IMAGE`: Docker image to use (default: `vllm/vllm-openai:latest`)
- `MODEL_NAME`: Model to serve
- `VLLM_PORT`: Port to expose (default: 8000)
- `HF_TOKEN`: HuggingFace token for private models
- `MODELS_DIR`: Custom models directory to mount
- `VLLM_EXTRA_ARGS`: Additional arguments for vLLM server

### Volume Mounts

The following directories are mounted by default:

- `~/.cache/huggingface`: HuggingFace model cache
- `~/.config/vllm-cli`: vLLM CLI configuration (read-only)
- `/models`: Custom models directory (optional)

### Using vLLM CLI Configuration

Your existing vLLM CLI profiles and settings can be used with Docker. The configuration directory is automatically mounted into the container.

## Advanced Usage

### Using Custom Models

To serve models from a custom directory:

1. Place your models in a local directory
2. Mount the directory when running Docker:

```bash
./docker-run.sh -m /path/to/models your-model-name
```

Or with docker-compose, set in `.env`:
```env
MODELS_DIR=/path/to/models
MODEL_NAME=your-model-name
```

### Multi-GPU Configuration

```bash
# Use specific GPUs
./docker-run.sh -g 0,1 model-name

# Or with docker-compose
docker run --gpus '"device=0,1"' ...
```

### Using Different vLLM Versions

```bash
# Specify a specific version
./docker-run.sh -i vllm/vllm-openai:v0.5.0 model-name

# Or in .env for docker-compose
VLLM_DOCKER_IMAGE=vllm/vllm-openai:v0.5.0
```

### Monitoring Container

```bash
# List running vLLM containers
docker ps --filter ancestor=vllm/vllm-openai

# View container logs
docker logs -f <container-id>

# Monitor GPU usage
nvidia-smi -l 1
```

## Integration with vLLM CLI

The Docker integration is designed to work seamlessly with vLLM CLI:

1. **Profile Support**: Your configured profiles are available in Docker
2. **Model Discovery**: Models detected by vLLM CLI can be served in Docker
3. **Configuration Sync**: Settings are shared between native and Docker deployments

### Using Docker from vLLM CLI

```python
from vllm_cli.docker import DockerManager

# Initialize Docker manager
docker_mgr = DockerManager()

# Check Docker availability
ready, message = docker_mgr.check_requirements()
if ready:
    # Run server in Docker
    process = docker_mgr.run_server(
        model="facebook/opt-125m",
        port=8000,
        vllm_args={"max_model_len": 4096}
    )
```

## Troubleshooting

### Docker not found
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

### NVIDIA runtime not available
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Permission denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Out of memory
Adjust the shared memory size:
```bash
--shm-size=32gb  # Increase from default 16gb
```

## Best Practices

1. **Resource Management**: Set appropriate GPU memory utilization
2. **Model Caching**: Mount HuggingFace cache to avoid re-downloading
3. **Configuration**: Use `.env` files for easy configuration management
4. **Monitoring**: Always monitor GPU usage and container logs
5. **Cleanup**: Remove unused containers and images regularly

```bash
# Clean up stopped containers
docker container prune

# Remove unused images
docker image prune
```
